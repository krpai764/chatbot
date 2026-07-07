"""Query engine – main orchestrator for the DataIntern RAG Engine.

Ties together retrieval, LLM generation, chart detection, chart
generation, conversation memory, and citation extraction into a single
``QueryEngine.ask()`` entry point.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.citation import CitationExtractor
from rag.chart_detector import ChartDetector
from rag.chart_generator import ChartGenerator
from rag.config import Config
from rag.conversation_memory import ConversationMemory
from rag.data_executor import DataQueryExecutor
from rag.llm import GeminiLLM, RAG_SYSTEM_INSTRUCTION
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


class QueryResult:
    """Container for the output of a single query.

    Attributes:
        answer:           The natural-language answer string.
        citations:        List of citation dicts.
        citations_text:   Human-readable citation block.
        retrieved_chunks: Raw retrieval results (list of dicts with
                          text, metadata, similarity_score).
        chart_json:       Plotly figure JSON dict, or ``None``.
        chart_params:     Chart parameters dict, or ``None``.
    """

    def __init__(
        self,
        answer: str,
        citations: list[dict],
        citations_text: str,
        retrieved_chunks: list[dict],
        chart_json: dict | None = None,
        chart_params: dict | None = None,
    ) -> None:
        self.answer = answer
        self.citations = citations
        self.citations_text = citations_text
        self.retrieved_chunks = retrieved_chunks
        self.chart_json = chart_json
        self.chart_params = chart_params

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_chunks": [
                {
                    "text": c["text"][:200] + "…" if len(c["text"]) > 200 else c["text"],
                    "metadata": c["metadata"],
                    "similarity_score": c["similarity_score"],
                }
                for c in self.retrieved_chunks
            ],
            "chart_json": self.chart_json,
        }


class QueryEngine:
    """Orchestrates the full RAG pipeline for a user question.

    Workflow per query:
    1. Expand follow-ups using conversation memory.
    2. Retrieve top-K chunks from ChromaDB.
    3. Detect if a chart is requested.
    4. Call Gemini with the retrieved context.
    5. Extract / merge citations.
    6. Generate chart from original dataset (if applicable).
    7. Store the turn in conversation memory.

    Args:
        config:    Application configuration.
        retriever: An initialised Retriever instance.
        llm:       An initialised GeminiLLM instance.
    """

    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        llm: GeminiLLM,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._llm = llm
        self._memory = ConversationMemory(max_turns=10)
        self._citation_extractor = CitationExtractor()
        self._chart_detector = ChartDetector(llm)
        self._chart_generator = ChartGenerator(config.DATA_DIR)
        self._data_executor = DataQueryExecutor(llm, config.DATA_DIR)
        logger.info("QueryEngine initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> QueryResult:
        """Process a user question through the full RAG pipeline.

        Args:
            question: Natural-language question from the user.

        Returns:
            A ``QueryResult`` with the answer, citations, retrieved
            chunks, and optional chart JSON.
        """
        logger.info("=" * 60)
        logger.info("USER QUESTION: %s", question)
        logger.info("=" * 60)

        # 1. Expand follow-up questions using conversation memory.
        expanded_query = self._memory.expand_query(question, self._llm)
        if expanded_query != question:
            logger.info("Expanded query: %s", expanded_query)

        # 2. Retrieve relevant chunks.
        retrieved = self._retriever.retrieve(expanded_query)
        context = self._retriever.format_context(retrieved)
        logger.info("Retrieved %d chunk(s).", len(retrieved))

        # 2b. Run quantitative direct calculation if detected (handles totals/averages/counts on full file)
        data_context = self._data_executor.analyze_and_execute(expanded_query)
        if data_context:
            logger.info("Quantitative data context appended.")
            context = data_context + "\n\n" + context

        # 3. Detect chart intent.
        is_chart = self._chart_detector.is_chart_request(expanded_query)
        chart_json: dict | None = None
        chart_params: dict | None = None

        # 4. Call Gemini with the context.
        llm_prompt = self._build_prompt(expanded_query, context)
        llm_response = self._llm.generate_json(
            llm_prompt,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
        )

        # Extract answer text.
        answer = llm_response.get("answer", llm_response.get("raw_response", ""))

        # 5. Extract and merge citations.
        llm_citations = self._citation_extractor.extract_from_response(llm_response)
        meta_citations = self._citation_extractor.extract_from_metadata(
            [r["metadata"] for r in retrieved]
        )
        merged_citations = self._citation_extractor.merge_citations(
            llm_citations, meta_citations
        )
        citations_text = self._citation_extractor.format_citations(merged_citations)

        # 6. Generate chart if requested.
        if is_chart:
            try:
                conversation_ctx = self._memory.get_context_prompt()
                chart_params = self._chart_detector.detect_chart_params(
                    expanded_query, context, conversation_ctx
                )
                if chart_params:
                    chart_json = self._chart_generator.generate_chart(chart_params)
                    logger.info("Chart generated: %s", chart_params.get("title", ""))
            except Exception as exc:
                logger.error("Chart generation failed: %s", exc, exc_info=True)
                answer += f"\n\n⚠️ Chart generation failed: {exc}"

        # 7. Store in conversation memory.
        turn_meta: dict[str, Any] = {}
        if chart_params:
            turn_meta["dataset"] = chart_params.get("dataset")
            turn_meta["filters"] = chart_params.get("filters")
            turn_meta["chart_type"] = chart_params.get("chart_type")
        self._memory.add_turn(question, answer, metadata=turn_meta)

        result = QueryResult(
            answer=answer,
            citations=merged_citations,
            citations_text=citations_text,
            retrieved_chunks=retrieved,
            chart_json=chart_json,
            chart_params=chart_params,
        )

        logger.info("Query processed successfully.")
        return result

    def clear_memory(self) -> None:
        """Reset the conversation memory."""
        self._memory.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Construct the LLM prompt with retrieved context.

        Args:
            query:   The (possibly expanded) user query.
            context: Formatted context string from the retriever.

        Returns:
            A prompt string ready for Gemini.
        """
        return (
            f"Context from CRM documents:\n"
            f"{'─' * 40}\n"
            f"{context}\n"
            f"{'─' * 40}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question using ONLY the context above. "
            f"Include citations referencing specific files, sheets, pages, or rows."
        )
