"""Retriever module for the DataIntern RAG Engine.

Orchestrates embedding and vector-store search to retrieve the most
relevant document chunks for a user query, and formats them into a
context string suitable for LLM prompts.
"""

import logging

from rag.embedding import EmbeddingManager
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever that combines embedding and vector search.

    Attributes:
        _embedding_manager: Handles text → vector encoding.
        _vector_store: Handles vector storage and similarity search.
        _top_k: Default number of results to retrieve.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        """Initialise the Retriever.

        Args:
            embedding_manager: An initialised EmbeddingManager instance.
            vector_store: An initialised VectorStore instance.
            top_k: Default number of top results to return per query.
        """
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store
        self._top_k = top_k

        logger.info(
            "Retriever initialised (model=%s, top_k=%d, docs_in_store=%d).",
            embedding_manager.model_name,
            top_k,
            vector_store.get_count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve the most relevant chunks for a query.

        The query is embedded, the vector store is searched, and results
        are returned sorted by descending similarity score.

        Args:
            query: Natural-language query string.
            top_k: Override the default number of results.  Uses the
                instance default when ``None``.

        Returns:
            A list of result dicts, each containing:

            * ``text``  – the chunk text
            * ``metadata`` – the chunk's metadata dict
            * ``similarity_score`` – cosine similarity (1 − distance)
        """
        k = top_k if top_k is not None else self._top_k
        logger.info("Retrieving top-%d results for query: %.100s…", k, query)

        # 1. Embed the query.
        query_embedding: list[float] = self._embedding_manager.embed_query(query)

        # 2. Search the vector store.
        raw_results: dict = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
        )

        # 3. Package results with similarity scores.
        results: list[dict] = []
        for doc, meta, distance in zip(
            raw_results["documents"],
            raw_results["metadatas"],
            raw_results["distances"],
        ):
            # ChromaDB returns cosine *distance*; convert to similarity.
            similarity: float = 1.0 - distance
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": round(similarity, 6),
                }
            )

        # Sort by similarity descending (highest first).
        results.sort(key=lambda r: r["similarity_score"], reverse=True)

        logger.info(
            "Retrieved %d result(s). Top score: %.4f, lowest: %.4f.",
            len(results),
            results[0]["similarity_score"] if results else 0.0,
            results[-1]["similarity_score"] if results else 0.0,
        )
        return results

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a context string for an LLM prompt.

        Each chunk is rendered as::

            [Source N] (file: X, sheet: Y, rows: Z, score: 0.85)
            Chunk text here…

        Metadata keys ``source_file``, ``sheet_name``, and ``row_range``
        are used when present; missing keys are gracefully omitted.

        Args:
            results: List of result dicts as returned by :meth:`retrieve`.

        Returns:
            A single formatted context string.
        """
        if not results:
            return ""

        sections: list[str] = []
        for idx, result in enumerate(results, start=1):
            meta = result.get("metadata", {})
            score = result.get("similarity_score", 0.0)

            # Build the source descriptor from available metadata.
            parts: list[str] = []
            if meta.get("filename"):
                parts.append(f"file: {meta['filename']}")
            if meta.get("sheet"):
                parts.append(f"sheet: {meta['sheet']}")
            if meta.get("row_range"):
                parts.append(f"rows: {meta['row_range']}")
            parts.append(f"score: {score:.2f}")

            header = f"[Source {idx}] ({', '.join(parts)})"
            text = result.get("text", "")

            sections.append(f"{header}\n{text}")

        context = "\n\n".join(sections)

        logger.debug(
            "Formatted context with %d source(s) (%d chars).",
            len(sections),
            len(context),
        )
        return context

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self._embedding_manager.model_name!r}, "
            f"top_k={self._top_k}, "
            f"docs={self._vector_store.get_count()})"
        )
