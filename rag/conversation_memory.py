"""
Conversation memory module for the DataIntern RAG Engine.

Maintains a sliding window of recent Q&A turns so that follow-up questions
can be resolved in context, and provides LLM-powered query expansion for
ambiguous references.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.llm import GeminiLLM

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Tracks conversation history and supports contextual query expansion.

    Attributes:
        max_turns: Maximum number of Q&A turns to retain.
    """

    def __init__(self, max_turns: int = 10) -> None:
        """Initialise an empty conversation memory.

        Args:
            max_turns: The number of recent turns to keep before pruning
                older entries.
        """
        self.max_turns = max_turns
        self._history: list[dict] = []
        logger.info("ConversationMemory initialised (max_turns=%d).", max_turns)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_turn(
        self,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> None:
        """Record a question-answer turn.

        Args:
            question: The user's question.
            answer: The assistant's answer.
            metadata: Optional extra info such as ``filters``, ``dataset``,
                or ``chart_type`` used during this turn.
        """
        turn: dict = {
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
        }
        self._history.append(turn)

        # Prune if we exceed the window.
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns :]

        logger.debug(
            "Turn added (total=%d). Q: %s",
            len(self._history),
            question[:80],
        )

    def get_history(self) -> list[dict]:
        """Return conversation history in chat-message format.

        Returns:
            A list of ``{"role": "user" | "assistant", "content": str}``
            dicts in chronological order.
        """
        messages: list[dict] = []
        for turn in self._history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def get_context_prompt(self) -> str:
        """Format recent history into a prompt string for context injection.

        Returns:
            A multi-line string summarising the last few exchanges, or an
            empty string when there is no history.
        """
        if not self._history:
            return ""

        lines: list[str] = ["Previous conversation:"]
        for turn in self._history:
            lines.append(f"User: {turn['question']}")
            # Truncate very long answers to keep context manageable.
            answer_snippet = turn["answer"][:500]
            if len(turn["answer"]) > 500:
                answer_snippet += "…"
            lines.append(f"Assistant: {answer_snippet}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str, llm: GeminiLLM) -> str:
        """Use Gemini to expand an ambiguous follow-up question.

        If the query already appears self-contained the LLM is instructed to
        return it unchanged.  Otherwise, conversation history is used to
        resolve pronouns and implicit references.

        Examples:
            * *"What about Europe?"* after asking about revenue by region →
              *"What is the revenue for the Europe region?"*

        Args:
            query: The user's latest question.
            llm: A ``GeminiLLM`` instance used for expansion.

        Returns:
            The (possibly rewritten) query string.
        """
        if not self._history:
            logger.debug("No history available; returning query unchanged.")
            return query

        context = self.get_context_prompt()
        prompt = (
            f"{context}\n\n"
            f"Latest user question: {query}\n\n"
            "If the latest question is a follow-up that relies on previous "
            "conversation context (e.g. pronouns like 'it', 'that', 'those', "
            "or short phrases like 'What about Europe?'), rewrite it as a "
            "fully self-contained question.\n"
            "If the question is already self-contained, return it exactly as-is.\n"
            "Return ONLY the rewritten question, nothing else."
        )

        try:
            expanded = llm.generate(prompt).strip()
            if expanded:
                logger.info(
                    "Query expanded: '%s' → '%s'",
                    query[:60],
                    expanded[:60],
                )
                return expanded
        except Exception as exc:
            logger.warning("Query expansion failed (%s); using original query.", exc)

        return query

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_last_dataset(self) -> str | None:
        """Return the dataset/filename used in the most recent turn.

        Returns:
            The filename string, or ``None`` if no dataset was recorded.
        """
        for turn in reversed(self._history):
            dataset = turn.get("metadata", {}).get("dataset")
            if dataset:
                return dataset
        return None

    def get_last_filters(self) -> dict | None:
        """Return the filters applied in the most recent turn.

        Returns:
            A dict of column→value filters, or ``None`` if none were recorded.
        """
        for turn in reversed(self._history):
            filters = turn.get("metadata", {}).get("filters")
            if filters:
                return filters
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        logger.info("Conversation memory cleared.")
