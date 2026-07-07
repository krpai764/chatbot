"""
Chart detection module for the DataIntern RAG Engine.

Determines whether a user query is requesting a visualisation and, if so,
uses Gemini to infer the chart type, dataset, columns, filters, and
aggregation from the query and retrieved context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.llm import GeminiLLM

logger = logging.getLogger(__name__)

# Keywords that signal the user wants a chart / visualisation.
_CHART_KEYWORDS: list[str] = [
    "chart",
    "plot",
    "graph",
    "histogram",
    "visualize",
    "visualise",
    "show",
    "bar chart",
    "pie chart",
    "scatter",
    "line chart",
    "distribution",
]

# Prompt template describing available CRM datasets and their columns.
_CHART_DETECTION_PROMPT = """\
You are a chart parameter extractor for a CRM analytics system.

Given a user query and retrieved context, determine the chart parameters.

Available datasets and their columns:
- deals.csv: deal_id, account_id, company, owner, region, stage, amount_usd, probability_pct, source, created_date, close_date
- contacts.csv: contact_id, name, email, title, account_id, company, phone
- accounts.json: account_id, company, industry, employees, country, annual_revenue_usd
- activities.json: activity_id, deal_id, company, owner, type, date, notes
- crm_workbook.xlsx: multiple sheets (Deals, Contacts, Pipeline, etc.)

Respond with a JSON object containing exactly these keys:
- "chart_type": one of "bar", "line", "pie", "scatter", "histogram"
- "dataset": the filename to load (e.g. "deals.csv")
- "x_column": column for the x-axis
- "y_column": column for the y-axis (omit for histogram)
- "color_column": optional grouping column (null if not applicable)
- "filters": dict of column->value filters (empty dict if none)
- "aggregation": one of "sum", "count", "mean", "median", "min", "max" (null if not applicable)
- "title": a descriptive chart title

User query: {query}

{conversation_section}

Retrieved context:
{context}

Return ONLY valid JSON, no extra text."""


class ChartDetector:
    """Detects chart requests and extracts Plotly chart parameters via Gemini.

    Attributes:
        llm: The ``GeminiLLM`` instance used for parameter extraction.
    """

    def __init__(self, llm: GeminiLLM) -> None:
        """Initialise the chart detector.

        Args:
            llm: A configured ``GeminiLLM`` instance.
        """
        self.llm = llm
        logger.info("ChartDetector initialised.")

    def is_chart_request(self, query: str) -> bool:
        """Check whether *query* is asking for a visualisation.

        Uses simple keyword matching (case-insensitive).

        Args:
            query: The user's natural-language question.

        Returns:
            ``True`` if the query contains any chart-related keyword.
        """
        query_lower = query.lower()
        for keyword in _CHART_KEYWORDS:
            if keyword in query_lower:
                logger.debug("Chart keyword '%s' found in query.", keyword)
                return True
        return False

    def detect_chart_params(
        self,
        query: str,
        retrieved_context: str,
        conversation_context: str = "",
    ) -> dict | None:
        """Use Gemini to infer chart parameters from the query and context.

        Args:
            query: The user's question.
            retrieved_context: Text chunks retrieved from the vector store.
            conversation_context: Optional prior conversation context for
                resolving follow-up requests.

        Returns:
            A dict of chart parameters suitable for ``ChartGenerator``, or
            ``None`` if detection fails.
        """
        conversation_section = ""
        if conversation_context:
            conversation_section = f"Conversation history:\n{conversation_context}"

        prompt = _CHART_DETECTION_PROMPT.format(
            query=query,
            context=retrieved_context,
            conversation_section=conversation_section,
        )

        try:
            result = self.llm.generate_json(prompt)
        except Exception as exc:
            logger.error("Chart parameter detection failed: %s", exc, exc_info=True)
            return None

        # If the LLM returned a fallback raw_response, treat as failure.
        if "raw_response" in result and "chart_type" not in result:
            logger.warning("LLM did not return valid chart params: %s", result)
            return None

        # Validate required keys.
        if "chart_type" not in result or "dataset" not in result:
            logger.warning("Missing required chart param keys: %s", result)
            return None

        # Normalise chart_type.
        valid_types = {"bar", "line", "pie", "scatter", "histogram"}
        if result["chart_type"] not in valid_types:
            logger.warning("Invalid chart_type '%s'; defaulting to 'bar'.", result["chart_type"])
            result["chart_type"] = "bar"

        # Ensure filters is a dict.
        if not isinstance(result.get("filters"), dict):
            result["filters"] = {}

        logger.info("Detected chart params: %s", result)
        return result
