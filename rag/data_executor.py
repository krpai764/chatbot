"""Data executor module for the DataIntern RAG Engine.

Provides automated pandas execution of quantitative queries (filtering,
aggregating, grouping) on the original datasets to bypass RAG token limits.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from rag.llm import GeminiLLM

logger = logging.getLogger(__name__)

_DATA_DETECTION_PROMPT = """You are a structured data query analyzer for a CRM system.

Your job is to determine if a user query requires quantitative calculation (sum, count, average, min, max, grouping, or listing records) over the CRM datasets, and extract the exact parameters needed to run a pandas query.

Available datasets and columns:
- deals.csv: deal_id, account_id, company, owner, region, stage, amount_usd, probability_pct, source, created_date, close_date
- contacts.csv: contact_id, name, email, title, account_id, company, phone
- accounts.json: account_id, company, industry, employees, country, annual_revenue_usd
- activities.json: activity_id, deal_id, company, owner, type, date, notes
- crm_workbook.xlsx: multiple sheets (Deals, Contacts, Pipeline, etc.)

=== CRITICAL BUSINESS RULES - Always follow these ===
1. "sales", "revenue", "deals closed", "closed deals" ALWAYS means:
   - dataset: deals.csv
   - filter stage == "Closed Won"
   - target_column: amount_usd
   - date filter uses close_date (NOT created_date)

2. "total sales in [month]" or "sales for [month]" means:
   - filter stage == "Closed Won"
   - filter close_date contains "-MM-" pattern
   - aggregation: sum, target_column: amount_usd

3. Month name to number mapping:
   January=01, February=02, March=03, April=04, May=05, June=06,
   July=07, August=08, September=09, October=10, November=11, December=12

4. "new deals", "deals created", "deals opened" uses created_date (NOT close_date)

5. "employees", "headcount", "team size" refers to accounts.json employees column

6. "activities", "meetings", "calls" refers to activities.json

Provide a JSON response with:
- "is_data_query": boolean (true if query asks for calculations, sums, averages, lists of rows, counts, or filters on rows)
- "dataset": filename to load (e.g. "deals.csv", "accounts.json", "contacts.csv", "activities.json")
- "sheet_name": sheet name if Excel (null otherwise)
- "filters": list of dicts, each with "column", "operator" (one of "==", "!=", ">", "<", "contains", "in_month"), and "value"
- "aggregation": "sum", "count", "mean", "median", "min", "max", "list" (null if not applicable)
- "target_column": column to aggregate or calculate (e.g., "amount_usd", "annual_revenue_usd", or null)
- "group_by": column to group by (null if none)
- "explanation": brief explanation of what calculation is needed

Example 1: "total sales in march" or "what is the total sales in month of march?"
{{
  "is_data_query": true,
  "dataset": "deals.csv",
  "sheet_name": null,
  "filters": [
    {{"column": "stage", "operator": "==", "value": "Closed Won"}},
    {{"column": "close_date", "operator": "contains", "value": "-03-"}}
  ],
  "aggregation": "sum",
  "target_column": "amount_usd",
  "group_by": null,
  "explanation": "Sum amount_usd for Closed Won deals with close_date in March (month 03)"
}}

Example 2: "how many deals were created in January?"
{{
  "is_data_query": true,
  "dataset": "deals.csv",
  "sheet_name": null,
  "filters": [
    {{"column": "created_date", "operator": "contains", "value": "-01-"}}
  ],
  "aggregation": "count",
  "target_column": null,
  "group_by": null,
  "explanation": "Count deals with created_date in January (month 01)"
}}

Return ONLY valid JSON.
"""


class DataQueryExecutor:
    """Detects and executes data queries directly against the datasets using pandas.

    This bypasses chunking limits for math/aggregation questions.
    """

    def __init__(self, llm: GeminiLLM, data_dir: str) -> None:
        self.llm = llm
        self.data_dir = data_dir
        logger.info("DataQueryExecutor initialised with data_dir=%s", data_dir)

    def analyze_and_execute(self, query: str) -> str | None:
        """Analyze the query, execute the pandas operation if it is a data query,

        and return a formatted summary of the result.
        """
        try:
            # 1. Ask Gemini if this is a data query and extract parameters
            analysis = self.llm.generate_json(_DATA_DETECTION_PROMPT + f"\nUser Query: {query}")
            if not analysis.get("is_data_query"):
                return None

            logger.info("Quantitative data query detected: %s", analysis.get("explanation"))
            dataset = analysis.get("dataset")
            if not dataset:
                return None

            # 2. Load dataset
            df = self._load_dataset(dataset, analysis.get("sheet_name"))

            # 3. Apply filters
            filters = analysis.get("filters", [])
            df = self._apply_filters(df, filters)

            # 4. Perform aggregation / extraction
            result_str = self._perform_calculation(df, analysis)
            logger.info("Data query executed successfully: %s", result_str)
            return result_str

        except Exception as exc:
            logger.error("Failed to execute data query: %s", exc, exc_info=True)
            return f"⚠️ Note: Failed to compute direct dataset calculation: {exc}"

    def _load_dataset(self, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".csv":
            return pd.read_csv(filepath)
        elif ext == ".json":
            import json
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                list_values = [v for v in data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    return pd.DataFrame(list_values[0])
                return pd.DataFrame([data])
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(filepath, sheet_name=sheet_name or 0)
        raise ValueError(f"Unsupported file format: {ext}")

    def _apply_filters(self, df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
        for f in filters:
            col = f.get("column")
            op = f.get("operator")
            val = f.get("value")

            if not col or col not in df.columns:
                continue

            if op == "==":
                if df[col].dtype == object:
                    df = df[df[col].astype(str).str.lower() == str(val).lower()]
                else:
                    df = df[df[col] == val]
            elif op == "!=":
                if df[col].dtype == object:
                    df = df[df[col].astype(str).str.lower() != str(val).lower()]
                else:
                    df = df[df[col] != val]
            elif op == ">":
                df = df[df[col] > float(val)]
            elif op == "<":
                df = df[df[col] < float(val)]
            elif op == "contains":
                df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "in_month":
                # Matches YYYY-MM-DD or MM/DD/YYYY strings for the month part
                month_pattern = f"-{str(val).zfill(2)}-"
                df = df[df[col].astype(str).str.contains(month_pattern, na=False)]

        return df

    def _perform_calculation(self, df: pd.DataFrame, params: dict) -> str:
        agg = params.get("aggregation")
        target_col = params.get("target_column")
        group_by = params.get("group_by")
        dataset = params.get("dataset")

        summary = f"Direct Calculation Result from {dataset}:\n"
        summary += f"- Matching rows count: {len(df)}\n"

        if len(df) == 0:
            summary += "- No matching records found for the filters.\n"
            return summary

        if group_by and group_by in df.columns:
            if target_col and target_col in df.columns and agg in ("sum", "mean", "median", "min", "max"):
                grouped = df.groupby(group_by)[target_col].agg(agg)
                summary += f"- Grouped by {group_by} ({agg} of {target_col}):\n"
                for k, v in grouped.items():
                    summary += f"  * {k}: {v:,.2f}\n"
            else:
                grouped = df.groupby(group_by).size()
                summary += f"- Grouped by {group_by} (count):\n"
                for k, v in grouped.items():
                    summary += f"  * {k}: {v}\n"
            return summary

        if agg == "sum" and target_col in df.columns:
            val = df[target_col].sum()
            summary += f"- Total Sum of {target_col}: {val:,.2f}\n"
        elif agg == "mean" and target_col in df.columns:
            val = df[target_col].mean()
            summary += f"- Average (Mean) of {target_col}: {val:,.2f}\n"
        elif agg == "max" and target_col in df.columns:
            val = df[target_col].max()
            summary += f"- Maximum value of {target_col}: {val:,.2f}\n"
        elif agg == "min" and target_col in df.columns:
            val = df[target_col].min()
            summary += f"- Minimum value of {target_col}: {val:,.2f}\n"
        elif agg == "count":
            summary += f"- Record count: {len(df)}\n"
        elif agg == "list" or not agg:
            # Output first 10 rows as a formatted list
            summary += f"- Records (up to 10 shown):\n"
            cols = list(df.columns[:8])  # limit columns for display readability
            for idx, row in df.head(10).iterrows():
                row_str = " | ".join(f"{c}: {row[c]}" for c in cols)
                summary += f"  * {row_str}\n"
            if len(df) > 10:
                summary += f"  * ... and {len(df) - 10} more records\n"

        return summary
