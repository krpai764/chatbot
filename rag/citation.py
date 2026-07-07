"""
Citation extraction and formatting module for the DataIntern RAG Engine.

Provides utilities to extract citations from structured LLM responses,
build citations from vector-store chunk metadata, merge/deduplicate them,
and produce human-readable citation strings.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extracts, merges, and formats citations from multiple sources."""

    # Keys we care about when comparing citations for deduplication.
    _CITATION_KEYS = ("file", "sheet", "page", "rows")

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def extract_from_response(self, llm_response: dict) -> list[dict]:
        """Extract citations embedded in a structured LLM response.

        The LLM is instructed to return a ``"citations"`` key containing a
        list of objects.  This method safely retrieves that list and
        normalises each entry.

        Args:
            llm_response: The parsed JSON dict returned by ``GeminiLLM.generate_json``.

        Returns:
            A list of citation dicts, each with at least a ``"file"`` key.
        """
        raw_citations = llm_response.get("citations", [])
        if not isinstance(raw_citations, list):
            logger.warning("Expected 'citations' to be a list, got %s", type(raw_citations))
            return []

        normalised: list[dict] = []
        for entry in raw_citations:
            if not isinstance(entry, dict):
                logger.debug("Skipping non-dict citation entry: %s", entry)
                continue
            citation = self._normalise_citation(entry)
            if citation.get("file"):
                normalised.append(citation)
            else:
                logger.debug("Skipping citation without 'file': %s", entry)

        logger.info("Extracted %d citation(s) from LLM response.", len(normalised))
        return normalised

    def extract_from_metadata(self, chunk_metadatas: list[dict]) -> list[dict]:
        """Build citations from chunk metadata returned by the vector store.

        Chunks are grouped by filename so that each source file appears at
        most once, with aggregated sheet / page / row information.

        Args:
            chunk_metadatas: A list of metadata dicts attached to retrieved
                chunks (typically containing ``filename``, ``sheet``,
                ``page``, ``rows``, etc.).

        Returns:
            A deduplicated list of citation dicts grouped by source file.
        """
        if not chunk_metadatas:
            return []

        groups: dict[str, dict] = defaultdict(lambda: {
            "file": "",
            "sheets": set(),
            "pages": set(),
            "rows": [],
        })

        for meta in chunk_metadatas:
            if not isinstance(meta, dict):
                continue
            filename = meta.get("filename") or meta.get("file") or meta.get("source", "")
            if not filename:
                continue

            bucket = groups[filename]
            bucket["file"] = filename

            sheet = meta.get("sheet") or meta.get("sheet_name")
            if sheet:
                bucket["sheets"].add(str(sheet))

            page = meta.get("page") or meta.get("page_number")
            if page is not None:
                bucket["pages"].add(int(page))

            rows = meta.get("rows") or meta.get("row_range")
            if rows:
                bucket["rows"].append(str(rows))

        citations: list[dict] = []
        for bucket in groups.values():
            citation: dict = {"file": bucket["file"]}
            if bucket["sheets"]:
                citation["sheet"] = ", ".join(sorted(bucket["sheets"]))
            if bucket["pages"]:
                citation["page"] = sorted(bucket["pages"])
            if bucket["rows"]:
                citation["rows"] = ", ".join(bucket["rows"])
            citations.append(citation)

        logger.info("Built %d citation(s) from chunk metadata.", len(citations))
        return citations

    # ------------------------------------------------------------------
    # Merging & deduplication
    # ------------------------------------------------------------------

    def merge_citations(
        self,
        llm_citations: list[dict],
        metadata_citations: list[dict],
    ) -> list[dict]:
        """Merge and deduplicate citations from both sources.

        LLM-provided citations take precedence; metadata-derived citations
        are appended only when they introduce a new source file not already
        covered.

        Args:
            llm_citations: Citations extracted from the LLM response.
            metadata_citations: Citations built from chunk metadata.

        Returns:
            A merged, deduplicated list of citation dicts.
        """
        seen_files: set[str] = set()
        merged: list[dict] = []

        for cit in llm_citations:
            key = self._citation_key(cit)
            if key not in seen_files:
                seen_files.add(key)
                merged.append(cit)

        for cit in metadata_citations:
            key = self._citation_key(cit)
            if key not in seen_files:
                seen_files.add(key)
                merged.append(cit)

        logger.info(
            "Merged citations: %d LLM + %d metadata → %d unique.",
            len(llm_citations),
            len(metadata_citations),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_citations(self, citations: list[dict]) -> str:
        """Pretty-print citations for display.

        Each citation is rendered on its own line, e.g.::

            📄 deals.csv (sheet: Deals, rows: 1-10)
            📄 account_review.docx (page: 2)

        Args:
            citations: A list of citation dicts.

        Returns:
            A formatted multi-line string, or an empty string when there are
            no citations to display.
        """
        if not citations:
            return ""

        lines: list[str] = []
        for cit in citations:
            parts: list[str] = []
            filename = cit.get("file", "unknown")

            sheet = cit.get("sheet")
            if sheet:
                parts.append(f"sheet: {sheet}")

            page = cit.get("page")
            if page is not None:
                if isinstance(page, list):
                    parts.append(f"page: {', '.join(str(p) for p in page)}")
                else:
                    parts.append(f"page: {page}")

            rows = cit.get("rows")
            if rows:
                parts.append(f"rows: {rows}")

            detail = f" ({', '.join(parts)})" if parts else ""
            lines.append(f"📄 {filename}{detail}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_citation(entry: dict) -> dict:
        """Return a citation dict with only the recognised keys."""
        return {
            k: entry[k]
            for k in ("file", "sheet", "page", "rows")
            if k in entry and entry[k] is not None
        }

    @staticmethod
    def _citation_key(citation: dict) -> str:
        """Create a hashable key for deduplication."""
        file_val = citation.get("file", "")
        sheet_val = citation.get("sheet", "")
        page_val = str(citation.get("page", ""))
        return f"{file_val}|{sheet_val}|{page_val}"
