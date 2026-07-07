"""
utils.py – Shared utility helpers for the DataIntern RAG Engine.

Provides logging setup, file discovery, text cleaning, and
deterministic chunk-ID generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger and return a module-level logger.

    Sets a uniform format across all modules and avoids duplicate
    handlers when called more than once.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``).

    Returns:
        A ``logging.Logger`` instance named ``"dataintern"``.
    """
    logger = logging.getLogger("dataintern")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------

def discover_files(data_dir: str, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively discover files matching *extensions* under *data_dir*.

    Hidden files/directories (starting with ``'.'``) and ``__pycache__``
    directories are automatically skipped.

    Args:
        data_dir:   Root directory to search.
        extensions: Tuple of lowercase file suffixes (e.g. ``('.csv', '.pdf')``).

    Returns:
        Sorted list of ``Path`` objects for every matching file.
    """
    root = Path(data_dir).resolve()
    if not root.is_dir():
        logging.getLogger("dataintern").warning(
            "Data directory does not exist: %s", root
        )
        return []

    matched: list[Path] = []
    for path in root.rglob("*"):
        # Skip hidden entries and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in extensions:
            matched.append(path)

    matched.sort()
    logging.getLogger("dataintern").info(
        "Discovered %d file(s) in %s", len(matched), root
    )
    return matched


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------

_MULTI_WHITESPACE = re.compile(r"[^\S\n]+")  # whitespace except newline
_MULTI_NEWLINES = re.compile(r"\n{3,}")       # 3+ consecutive newlines


def clean_text(text: str) -> str:
    """Normalise whitespace and strip a block of text.

    * Collapses runs of spaces/tabs into a single space.
    * Reduces three-or-more consecutive newlines to two.
    * Strips leading/trailing whitespace.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text string.
    """
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


# ------------------------------------------------------------------
# Chunk ID generation
# ------------------------------------------------------------------

def generate_chunk_id(filename: str, index: int) -> str:
    """Create a deterministic, human-readable chunk identifier.

    The filename is sanitised (dots and spaces replaced with
    underscores, lowercased) and combined with a zero-padded index.

    Examples:
        >>> generate_chunk_id("deals.csv", 42)
        'deals_csv_chunk_0042'
        >>> generate_chunk_id("Annual Report.pdf", 7)
        'annual_report_pdf_chunk_0007'

    Args:
        filename: Original file name (e.g. ``'deals.csv'``).
        index:    Zero-based chunk index within that file.

    Returns:
        Chunk ID string in the form ``'<safe_name>_chunk_NNNN'``.
    """
    safe = re.sub(r"[.\s]+", "_", filename).lower().strip("_")
    return f"{safe}_chunk_{index:04d}"
