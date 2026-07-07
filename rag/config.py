"""
config.py – Centralised configuration for the DataIntern RAG Engine.

Loads settings from a .env file (if present) and falls back to sensible
defaults.  The GEMINI_API_KEY is resolved by checking the KRP_API
environment variable first, then GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this module
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


@dataclass
class Config:
    """Application-wide configuration.

    Attributes:
        GEMINI_API_KEY:        API key for the Google Gemini model.
        CHROMA_PATH:           Directory used by ChromaDB for persistence.
        EMBEDDING_MODEL:       HuggingFace model ID for sentence embeddings.
        TOP_K:                 Number of chunks returned per retrieval query.
        CHUNK_SIZE:            Target size (in characters) for each text chunk.
        CHUNK_OVERLAP:         Overlap (in characters) between consecutive chunks.
        DATA_DIR:              Root directory containing the CRM data files.
        SUPPORTED_EXTENSIONS:  File extensions the loader will process.
    """

    GEMINI_API_KEY: str = field(default_factory=lambda: (
        os.getenv("KRP_API")
        or os.getenv("GEMINI_API_KEY", "")
    ))
    CHROMA_PATH: str = field(
        default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db")
    )
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    TOP_K: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "15"))
    )
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1200"))
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "250"))
    )
    DATA_DIR: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "../")
    )
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".csv", ".tsv", ".xlsx", ".xls", ".json", ".pdf", ".docx",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Validate that all required configuration values are present.

        Raises:
            ValueError: If GEMINI_API_KEY is missing or still set to the
                        placeholder value from .env.example.
        """
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your-gemini-api-key-here":
            raise ValueError(
                "GEMINI_API_KEY is not set.  "
                "Export the KRP_API or GEMINI_API_KEY environment variable, "
                "or add it to your .env file."
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Resolve DATA_DIR to an absolute path relative to this module."""
        self.DATA_DIR = str(
            (Path(__file__).resolve().parent / self.DATA_DIR).resolve()
        )
