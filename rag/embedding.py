"""Embedding module for the DataIntern RAG Engine.

Provides the EmbeddingManager class that wraps SentenceTransformers
to generate dense vector embeddings for text chunks and queries.
Default model: BAAI/bge-small-en-v1.5 (384-dimensional embeddings).
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embedding using SentenceTransformer models.

    Attributes:
        _model_name: Name/path of the SentenceTransformer model.
        _model: Loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Initialise the EmbeddingManager and load the model.

        Args:
            model_name: HuggingFace model identifier or local path.
                Defaults to ``BAAI/bge-small-en-v1.5`` (384-dim).
        """
        self._model_name = model_name
        logger.info("Loading SentenceTransformer model: %s …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info(
            "Model loaded successfully. Embedding dimension: %d",
            self._model.get_sentence_embedding_dimension(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts encoded per forward pass.

        Returns:
            A list of embedding vectors, each represented as a list of
            floats with length equal to the model's output dimension.
        """
        if not texts:
            logger.warning("embed_texts called with an empty list.")
            return []

        logger.info(
            "Embedding %d text(s) with batch_size=%d …", len(texts), batch_size
        )

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Convert numpy array → list[list[float]]
        result: list[list[float]] = embeddings.tolist()

        logger.info(
            "Generated %d embedding(s), dimension=%d.",
            len(result),
            len(result[0]) if result else 0,
        )
        return result

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        logger.debug("Embedding query: %.80s…", query)
        embedding: np.ndarray = self._model.encode(
            query,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the name of the loaded embedding model."""
        return self._model_name

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self._model_name!r}, "
            f"dim={self._model.get_sentence_embedding_dimension()})"
        )
