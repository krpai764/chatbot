"""Vector store module for the DataIntern RAG Engine.

Provides the VectorStore class that wraps a ChromaDB persistent client
for storing, searching, and managing document embeddings.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Maximum number of documents to upsert in a single ChromaDB call.
_UPSERT_BATCH_SIZE = 100


class VectorStore:
    """Persistent vector store backed by ChromaDB.

    Attributes:
        _client: ChromaDB persistent client instance.
        _collection: Active ChromaDB collection.
        _collection_name: Name of the active collection.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "dataintern",
    ) -> None:
        """Initialise the VectorStore with a persistent ChromaDB client.

        Args:
            persist_dir: Directory where ChromaDB persists data.
            collection_name: Name of the collection to use.
        """
        self._collection_name = collection_name
        logger.info(
            "Initialising ChromaDB client (persist_dir=%s, collection=%s) …",
            persist_dir,
            collection_name,
        )

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Collection '%s' ready – %d document(s) currently stored.",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Return ``True`` if the collection contains at least one document."""
        return self._collection.count() > 0

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert documents into the collection in batches.

        Processes documents in batches of ``_UPSERT_BATCH_SIZE`` to stay
        within ChromaDB's per-call limits.  Metadata values are sanitised
        before insertion so that only ``str``, ``int``, ``float``, and
        ``bool`` types are stored (ChromaDB requirement).

        Args:
            ids: Unique identifiers for each document.
            embeddings: Pre-computed embedding vectors.
            documents: Raw text content of each document.
            metadatas: Metadata dicts associated with each document.
        """
        total = len(ids)
        if total == 0:
            logger.warning("add_documents called with zero documents.")
            return

        # Sanitise metadata up-front.
        clean_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        logger.info(
            "Upserting %d document(s) into '%s' (batch_size=%d) …",
            total,
            self._collection_name,
            _UPSERT_BATCH_SIZE,
        )

        for start in range(0, total, _UPSERT_BATCH_SIZE):
            end = min(start + _UPSERT_BATCH_SIZE, total)
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=clean_metadatas[start:end],
            )
            logger.info(
                "  Upserted batch [%d – %d) of %d.", start, end, total
            )

        logger.info(
            "Upsert complete. Collection now contains %d document(s).",
            self._collection.count(),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """Query the collection for the nearest neighbours.

        Args:
            query_embedding: The query's embedding vector.
            top_k: Number of results to return.

        Returns:
            A dict with keys ``ids``, ``documents``, ``metadatas``, and
            ``distances`` (each a list aligned by index).
        """
        logger.debug("Searching collection (top_k=%d) …", top_k)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (one per query); flatten to a
        # single-query response.
        output: dict = {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

        logger.info(
            "Search returned %d result(s).", len(output["ids"])
        )
        return output

    def get_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (destructive)."""
        logger.warning(
            "Resetting collection '%s' – all data will be deleted.",
            self._collection_name,
        )
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' has been reset.", self._collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
        """Sanitise a metadata dict for ChromaDB compatibility.

        ChromaDB metadata values must be ``str``, ``int``, ``float``, or
        ``bool``.  This method converts unsupported types:

        * ``None``  → ``""`` (empty string)
        * ``list`` / ``dict`` → ``str(value)``
        * Everything else → ``str(value)``

        Args:
            metadata: Raw metadata dict.

        Returns:
            A new dict with all values coerced to ChromaDB-safe types.
        """
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict, tuple)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"collection={self._collection_name!r}, "
            f"count={self._collection.count()})"
        )
