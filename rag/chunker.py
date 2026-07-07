"""Text chunker for the DataIntern RAG Engine.

Splits `Document` instances into smaller, overlapping chunks suitable
for embedding and vector-store indexing.  Chunking is character-level
with a preference for splitting on sentence boundaries.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from rag.document_loader import Document
from rag.utils import generate_chunk_id, clean_text

logger = logging.getLogger(__name__)


class TextChunker:
    """Character-level text chunker with overlap and sentence-boundary awareness.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks.

    Usage::

        chunker = TextChunker(chunk_size=800, chunk_overlap=150)
        chunks = chunker.chunk_documents(documents)
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document, start_idx: int = 0) -> list[Document]:
        """Split a single Document into overlapping chunks.

        If the document text is shorter than *chunk_size*, it is returned
        as-is (with an added ``chunk_id`` in metadata).  Otherwise the
        text is split into windows of *chunk_size* characters with
        *chunk_overlap* overlap, preferring sentence boundaries.

        Args:
            doc: The source Document to chunk.
            start_idx: The starting index to use for chunk ID generation.

        Returns:
            A list of new Document instances, each carrying the original
            metadata plus a unique ``chunk_id``.
        """
        text = clean_text(doc.text)
        filename = doc.metadata.get("filename", "unknown")

        if not text:
            return []

        # Short-circuit: text fits in a single chunk
        if len(text) <= self.chunk_size:
            meta = deepcopy(doc.metadata)
            meta["chunk_id"] = generate_chunk_id(filename, start_idx)
            return [Document(text=text, metadata=meta)]

        raw_chunks = self._split_text(text)
        chunks: list[Document] = []
        for idx, chunk_text in enumerate(raw_chunks):
            meta = deepcopy(doc.metadata)
            meta["chunk_id"] = generate_chunk_id(filename, start_idx + idx)
            chunks.append(Document(text=chunk_text, metadata=meta))

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        """Chunk a list of Documents.

        Args:
            docs: Source documents to split.

        Returns:
            A flat list of chunked Document instances.
        """
        all_chunks: list[Document] = []
        file_counters: dict[str, int] = {}
        for doc in docs:
            filename = doc.metadata.get("filename", "unknown")
            start_idx = file_counters.get(filename, 0)
            doc_chunks = self.chunk_document(doc, start_idx=start_idx)
            all_chunks.extend(doc_chunks)
            file_counters[filename] = start_idx + len(doc_chunks)

        logger.info(
            "Chunked %d document(s) into %d chunk(s) "
            "(size=%d, overlap=%d)",
            len(docs),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """Split *text* into overlapping character-level chunks.

        The algorithm walks through the text with a sliding window of
        *chunk_size*.  When deciding where to cut, it looks backwards
        from the window boundary for a sentence-ending delimiter
        (``'. '`` or ``'\\n'``).  If one is found within the last 20 %
        of the window, the cut is placed right after that delimiter;
        otherwise the window boundary is used as-is.

        Args:
            text: The full text to split.

        Returns:
            A list of chunk strings.
        """
        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                # Last chunk – take everything remaining
                chunks.append(text[start:].strip())
                break

            # Try to find a clean sentence boundary near the end
            split_pos = self._find_sentence_boundary(text, start, end)
            chunks.append(text[start:split_pos].strip())

            # Advance with overlap
            start = split_pos - self.chunk_overlap
            if start < 0:
                start = 0
            # Guard against no forward progress
            if start >= split_pos:
                start = split_pos

        return [c for c in chunks if c]

    @staticmethod
    def _find_sentence_boundary(text: str, start: int, end: int) -> int:
        """Find the best sentence-boundary split position.

        Searches backwards from *end* looking for ``'. '`` or ``'\\n'``
        within the last 20 % of the ``[start, end)`` window.  Returns
        *end* if no suitable boundary is found.

        Args:
            text: The full text.
            start: Window start index.
            end: Window end index.

        Returns:
            The character index at which to split.
        """
        window_size = end - start
        search_start = end - int(window_size * 0.20)

        # Prefer newline boundaries first, then ". "
        for delimiter in ("\n", ". "):
            pos = text.rfind(delimiter, search_start, end)
            if pos != -1:
                return pos + len(delimiter)

        return end
