#!/usr/bin/env python3
"""main.py – CLI entry point for the DataIntern RAG Engine.

Runs the one-time indexing pipeline, then enters an interactive Q&A loop.

Usage::

    cd rag/
    python main.py              # Index + interactive Q&A
    python main.py --reindex    # Force re-index even if ChromaDB exists
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rag.chunker import TextChunker
from rag.config import Config
from rag.document_loader import DocumentLoader
from rag.embedding import EmbeddingManager
from rag.llm import GeminiLLM
from rag.query_engine import QueryEngine
from rag.retriever import Retriever
from rag.utils import setup_logging
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Indexing pipeline (runs once)
# ──────────────────────────────────────────────────────────────────────

def run_indexing(
    config: Config,
    embedding_mgr: EmbeddingManager,
    vector_store: VectorStore,
    force: bool = False,
) -> None:
    """Parse, chunk, embed, and store all CRM documents.

    This is skipped when the ChromaDB collection already contains data,
    unless *force* is ``True``.

    Args:
        config:        Application config.
        embedding_mgr: Embedding manager.
        vector_store:  Vector store (ChromaDB).
        force:         If ``True``, delete existing data and re-index.
    """
    if vector_store.collection_exists() and not force:
        logger.info(
            "ChromaDB already contains %d document(s) – skipping indexing. "
            "Use --reindex to force.",
            vector_store.get_count(),
        )
        return

    if force and vector_store.collection_exists():
        logger.info("Force re-index requested. Resetting collection.")
        vector_store.reset()

    # 1. Parse
    logger.info("─" * 50)
    logger.info("STEP 1 / 4 — Parsing documents from: %s", config.DATA_DIR)
    logger.info("─" * 50)
    loader = DocumentLoader()
    documents = loader.load_all(config.DATA_DIR, config.SUPPORTED_EXTENSIONS)
    logger.info("Parsed %d raw document(s).", len(documents))

    if not documents:
        logger.error("No documents found. Check DATA_DIR in your .env file.")
        sys.exit(1)

    # 2. Chunk
    logger.info("─" * 50)
    logger.info("STEP 2 / 4 — Chunking documents")
    logger.info("─" * 50)
    chunker = TextChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(documents)
    logger.info("Created %d chunk(s).", len(chunks))

    # 3. Embed
    logger.info("─" * 50)
    logger.info("STEP 3 / 4 — Generating embeddings")
    logger.info("─" * 50)
    texts = [c.text for c in chunks]
    embeddings = embedding_mgr.embed_texts(texts)
    logger.info("Generated %d embedding(s).", len(embeddings))

    # 4. Store
    logger.info("─" * 50)
    logger.info("STEP 4 / 4 — Storing in ChromaDB")
    logger.info("─" * 50)
    ids = [c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
    metadatas = [c.metadata for c in chunks]
    vector_store.add_documents(ids, embeddings, texts, metadatas)
    logger.info(
        "Indexing complete. %d chunks stored in ChromaDB.",
        vector_store.get_count(),
    )


# ──────────────────────────────────────────────────────────────────────
# Interactive Q&A loop
# ──────────────────────────────────────────────────────────────────────

def interactive_loop(engine: QueryEngine) -> None:
    """Run an interactive question-answer loop in the terminal.

    Special commands:
        /quit, /exit  — exit the loop
        /clear        — clear conversation memory
        /help         — show help text
    """
    print("\n" + "═" * 60)
    print("  🤖  DataIntern RAG Engine — Interactive Mode")
    print("═" * 60)
    print("  Type your question below. Commands:")
    print("    /quit or /exit  — exit")
    print("    /clear          — clear conversation memory")
    print("    /help           — show this help")
    print("═" * 60 + "\n")

    while True:
        try:
            question = input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue

        if question.lower() in ("/quit", "/exit"):
            print("Goodbye! 👋")
            break

        if question.lower() == "/clear":
            engine.clear_memory()
            print("🔄 Conversation memory cleared.\n")
            continue

        if question.lower() == "/help":
            print("  /quit or /exit  — exit")
            print("  /clear          — clear conversation memory")
            print("  /help           — show this help\n")
            continue

        # Process the question.
        try:
            result = engine.ask(question)
        except Exception as exc:
            logger.error("Error processing question: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}\n")
            continue

        # Display answer.
        print(f"\n{'─' * 60}")
        print(f"💡 Answer:\n{result.answer}")

        # Display citations.
        if result.citations_text:
            print(f"\n📚 Citations:\n{result.citations_text}")

        # Display retrieved chunk metadata summary.
        if result.retrieved_chunks:
            print(f"\n🔍 Retrieved {len(result.retrieved_chunks)} chunk(s):")
            for i, chunk in enumerate(result.retrieved_chunks[:5], 1):
                score = chunk["similarity_score"]
                meta = chunk["metadata"]
                fname = meta.get("filename", "?")
                extra = ""
                if meta.get("sheet"):
                    extra += f", sheet: {meta['sheet']}"
                if meta.get("row_range"):
                    extra += f", rows: {meta['row_range']}"
                if meta.get("page"):
                    extra += f", page: {meta['page']}"
                print(f"   [{i}] {fname}{extra} (score: {score:.4f})")

        # Display chart info.
        if result.chart_json:
            print(f"\n📊 Chart generated: {result.chart_params.get('title', 'Untitled')}")
            print(f"   Type: {result.chart_params.get('chart_type', '?')}")
            print(f"   Dataset: {result.chart_params.get('dataset', '?')}")
            print(f"   (Plotly JSON available in result.chart_json)")

        print(f"{'─' * 60}\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    parser = argparse.ArgumentParser(
        description="DataIntern RAG Engine — CRM Q&A powered by Gemini",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing even if ChromaDB already has data.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    # Also configure the root logger so all 'rag.*' modules log too.
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        force=True,
    )

    logger.info("Starting DataIntern RAG Engine…")
    config = Config()
    config.validate()
    logger.info("Configuration loaded. DATA_DIR=%s", config.DATA_DIR)

    # ── Initialise components ────────────────────────────────────────
    embedding_mgr = EmbeddingManager(model_name=config.EMBEDDING_MODEL)
    vector_store = VectorStore(
        persist_dir=config.CHROMA_PATH,
        collection_name="dataintern",
    )
    llm = GeminiLLM(api_key=config.GEMINI_API_KEY)

    # ── Indexing (one-time) ──────────────────────────────────────────
    run_indexing(config, embedding_mgr, vector_store, force=args.reindex)

    # ── Query engine ─────────────────────────────────────────────────
    retriever = Retriever(embedding_mgr, vector_store, top_k=config.TOP_K)
    engine = QueryEngine(config, retriever, llm)

    # ── Single query or interactive ──────────────────────────────────
    if args.query:
        result = engine.ask(args.query)
        output = result.to_dict()
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        interactive_loop(engine)


if __name__ == "__main__":
    main()
