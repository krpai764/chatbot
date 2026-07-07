# DataIntern RAG Engine

A production-quality **Retrieval-Augmented Generation** pipeline for CRM analytics, powered by **Gemini 2.5 Flash**, **ChromaDB**, and **sentence-transformers**.

## Architecture

```
CRM Files (CSV, JSON, XLSX, PDF, DOCX)
        │
        ▼
   ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌───────────┐
   │  Parse   │ ──▶ │  Chunk   │ ──▶ │   Embed      │ ──▶ │ ChromaDB  │
   │ (loader) │     │ (800/150)│     │ (BGE-small)  │     │ (persist) │
   └──────────┘     └──────────┘     └──────────────┘     └─────┬─────┘
                                                                │
                                          ┌─────────────────────┘
                                          ▼
   User Question ──▶ Embed Query ──▶ Semantic Search (Top-5)
                                          │
                                          ▼
                                  ┌──────────────┐
                                  │ Gemini 2.5   │
                                  │   Flash      │
                                  │ (context-    │
                                  │  grounded)   │
                                  └──────┬───────┘
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                      Answer        Citations     Chart (Plotly)
```

## Features

- **7 file format parsers**: CSV, TSV, JSON, Excel (all sheets), PDF, DOCX
- **One-time indexing**: Embeddings persist in ChromaDB — never regenerated per query
- **Context-grounded answers**: Gemini answers ONLY from retrieved chunks
- **Structured citations**: Every answer includes file/sheet/page/row references
- **Chart generation**: Plotly Express charts built from original datasets (not chunks)
- **Conversation memory**: Follow-up questions resolved using sliding-window history
- **5 chart types**: Bar, Line, Pie, Scatter, Histogram

## Quick Start

### 1. Install dependencies

```bash
cd rag/
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Option A: Use your existing env var
export KRP_API="your-gemini-api-key"

# Option B: Create a .env file
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 3. Run the engine

```bash
# From the CRM_SampleData directory
cd rag/
python -m rag.main
```

On first run, the engine will:
1. Parse all 7 CRM files
2. Create ~100+ chunks
3. Generate embeddings (BAAI/bge-small-en-v1.5)
4. Store everything in ChromaDB

Subsequent runs skip indexing automatically.

### 4. Ask questions

```
📝 You: What was total closed-won revenue?
💡 Answer: The total closed-won revenue is $1,171,000...

📝 You: Show revenue by region
📊 Chart generated: Revenue by Region (Plotly JSON)

📝 You: What about only North?
💡 Answer: (follows up on previous context)

📝 You: /quit
```

## CLI Options

```bash
python -m rag.main                          # Interactive mode
python -m rag.main --query "Total revenue?" # Single query (JSON output)
python -m rag.main --reindex                # Force re-indexing
python -m rag.main --debug                  # Enable debug logging
```

## Project Structure

```
rag/
├── __init__.py            # Package init
├── config.py              # Configuration from .env
├── utils.py               # Logging, file discovery, text cleaning
├── document_loader.py     # Parsers for CSV/TSV/JSON/XLSX/PDF/DOCX
├── chunker.py             # Text chunking (800 chars, 150 overlap)
├── embedding.py           # Sentence-transformers wrapper
├── vector_store.py        # ChromaDB wrapper
├── retriever.py           # Semantic search orchestrator
├── llm.py                 # Gemini 2.5 Flash client
├── query_engine.py        # Main pipeline orchestrator
├── citation.py            # Citation extraction & formatting
├── conversation_memory.py # Follow-up question handling
├── chart_detector.py      # Chart intent detection via Gemini
├── chart_generator.py     # Plotly chart generation from datasets
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## Configuration

All settings are configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `KRP_API` / `GEMINI_API_KEY` | — | Google AI API key (required) |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `DATA_DIR` | `../` | Directory containing CRM data files |

## Programmatic Usage

```python
from rag.config import Config
from rag.embedding import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.llm import GeminiLLM
from rag.query_engine import QueryEngine

# Initialise
config = Config()
config.validate()
embedding_mgr = EmbeddingManager(config.EMBEDDING_MODEL)
vector_store = VectorStore(config.CHROMA_PATH)
llm = GeminiLLM(api_key=config.GEMINI_API_KEY)
retriever = Retriever(embedding_mgr, vector_store, top_k=config.TOP_K)
engine = QueryEngine(config, retriever, llm)

# Ask a question
result = engine.ask("What was total closed-won revenue?")
print(result.answer)       # Natural language answer
print(result.citations)    # [{"file": "deals.csv", "rows": "..."}]
print(result.chart_json)   # None (or Plotly JSON for chart queries)
```

## Data Files

| File | Format | Records | Key |
|------|--------|---------|-----|
| `deals.csv` | CSV | 60 deals | `deal_id` |
| `contacts.csv` | CSV | 40 contacts | `contact_id` |
| `accounts.json` | JSON | 20 accounts | `account_id` |
| `activities.json` | JSON | 80 activities | `activity_id` |
| `crm_workbook.xlsx` | XLSX | 5+ sheets | — |
| `account_review.docx` | DOCX | Narrative | — |
| `deal_contracts.pdf` | PDF | Contracts | `deal_id` |

## Test Questions

1. **"What was total closed-won revenue?"** → $1,171,000
2. **"Who is the top rep by pipeline?"** → Check Summary sheet
3. **"List at-risk accounts"** → From account_review.docx
4. **"Show revenue by region"** → Plotly bar chart
5. **"Show a histogram of deal values"** → Plotly histogram
