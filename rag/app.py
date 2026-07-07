"""Streamlit Web UI for the DataIntern RAG Engine.

Provides a modern chat interface, dynamic inline Plotly chart rendering,
and a sidebar for uploading new CRM documents mid-conversation.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure the parent directory is in sys.path so 'rag.*' imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="DataIntern AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag.chunker import TextChunker
from rag.config import Config
from rag.document_loader import DocumentLoader
from rag.embedding import EmbeddingManager
from rag.llm import GeminiLLM
from rag.query_engine import QueryEngine
from rag.retriever import Retriever
from rag.utils import setup_logging
from rag.vector_store import VectorStore

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def get_system_components():
    """Initialize and cache the core RAG engine components."""
    config = Config()
    config.validate()
    
    embedding_mgr = EmbeddingManager(model_name=config.EMBEDDING_MODEL)
    vector_store = VectorStore(
        persist_dir=config.CHROMA_PATH,
        collection_name="dataintern",
    )
    llm = GeminiLLM(api_key=config.GEMINI_API_KEY)
    retriever = Retriever(embedding_mgr, vector_store, top_k=config.TOP_K)
    engine = QueryEngine(config, retriever, llm)
    
    return config, embedding_mgr, vector_store, engine


def init_session_state():
    """Initialize chat history in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm DataIntern, your CRM AI assistant. How can I help you analyze your data today?"}
        ]


def process_uploaded_file(uploaded_file, config, embedding_mgr, vector_store):
    """Save an uploaded file to DATA_DIR and index it dynamically."""
    save_path = Path(config.DATA_DIR) / uploaded_file.name
    
    # Save to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.sidebar.info(f"Indexing `{uploaded_file.name}`...")
    
    try:
        # 1. Parse
        loader = DocumentLoader()
        docs = loader.load_file(save_path)
        if not docs:
            st.sidebar.error("Failed to extract text from file.")
            return
            
        # 2. Chunk
        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(docs)
        
        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = embedding_mgr.embed_texts(texts)
        
        # 4. Store
        ids = [c.metadata.get("chunk_id", f"chunk_{uploaded_file.name}_{i}") for i, c in enumerate(chunks)]
        metadatas = [c.metadata for c in chunks]
        vector_store.add_documents(ids, embeddings, texts, metadatas)
        
        st.sidebar.success(f"✅ Indexed {len(chunks)} chunks from {uploaded_file.name}!")
        
    except Exception as e:
        logger.exception("Upload indexing failed")
        st.sidebar.error(f"Error indexing file: {e}")


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

def render_sidebar(config, embedding_mgr, vector_store, engine):
    """Render the sidebar with upload controls and stats."""
    with st.sidebar:
        st.header("🤖 DataIntern Settings")
        
        # Stats
        db_count = vector_store.get_count()
        st.metric("Indexed Chunks", db_count)
        
        st.divider()
        st.subheader("Upload Documents")
        st.write("Upload new CSV, Excel, PDF, JSON, or Word files to instantly add them to the database.")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'xls', 'json', 'pdf', 'docx', 'tsv']
        )
        
        if st.button("Process Uploads"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    process_uploaded_file(uploaded_file, config, embedding_mgr, vector_store)
            else:
                st.warning("Please select a file first.")
                
        st.divider()
        if st.button("Clear Chat History"):
            engine.clear_memory()
            st.session_state.messages = [
                {"role": "assistant", "content": "Memory cleared. How can I help you?"}
            ]
            st.rerun()


def render_chat_message(msg):
    """Render a single message in the chat UI."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render citations if they exist
        if msg.get("citations_text"):
            with st.expander("📚 View Citations"):
                st.markdown(msg["citations_text"])
                
        # Render Plotly chart if it exists
        if msg.get("chart_json"):
            try:
                import plotly.io as pio
                fig = pio.from_json(msg["chart_json"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to render chart: {e}")


def main():
    config, embedding_mgr, vector_store, engine = get_system_components()
    init_session_state()
    
    render_sidebar(config, embedding_mgr, vector_store, engine)
    
    st.title("DataIntern CRM Assistant")
    st.markdown("Ask natural language questions about your business data, and get cited answers and charts.")
    
    # Display existing chat history
    for msg in st.session_state.messages:
        render_chat_message(msg)
        
    # Chat Input
    if prompt := st.chat_input("Ask a question (e.g. 'What is the total sales for March?')"):
        # Append and display user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Process and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    result = engine.ask(prompt)
                    
                    st.markdown(result.answer)
                    
                    # Optional: Convert chart dict back to JSON string for Plotly
                    chart_json_str = None
                    if result.chart_json:
                        import json
                        chart_json_str = json.dumps(result.chart_json)
                        import plotly.io as pio
                        fig = pio.from_json(chart_json_str)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    if result.citations_text:
                        with st.expander("📚 View Citations"):
                            st.markdown(result.citations_text)
                            
                    # Save AI message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "citations_text": result.citations_text,
                        "chart_json": chart_json_str
                    })
                    
                except Exception as e:
                    logger.exception("Error processing question")
                    error_msg = f"❌ An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
