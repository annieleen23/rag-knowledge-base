import streamlit as st
from src.generation.rag_pipeline import ingest_documents, query_knowledge_base
from src.retrieval.vector_store import get_collection_stats

st.set_page_config(page_title="RAG Knowledge Base", page_icon="🧠", layout="wide")

st.title("🧠 RAG Knowledge Base Q&A")
st.markdown("Ask questions about your documents using AI-powered semantic search.")

with st.sidebar:
    st.header("📚 Knowledge Base")
    stats = get_collection_stats()
    st.metric("Documents in store", stats["total_chunks"])

    st.markdown("---")
    st.subheader("⚙️ Ingest Documents")

    if st.button("Load Sample Documents", type="primary"):
        with st.spinner("Ingesting documents..."):
            result = ingest_documents(use_samples=True)
        if result["status"] == "success":
            st.success(f"✅ Ingested {result['documents_processed']} docs, {result['chunks_created']} chunks")
            st.rerun()
        else:
            st.error(f"❌ {result['message']}")

    st.markdown("---")
    st.markdown("**Sample Topics:**")
    st.markdown("- 🔧 Data Engineering")
    st.markdown("- 🤖 Machine Learning & RAG")
    st.markdown("- ☁️ Cloud Computing")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")

    sample_questions = [
        "What is ETL in data engineering?",
        "What is RAG and how does it work?",
        "What tools are used in data engineering?",
        "What is the difference between supervised and unsupervised learning?",
        "What is Docker used for?"
    ]

    selected = st.selectbox("Try a sample question:", [""] + sample_questions)
    question = st.text_input("Or type your own question:", value=selected)

    if st.button("🔍 Search", type="primary"):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching knowledge base..."):
                result = query_knowledge_base(question)

            st.markdown("### 💡 Answer")
            st.info(result["answer"])

            col_a, col_b = st.columns(2)
            col_a.metric("Chunks Used", result["chunks_used"])
            col_b.metric("Sources", len(result["sources"]))

            if result["sources"]:
                st.markdown("**📄 Sources:**")
                for source in result["sources"]:
                    st.markdown(f"- `{source}`")

            if result.get("context_preview"):
                with st.expander("🔍 View Retrieved Context"):
                    st.text(result["context_preview"] + "...")

with col2:
    st.subheader("🏗️ How it works")
    st.markdown("""
    1. **📥 Ingest** — Load documents and split into chunks
    2. **🔢 Embed** — Convert chunks to vectors using OpenAI embeddings
    3. **💾 Store** — Save vectors in ChromaDB
    4. **🔍 Retrieve** — Find semantically similar chunks for your query
    5. **🤖 Generate** — GPT generates answer from retrieved context
    """)

    st.markdown("---")
    st.markdown("**🛠️ Tech Stack**")
    st.markdown("""
    - `LangChain` — RAG orchestration
    - `ChromaDB` — Vector database
    - `OpenAI` — Embeddings + GPT
    - `Streamlit` — UI
    """)
