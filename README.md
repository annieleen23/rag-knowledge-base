# 🧠 RAG Knowledge Base Q&A

> AI-powered document Q&A system using Retrieval-Augmented Generation — ask questions about your documents and get accurate, context-grounded answers.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.0.335-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings+GPT-orange?style=flat-square&logo=openai)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                        │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Document  │──▶│    Text     │──▶│   Embedding     │   │
│  │   Loader    │   │   Chunker   │   │   Generation    │   │
│  │             │   │             │   │                 │   │
│  │ • .txt files│   │ • 500 tokens│   │ • OpenAI Ada-2  │   │
│  │ • PDF files │   │ • 50 overlap│   │ • 1536 dims     │   │
│  │ • Sample docs   │ • Metadata  │   │ • Batch process │   │
│  └─────────────┘   └─────────────┘   └────────┬────────┘   │
│                                               │             │
│                                               ▼             │
│                                    ┌─────────────────┐      │
│                                    │    ChromaDB     │      │
│                                    │  Vector Store   │      │
│                                    │  HNSW Index     │      │
│                                    │  Cosine Sim     │      │
│                                    └─────────────────┘      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                           │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Query     │──▶│  Semantic   │──▶│   GPT-3.5       │   │
│  │  Embedding  │   │   Search    │   │   Generation    │   │
│  │             │   │             │   │                 │   │
│  │ • Same model│   │ • Top-K=3   │   │ • Context-only  │   │
│  │ • 1536 dims │   │ • Cosine sim│   │ • Temp=0.1      │   │
│  │             │   │ • Metadata  │   │ • Grounded ans  │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│                                               │             │
│                                               ▼             │
│                                    Structured Answer        │
│                                    + Sources + Context      │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Semantic Search** — Finds contextually relevant document chunks using vector similarity, not just keyword matching
- **Chunk Overlap** — 10-20% overlap between chunks preserves context at boundaries
- **Source Attribution** — Every answer cites which documents were used
- **Mock Embedding Fallback** — System works for development and testing without OpenAI API quota
- **Context Preview** — Users can inspect the retrieved context that informed the answer
- **Persistent Vector Store** — ChromaDB persists embeddings to disk for reuse across sessions
- **Streamlit UI** — Interactive Q&A interface with sample questions and real-time responses

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| RAG Framework | LangChain |
| Vector Database | ChromaDB (HNSW index, cosine similarity) |
| Embeddings | OpenAI text-embedding-ada-002 (1536 dims) |
| LLM | OpenAI GPT-3.5 Turbo |
| UI | Streamlit |
| Testing | Pytest |

---

## 📁 Project Structure
```
rag-knowledge-base/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # Load .txt, PDF, sample docs
│   │   └── text_chunker.py       # Split text into overlapping chunks
│   ├── retrieval/
│   │   ├── embeddings.py         # OpenAI embedding generation
│   │   └── vector_store.py       # ChromaDB operations
│   ├── generation/
│   │   ├── llm_client.py         # GPT answer generation
│   │   └── rag_pipeline.py       # End-to-end orchestration
│   └── utils/
│       └── logger.py             # Structured logging
├── tests/
│   └── test_rag.py               # Unit tests
├── app.py                         # Streamlit UI
└── requirements.txt
```

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/annieleen23/rag-knowledge-base
cd rag-knowledge-base
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

### Run the App
```bash
PYTHONPATH=. streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Click **Load Sample Documents** to ingest knowledge base
2. Ask any question about Data Engineering, ML, or Cloud Computing
3. View the answer, sources, and retrieved context

---

## 💬 Sample Questions

- *"What is ETL in data engineering?"*
- *"What is RAG and how does it work?"*
- *"What tools are used in data engineering?"*
- *"What is Docker used for?"*
- *"What is the difference between supervised and unsupervised learning?"*

---

## 🔑 Key Engineering Decisions

- **Chunk size tuning** — 500 tokens with 10-20% overlap balances context preservation with retrieval precision; too large = irrelevant context, too small = lost meaning
- **Cosine similarity** — More robust than Euclidean distance for comparing embeddings of different lengths
- **Temperature=0.1 for generation** — Keeps answers factual and grounded in retrieved context, reduces hallucination
- **Mock embedding fallback** — Deterministic hash-based mock embeddings allow full system testing without API quota
- **Graceful degradation** — When LLM fails, system returns retrieved context directly rather than crashing
