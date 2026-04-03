import os
from typing import List
from dotenv import load_dotenv
from src.ingestion.document_loader import load_sample_documents, load_documents_from_directory
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.embeddings import generate_embeddings_batch, generate_mock_embeddings_batch, generate_embedding, generate_mock_embedding
from src.retrieval.vector_store import add_chunks_to_store, search_similar, get_collection_stats
from src.generation.llm_client import generate_answer
from src.utils.logger import rag_logger

load_dotenv()


def has_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("your-")


def ingest_documents(directory: str = None, use_samples: bool = True) -> dict:
    """Ingest documents into the vector store."""
    rag_logger.info("Starting document ingestion pipeline")

    if use_samples or not directory:
        documents = load_sample_documents()
    else:
        documents = load_documents_from_directory(directory)

    if not documents:
        return {"status": "error", "message": "No documents found"}

    chunks = chunk_documents(documents, chunk_size=100, overlap=10)

    texts = [chunk["text"] for chunk in chunks]
    if has_api_key():
        embeddings = generate_embeddings_batch(texts)
    else:
        rag_logger.warning("No API key — using mock embeddings")
        embeddings = generate_mock_embeddings_batch(texts)

    add_chunks_to_store(chunks, embeddings)
    stats = get_collection_stats()

    rag_logger.info(f"Ingestion complete: {len(chunks)} chunks stored")
    return {
        "status": "success",
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "total_in_store": stats["total_chunks"]
    }


def query_knowledge_base(question: str) -> dict:
    """Query the knowledge base with a question."""
    rag_logger.info(f"Processing query: {question[:50]}...")

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        return {
            "answer": "Knowledge base is empty. Please ingest documents first.",
            "sources": [],
            "chunks_used": 0
        }

    if has_api_key():
        query_embedding = generate_embedding(question)
    else:
        query_embedding = generate_mock_embedding(question)

    relevant_chunks = search_similar(query_embedding, top_k=3)
    answer = generate_answer(question, relevant_chunks)

    sources = list(set([chunk["metadata"].get("filename", "unknown") for chunk in relevant_chunks]))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(relevant_chunks),
        "context_preview": relevant_chunks[0]["text"][:200] if relevant_chunks else ""
    }
