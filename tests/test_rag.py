import pytest
from src.ingestion.document_loader import load_sample_documents
from src.ingestion.text_chunker import chunk_text, chunk_documents
from src.retrieval.embeddings import generate_mock_embedding, generate_mock_embeddings_batch
from src.generation.rag_pipeline import ingest_documents, query_knowledge_base


def test_load_sample_documents():
    docs = load_sample_documents()
    assert len(docs) > 0
    assert "content" in docs[0]
    assert "filename" in docs[0]


def test_chunk_text():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_documents():
    docs = load_sample_documents()
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(chunks) > len(docs)
    assert "text" in chunks[0]
    assert "chunk_id" in chunks[0]


def test_mock_embedding():
    embedding = generate_mock_embedding("test text")
    assert len(embedding) == 1536


def test_mock_embeddings_batch():
    texts = ["text one", "text two", "text three"]
    embeddings = generate_mock_embeddings_batch(texts)
    assert len(embeddings) == 3


def test_ingest_and_query():
    result = ingest_documents(use_samples=True)
    assert result["status"] == "success"
    assert result["chunks_created"] > 0

    result = query_knowledge_base("What is ETL?")
    assert "answer" in result
    assert result["chunks_used"] > 0
