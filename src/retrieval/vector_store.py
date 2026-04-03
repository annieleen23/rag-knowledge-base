import os
import chromadb
from typing import List
from src.utils.logger import rag_logger


def get_chroma_client():
    """Get ChromaDB client."""
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(collection_name: str = "knowledge_base"):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    rag_logger.info(f"Using collection: {collection_name} ({collection.count()} docs)")
    return collection


def add_chunks_to_store(chunks: List[dict], embeddings: List[List[float]], collection_name: str = "knowledge_base"):
    """Add text chunks and their embeddings to ChromaDB."""
    collection = get_or_create_collection(collection_name)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "filename": chunk["filename"]} for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    rag_logger.info(f"Added {len(chunks)} chunks to vector store")


def search_similar(query_embedding: List[float], top_k: int = 3, collection_name: str = "knowledge_base") -> List[dict]:
    """Search for similar chunks using query embedding."""
    collection = get_or_create_collection(collection_name)

    if collection.count() == 0:
        rag_logger.warning("Vector store is empty")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    rag_logger.info(f"Found {len(chunks)} similar chunks for query")
    return chunks


def get_collection_stats(collection_name: str = "knowledge_base") -> dict:
    """Get stats about the vector store."""
    collection = get_or_create_collection(collection_name)
    return {"total_chunks": collection.count(), "collection_name": collection_name}
