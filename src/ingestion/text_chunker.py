from typing import List
from src.utils.logger import rag_logger


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    rag_logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def chunk_documents(documents: List[dict], chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Chunk all documents and return list of chunk dicts with metadata."""
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "filename": doc.get("filename", "unknown"),
                "chunk_index": i,
                "chunk_id": f"{doc.get('filename', 'doc')}_{i}"
            })

    rag_logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks
