import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import rag_logger

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    rag_logger.info(f"Generating embeddings for {len(texts)} texts")
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    rag_logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


def generate_mock_embedding(text: str) -> List[float]:
    """Generate mock embedding for testing without API key."""
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    import random
    random.seed(hash_val)
    return [random.uniform(-1, 1) for _ in range(1536)]


def generate_mock_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate mock embeddings for testing."""
    return [generate_mock_embedding(text) for text in texts]
