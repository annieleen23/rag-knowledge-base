import os
from typing import List
from src.utils.logger import rag_logger


def load_text_file(file_path: str) -> str:
    """Load text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    rag_logger.info(f"Loaded text file: {file_path} ({len(content)} chars)")
    return content


def load_documents_from_directory(directory: str) -> List[dict]:
    """Load all .txt files from a directory."""
    documents = []
    if not os.path.exists(directory):
        rag_logger.warning(f"Directory not found: {directory}")
        return documents

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            content = load_text_file(filepath)
            documents.append({
                "filename": filename,
                "content": content,
                "source": filepath
            })

    rag_logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def load_sample_documents() -> List[dict]:
    """Load sample documents for demo purposes."""
    samples = [
        {
            "filename": "data_engineering.txt",
            "content": """Data engineering is the practice of designing and building systems for collecting, storing, and analyzing data at scale.
A data engineer builds pipelines that transform and transport data into a format where data scientists can analyze it.
Key tools in data engineering include Apache Kafka for real-time streaming, Apache Spark for distributed processing,
Apache Airflow for workflow orchestration, and various cloud platforms like AWS, GCP, and Azure.
ETL stands for Extract, Transform, Load - the core process of moving data from source systems to data warehouses.
Data quality is critical in data engineering - pipelines must validate, clean, and monitor data at every stage.""",
            "source": "sample"
        },
        {
            "filename": "machine_learning.txt",
            "content": """Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
Unsupervised learning finds patterns in data without labeled examples.
Deep learning uses neural networks with many layers to learn complex representations.
RAG stands for Retrieval-Augmented Generation - it combines search with language models to answer questions from documents.
Vector databases store embeddings - numerical representations of text - enabling semantic similarity search.
Popular vector databases include Chroma, Pinecone, Weaviate, and pgvector.""",
            "source": "sample"
        },
        {
            "filename": "cloud_computing.txt",
            "content": """Cloud computing provides on-demand access to computing resources over the internet.
AWS (Amazon Web Services) is the largest cloud provider, offering services like EC2, S3, RDS, and Lambda.
S3 is object storage used for data lakes - storing raw data files like CSV, Parquet, and JSON.
EC2 provides virtual servers for running applications and services.
Docker containers package applications with their dependencies for consistent deployment.
Kubernetes orchestrates containers at scale, managing deployment, scaling, and networking.
CI/CD pipelines automate the build, test, and deployment process for software applications.""",
            "source": "sample"
        }
    ]
    rag_logger.info(f"Loaded {len(samples)} sample documents")
    return samples
