import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import rag_logger

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, context_chunks: list) -> str:
    """Generate answer using retrieved context chunks."""
    if not context_chunks:
        return "I could not find relevant information to answer your question."

    context = "\n\n".join([chunk["text"] for chunk in context_chunks])

    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Only use information from the context to answer. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer only based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        rag_logger.info("Generated answer successfully")
        return answer
    except Exception as e:
        rag_logger.error(f"LLM generation failed: {e}")
        return f"I found relevant information but could not generate a response. Context: {context[:200]}..."
