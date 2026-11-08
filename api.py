import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
COLLECTION_NAME = "kazakh_culture"
TOP_K = 4

REFUSAL_INSTRUCTION = (
    "Answer the question using ONLY the context below. If the context doesn't "
    "contain the answer, say you don't have enough information."
)


# Heavy resources are created on first use, not at import time, so the API
# starts the same way but the tests can run without models or API keys.
@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_llm_client():
    from groq import Groq

    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(question, chunks):
    """Ground the LLM in the retrieved chunks and tell it to refuse otherwise."""
    context = "\n\n".join(chunks)
    return f"""{REFUSAL_INSTRUCTION}

Context:
{context}

Question: {question}

Answer:"""


@app.get("/")
def root():
    return {"message": "Kazakh Culture RAG Q&A API is running"}


@app.post("/ask")
def ask(question: str):
    # 1. Embed the question
    query_embedding = get_embed_model().encode([question]).tolist()

    # 2. Retrieve relevant chunks
    results = get_collection().query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]

    # 3. Generate an answer grounded in the retrieved context
    response = get_llm_client().chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        # Keep retrieval order (most relevant first) while removing duplicates
        "sources": list(dict.fromkeys(topics)),
    }
