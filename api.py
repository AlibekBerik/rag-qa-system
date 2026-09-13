import os
from dotenv import load_dotenv
from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

app = FastAPI()

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="kazakh_culture")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
def root():
    return {"message": "Kazakh Culture RAG Q&A API is running"}

@app.post("/ask")
def ask(question: str):
    # 1. Embed the question
    query_embedding = embed_model.encode([question]).tolist()

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4
    )
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n".join(chunks)

    # 3. Generate an answer grounded in the retrieved context
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": list(set(topics)),
    }