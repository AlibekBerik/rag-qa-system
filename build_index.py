import json
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_text

# Load the articles we fetched
with open("articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

all_chunks = []
all_ids = []
all_metadata = []

for topic, text in articles.items():
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{topic}_{i}")
        all_metadata.append({"topic": topic, "chunk_index": i})

print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

# --- Embeddings ---
print("Loading embedding model (first run downloads it, may take a minute)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# --- Store in ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="kazakh_culture")

collection.add(
    ids=all_ids,
    embeddings=embeddings.tolist(),
    documents=all_chunks,
    metadatas=all_metadata,
)

print(f"\nStored {len(all_chunks)} chunks in ChromaDB collection 'kazakh_culture'")
print("Done! You can now query this collection.")