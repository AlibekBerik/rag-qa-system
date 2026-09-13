import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="kazakh_culture")

query = "What is a yurt made of?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print(f"Query: {query}\n")
for i, doc in enumerate(results["documents"][0]):
    topic = results["metadatas"][0][i]["topic"]
    print(f"--- Result {i+1} (from: {topic}) ---")
    print(doc[:300] + "...\n")