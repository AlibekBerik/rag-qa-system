\# Kazakh Culture RAG Q\&A System



A Retrieval-Augmented Generation (RAG) system that answers questions about Kazakh nomadic traditions, music, and customs — grounded in real Wikipedia articles, not model guesses.



\## Demo



!\[demo](demo.gif)



Ask a question, get an answer generated from real retrieved source text — plus which articles it came from. Ask something outside its knowledge base, and it correctly says so instead of making something up.



\## How it works



1\. \*\*Fetch\*\* — Pulls real Wikipedia articles on Kazakh culture (nomadic pastoralism, yurts, dombra, Nauryz, cuisine, eagle hunting, the Kazakh Khanate, and more) using the Wikipedia API.

2\. \*\*Chunk\*\* — Splits articles into overlapping \~500-word chunks so retrieval can find specific relevant passages instead of whole articles.

3\. \*\*Embed\*\* — Converts each chunk into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`), capturing meaning rather than exact keywords.

4\. \*\*Store\*\* — Saves all embeddings in a local ChromaDB vector database.

5\. \*\*Retrieve\*\* — When a question comes in, it's embedded the same way, and ChromaDB finds the most semantically relevant chunks.

6\. \*\*Generate\*\* — The question + retrieved chunks are sent to an LLM (Groq, `openai/gpt-oss-120b`) with instructions to answer \*only\* using that context — and to say so if the context doesn't contain the answer.



\## Example



\*\*Q: What do Kazakhs eat traditionally?\*\*

> Traditionally Kazakh food is centered on meat and dairy — large pieces of boiled horse or mutton (besbarmak), offal dishes like quwyrdaq, and fermented mare's milk (kumys)...



\*Sources: Kazakh cuisine, Kazakhs, Kazakh clothing\*



\*\*Q: What is the capital of France?\*\*

> I don't have enough information.



This refusal behavior is intentional and tested — the system is grounded in its actual knowledge base and doesn't hallucinate answers outside it.



\## Tech stack



\- Python, FastAPI

\- ChromaDB (vector database)

\- sentence-transformers (embeddings)

\- Groq API (LLM generation)

\- Docker (containerization)



\## Running it yourself



\### With Docker (recommended)



\\`\\`\\`bash

git clone https://github.com/AlibekBerik/rag-qa-system

cd rag-qa-system

\\`\\`\\`



Create a `.env` file with your own Groq API key:

\\`\\`\\`

GROQ\_API\_KEY=your\_key\_here

\\`\\`\\`



Build the vector database (only needed once):

\\`\\`\\`bash

pip install -r requirements.txt

python fetch\_articles.py

python build\_index.py

\\`\\`\\`



Build and run the container:

\\`\\`\\`bash

docker build -t rag-qa-system .

docker run -p 8000:8000 --env-file .env rag-qa-system

\\`\\`\\`



\### Without Docker



\\`\\`\\`bash

python -m venv venv

venv\\\\Scripts\\\\activate      # Windows

pip install -r requirements.txt

python fetch\_articles.py

python build\_index.py

uvicorn api:app --reload

\\`\\`\\`



Then open `http://127.0.0.1:8000/docs` and try the `/ask` endpoint.



\## Note on data



`chroma\_db/` (the built vector database) isn't included in this repo to keep it lightweight — running `fetch\_articles.py` then `build\_index.py` rebuilds it locally in under a minute.

