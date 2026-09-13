import wikipediaapi
import json

wiki = wikipediaapi.Wikipedia(
    user_agent='RAG-QA-System (alibekberik13@gmail.com)',
    language='en'
)

topics = [
    "Nomadic pastoralism",
    "Kazakhs",
    "Yurt",
    "Dombra",
    "Nauryz",
    "Kazakh cuisine",
    "Kazakh clothing",
    "Eagle hunting",
    "Kazakh Khanate",
    "Kazakh music"
]

articles = {}

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        articles[topic] = page.text
        print(f"Fetched: {topic} ({len(page.text)} characters)")
    else:
        print(f"NOT FOUND: {topic}")

with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(articles)} articles to articles.json")