import json
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def main():
    client = chromadb.PersistentClient(path="db/chroma")

    embedder = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )

    collection = client.get_or_create_collection(
        name="persian_literature",
        embedding_function=embedder
    )

    with open("data/raw_pages.jsonl", encoding="utf-8") as f:
        i = 0
        for line in f:
            rec = json.loads(line)
            for ch in chunk_text(rec["text"]):
                i += 1
                collection.add(
                    ids=[str(i)],
                    documents=[ch],
                    metadatas=[{
                        "title": rec["title"],
                        "url": rec["url"]
                    }]
                )

    print("DONE: دیتابیس برداری ساخته شد")

if __name__ == "__main__":
    main()
