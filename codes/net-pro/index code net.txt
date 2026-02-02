import json
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text: str, chunk_size=1200, overlap=150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def main():
    client = chromadb.PersistentClient(path="db/chroma")

    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )

    col = client.get_or_create_collection(
        name="networking_en",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    doc_id = 0

    with open("data/raw_pages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            for i, ch in enumerate(chunk_text(text)):
                doc_id += 1
                ids.append(str(doc_id))
                docs.append(ch)
                metas.append({
                    "url": rec["url"],
                    "domain": rec["domain"],
                    "title": rec["title"],
                    "chunk_index": i
                })

    col.add(ids=ids, documents=docs, metadatas=metas)
    print("DONE: Vector DB built. Chunks:", len(ids))

if __name__ == "__main__":
    main()
