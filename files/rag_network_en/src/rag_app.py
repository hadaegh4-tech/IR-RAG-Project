import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama

st.set_page_config(page_title="RAG - Networking (EN)", layout="wide")
st.title("Semantic Search + RAG | Networking (English)")

@st.cache_resource
def load_collection():
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
    return col

def build_prompt(question: str, contexts):
    # مدل را مجبور می‌کنیم فقط از منابع جواب بدهد
    ctx = ""
    for i, c in enumerate(contexts, start=1):
        ctx += f"[{i}] TITLE: {c['title']}\nURL: {c['url']}\nTEXT: {c['text']}\n\n"

    return f"""
You are a networking assistant. Answer ONLY using the provided sources.
If the answer is not in the sources, say: "Not found in the provided sources."

Rules:
- Be concise and technical.
- Use bullet points if helpful.
- At the end, cite sources like: Sources: [1], [3]

Question: {question}

Sources:
{ctx}
"""

def main():
    col = load_collection()

    q = st.text_input("Ask a networking question:", placeholder="e.g., What is DNS and how does it work?")
    top_k = st.slider("Top-K retrieved chunks", 5, 20, 12)

    if st.button("Search") and q.strip():
        res = col.query(query_texts=[q], n_results=top_k)

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        rows = []
        contexts = []
        for i in range(len(docs)):
            snippet = docs[i][:240].replace("\n", " ") + ("..." if len(docs[i]) > 240 else "")
            sim = 1.0 - float(dists[i])
            rows.append({
                "Select": False,
                "Rank": i + 1,
                "Similarity": round(sim, 4),
                "Title": metas[i].get("title", ""),
                "Snippet": snippet,
                "Link": metas[i].get("url", ""),
                "Domain": metas[i].get("domain", ""),
            })
            contexts.append({
                "title": metas[i].get("title", ""),
                "url": metas[i].get("url", ""),
                "text": docs[i],
            })

        st.subheader("Retrieved Results (Ranked)")
        df = pd.DataFrame(rows)
        st.data_editor(
            df,
            use_container_width=True,
            column_config={"Link": st.column_config.LinkColumn("Link")},
            disabled=["Rank", "Similarity", "Title", "Snippet", "Link", "Domain"]
        )

        # برای جلوگیری از شلوغی: فقط 6 متن اول را به LLM بده
        prompt = build_prompt(q, contexts[:6])

        st.subheader("RAG Answer (LLM: llama3.1)")
        out = ollama.generate(model="llama3.1", prompt=prompt)
        st.write(out["response"])

if __name__ == "__main__":
    main()
