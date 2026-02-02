import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama

st.title("RAG | شعر و ادبیات فارسی")

client = chromadb.PersistentClient(path="db/chroma")
embedder = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

collection = client.get_or_create_collection(
    name="persian_literature",
    embedding_function=embedder
)

question = st.text_input("سوال ادبی خود را بنویس:")

if st.button("جستجو") and question:
    res = collection.query(query_texts=[question], n_results=5)

    rows = []
    contexts = []

    for i in range(len(res["documents"][0])):
        doc = res["documents"][0][i]
        meta = res["metadatas"][0][i]
        rows.append({
            "رتبه": i + 1,
            "متن": doc[:200] + "...",
            "منبع": meta["title"],
            "لینک": meta["url"]
        })
        contexts.append(doc)

    st.subheader("نتایج بازیابی")
    st.dataframe(pd.DataFrame(rows))

    prompt = f"""
با توجه به متون زیر به سوال پاسخ بده:

سوال: {question}

متون:
{contexts}
"""

    answer = ollama.generate(
        model="llama3.1",
        prompt=prompt
    )

    st.subheader("پاسخ RAG")
    st.write(answer["response"])
