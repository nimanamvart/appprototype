
# Web app structure for EU Law QA system

# 1. Import dependencies
import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# 2. Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load full structured EU law content from parsed JSON
with open("real_eu_laws.json", "r") as f:
    real_laws = json.load(f)

# 4. Flatten for embedding
def flatten_laws(laws):
    passages = []
    meta = []
    for law in laws:
        for art in law["articles"]:
            for para in art["paragraphs"]:
                ref = f"{law['title']}, Article {art['article']}, Paragraph {para['number']}"
                passages.append(para['text'])
                meta.append({"text": para['text'], "ref": ref, "url": law['url']})
    return passages, meta

texts, metadata = flatten_laws(real_laws)
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 5. Streamlit UI
st.set_page_config(page_title="EU Law Q&A", layout="centered")
st.title("\U0001F4D6 EU Law Assistant")

query = st.text_input("Ask a question about EU digital laws:", placeholder="e.g., Do municipalities need to follow Article 5 of the Data Act?")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)
    st.subheader("Answer")
    for idx in I[0]:
        result = metadata[idx]
        st.write(f"**{result['ref']}**")
        st.write(result['text'])
        st.markdown(f"[View full law]({result['url']})")
        st.markdown("---")
