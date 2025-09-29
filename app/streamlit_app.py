import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="DS Interview Coach", page_icon="🧠", layout="wide")
st.title("DS Interview Coach — RAG Agent")

from services.rag import RagOrchestrator

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "interview_chunks")

rag = RagOrchestrator(qdrant_host=QDRANT_HOST, qdrant_port=QDRANT_PORT, collection=COLLECTION)

st.sidebar.header("Modes")
mode = st.sidebar.selectbox("Focus", ["Technical", "Behavioral", "All"], index=2)
top_k = st.sidebar.slider("Top-K", 3, 15, 6)

st.caption("Type a question (e.g., *Explain vanishing gradients and how to mitigate them*).")
query = st.text_input("Your question")

if "history" not in st.session_state: st.session_state.history = []

if st.button("Ask") and query:
    with st.spinner("Retrieving..."):
        results = rag.retrieve(query, top_k=top_k, mode=mode.lower())
    st.session_state.history.append((query, results))

for q, res in reversed(st.session_state.history):
    st.markdown(f"### ❓ {q}")
    if not res:
        st.info("No results yet. Did you run the ingestion? `make ingest`")
        continue
    for i, r in enumerate(res, 1):
        with st.expander(f"Match #{i} — score {r.get('score', 0):.3f}"):
            st.markdown(f"**Source:** `{r.get('source', 'unknown')}`")
            st.write(r.get("text", "")[:1200])
