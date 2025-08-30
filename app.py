# app.py
import streamlit as st
from main import query_rag, ingest

st.set_page_config(page_title="ChemE Safety RAG", layout="wide")
st.title("🦺 Chemical Engineering Safety RAG Assistant")

if st.button("Ingest Documents"):
    ingest()
    st.success("Data ingested successfully!")

query = st.text_input("Ask a safety-related question:")
if query:
    with st.spinner("🔎 Searching and thinking..."):
        answer = query_rag(query)
    st.markdown("### Answer")
    st.write(answer)


