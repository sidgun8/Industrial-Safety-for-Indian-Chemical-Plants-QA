import streamlit as st
from main import ingest, query_rag

st.set_page_config(page_title="ChemE Safety RAG", layout="wide")
st.title("🦺 Chemical Engineering Safety Assistant")

if st.button("Ingest Documents"):
    ingest(force_reindex=True)
    st.success("Data ingested successfully!")

query = st.text_input("Ask a safety-related question:")
if query:
    with st.spinner("🔎 Searching..."):
        answer = query_rag(query, show_chunks=True)
    st.markdown("### Answer")
    st.write(answer)
