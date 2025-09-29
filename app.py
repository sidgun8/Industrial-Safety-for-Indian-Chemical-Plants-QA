# app.py
import streamlit as st
from main import query_rag, ingest
from llm import translate_with_llm, detect_language

st.set_page_config(page_title="ChemE Safety RAG", layout="wide")
st.title("🦺 Chemical Engineering Safety Assistant")

if st.button("Ingest Documents"):
    ingest()
    st.success("Data ingested successfully!")

# def query_multilingual(query: str, hybrid_retriever=None) -> str:
#     """
#     Accepts Hindi or English query, returns answer in same language.
#     """
#     lang = detect_language(query)

#     # Step 1: translate Hindi query to English if needed
#     if lang == "hi":
#         query_en = translate_with_llm(query, source_lang="Hindi", target_lang="English", max_tokens=256)
#     else:
#         query_en = query

#     # Step 2: call your existing RAG pipeline
#     answer_en = query_rag(query_en)  # optionally pass hybrid_retriever if your pipeline needs it

#     # Step 3: translate back to Hindi if original query was Hindi
#     if lang == "hi":
#         answer_hi = translate_with_llm(answer_en, source_lang="English", target_lang="Hindi", max_tokens=512)
#         return answer_hi
#     else:
#         return answer_en

from llm import translate_with_llm, detect_language

def query_multilingual(query: str) -> str:
    """
    Accepts queries in any language, returns answer in the same language.
    """
    lang = detect_language(query)

    # Step 1: Translate query to English if not English
    if lang != "en":
        query_en = translate_with_llm(query, source_lang=lang, target_lang="English", max_tokens=256)
    else:
        query_en = query

    # Step 2: Pass to your existing BM25 + RAG pipeline
    answer_en = query_rag(query_en)

    # Step 3: Translate answer back to original language if needed
    if lang != "en":
        answer_out = translate_with_llm(answer_en, source_lang="English", target_lang=lang, max_tokens=512)
        return answer_out
    else:
        return answer_en

LANGUAGE_MAP = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ja": "Japanese",
    "zh": "Chinese",
    "th": "Thai",
    "es": "Spanish",
    "de": "German",
    "en": "English"
}

def get_language_name(text: str) -> str:
    code = detect_language(text)
    return LANGUAGE_MAP.get(code, "Unknown")


query = st.text_input("Ask a safety-related question:")
if query:
    lang_name = get_language_name(query)
    st.markdown(f"**Detected language:** {lang_name}")

    with st.spinner("🔎 Searching and thinking..."):
        answer = query_multilingual(query)
    st.markdown("### Answer")
    st.write(answer)
