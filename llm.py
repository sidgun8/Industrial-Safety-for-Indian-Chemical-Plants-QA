from config import get_bedrock_client, MODEL_ID

bedrock_client = get_bedrock_client()

def answer_with_llama(query: str, hybrid_retriever, show_chunks: bool = True):
    retrieved = hybrid_retriever.get_relevant_documents(query)
    
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source','unknown')}] {d.page_content}" for d in retrieved
    )
    
    if show_chunks:
        print("\n==== Retrieved Chunks ====\n")
        for i, d in enumerate(retrieved, 1):
            print(f"{i}. [{d.metadata.get('source','unknown')}]")
            print(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
            print()
    
    prompt = f"""You are a strict assistant that answers ONLY using the provided documents as context.

    Question: {query}

    Context:
    {context}

    Answer:"""

    try:
        response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 256, "temperature": 0.0, "topP": 0.9}
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f" Bedrock call failed: {e}")
        return None


def translate_with_llm(text: str, source_lang: str, target_lang: str, max_tokens=512, temperature=0.0):
    """
    Translate text between languages using Meta Llama 4 on Bedrock.
    """
    prompt = f"""You are an accurate translator. Translate the following text from {source_lang} to {target_lang}. Preserve meaning, do not add anything.
Text:
\"\"\"{text}\"\"\""""
    try:
        response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9}
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f" Bedrock translation failed: {e}")
        return text  # fallback: return original


def detect_language(text: str) -> str:
    """
    Detect language by Unicode block or script for common languages.
    Returns ISO-like codes:
    - Hindi (Devanagari) : 'hi'
    - Marathi (Devanagari) : 'mr'
    - Bengali : 'bn'
    - Gujarati : 'gu'
    - Kannada : 'kn'
    - Malayalam : 'ml'
    - Oriya/Odia : 'or'
    - Punjabi (Gurmukhi) : 'pa'
    - Tamil : 'ta'
    - Telugu : 'te'
    - Japanese : 'ja'
    - Chinese : 'zh'
    - Thai : 'th'
    - Spanish : 'es'
    - German : 'de'
    - English : 'en' (default)
    """
    for ch in text:
        # Indian scripts
        if '\u0900' <= ch <= '\u097F':  # Devanagari
            return "hi"
        elif '\u0980' <= ch <= '\u09FF':  # Bengali
            return "bn"
        elif '\u0A80' <= ch <= '\u0AFF':  # Gujarati
            return "gu"
        elif '\u0C80' <= ch <= '\u0CFF':  # Kannada
            return "kn"
        elif '\u0D00' <= ch <= '\u0D7F':  # Malayalam
            return "ml"
        elif '\u0B00' <= ch <= '\u0B7F':  # Odia
            return "or"
        elif '\u0A00' <= ch <= '\u0A7F':  # Gurmukhi
            return "pa"
        elif '\u0B80' <= ch <= '\u0BFF':  # Tamil
            return "ta"
        elif '\u0C00' <= ch <= '\u0C7F':  # Telugu
            return "te"
        
        # East Asian scripts
        elif '\u4E00' <= ch <= '\u9FFF':  # Chinese characters (CJK Unified Ideographs)
            return "zh"
        elif '\u3040' <= ch <= '\u30FF':  # Japanese Hiragana + Katakana
            return "ja"
        elif '\u0E00' <= ch <= '\u0E7F':  # Thai
            return "th"
        
        # Basic Latin letters for Western European languages
        elif ('\u0041' <= ch <= '\u005A') or ('\u0061' <= ch <= '\u007A'):
            # Could be English, Spanish, or German — use naive word detection
            # This is crude; assumes non-ASCII diacritics imply other languages
            if any(c in text for c in 'áéíóúñüßäö'):
                # common Spanish / German characters
                if any(c in text for c in 'áéíóúñ'):  
                    return "es"
                elif any(c in text for c in 'üßäö'):
                    return "de"
            return "en"
    
    return "en"  # fallback

