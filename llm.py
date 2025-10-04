# llm.py
from config import get_bedrock_client, MODEL_ID
import json
import re

bedrock_client = get_bedrock_client()

def detect_language(text: str) -> str:
    """
    Lightweight Unicode-based language detection.
    Returns ISO-like short codes (hi, bn, gu, kn, ml, or, pa, ta, te, ja, zh, th, es, de, en)
    """
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            return "hi"  # Devanagari - use 'hi' as default for Hindi/Marathi
        elif '\u0980' <= ch <= '\u09FF':
            return "bn"
        elif '\u0A80' <= ch <= '\u0AFF':
            return "gu"
        elif '\u0C80' <= ch <= '\u0CFF':
            return "kn"
        elif '\u0D00' <= ch <= '\u0D7F':
            return "ml"
        elif '\u0B00' <= ch <= '\u0B7F':
            return "or"
        elif '\u0A00' <= ch <= '\u0A7F':
            return "pa"
        elif '\u0B80' <= ch <= '\u0BFF':
            return "ta"
        elif '\u0C00' <= ch <= '\u0C7F':
            return "te"
        elif '\u4E00' <= ch <= '\u9FFF':
            return "zh"
        elif '\u3040' <= ch <= '\u30FF':
            return "ja"
        elif '\u0E00' <= ch <= '\u0E7F':
            return "th"
        elif ('\u0041' <= ch <= '\u005A') or ('\u0061' <= ch <= '\u007A'):
            if any(c in text for c in 'áéíóúñüßäö'):
                if any(c in text for c in 'áéíóúñ'):
                    return "es"
                elif any(c in text for c in 'üßäö'):
                    return "de"
            return "en"
    return "en"

def translate_with_llm(text: str, source_lang: str, target_lang: str, max_tokens=512, temperature=0.0):
    """
    Translate using Bedrock model. Returns original text on error (safer fallback).
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
        content = response["output"]["message"]["content"][0]["text"]
        return content.strip()
    except Exception as e:
        print("Bedrock translate failed:", e)
        return text

import os
from config import MODEL_ID, get_bedrock_client

bedrock_client = get_bedrock_client()

def answer_with_llama(prompt: str):
    """Use AWS Bedrock Llama to generate answer."""
    try:
        response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.0, "topP": 0.9}
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f"Bedrock call failed: {e}")
        return "❌ LLM call failed."
