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
