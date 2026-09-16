from ollama import chat

def compress_memory(text):
    if len(text) < 2000:
        return text

    prompt = f"Summarize key beliefs and themes:\n{text}"

    response = chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]