from ollama import chat
import os

SUMMARY_FILE = "memory/daily_summary.txt"

def load_daily_summary():
    if not os.path.exists(SUMMARY_FILE):
        return ""
    return open(SUMMARY_FILE, "r", encoding="utf-8").read()

def update_daily_summary(journal_text):
    prompt = f"""
Summarize this journal into core beliefs and ideas.

Rules:
- Keep it short
- Abstract ideas only
- No repetition

TEXT:
{journal_text}
"""

    response = chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response["message"]["content"]

    os.makedirs("memory", exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary