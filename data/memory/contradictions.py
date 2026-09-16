from ollama import chat

def detect_contradictions(previous_summary, current_journal):
    if not previous_summary:
        return "No previous summary to compare."

    prompt = f"""
Compare these two texts and identify contradictions in beliefs.

PREVIOUS SUMMARY:
{previous_summary}

CURRENT JOURNAL:
{current_journal}

Instructions:
- Identify direct contradictions
- Be concise
- If none, say "No contradictions found"
"""

    response = chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def resolve_contradictions(previous_summary, current_journal, contradictions):
    if "No contradictions" in contradictions:
        return current_journal

    prompt = f"""
You previously held these beliefs:
{previous_summary}

You now wrote:
{current_journal}

Contradictions detected:
{contradictions}

Rewrite the journal entry so that:
- Contradictions are acknowledged
- Beliefs evolve logically
- The tone remains philosophical

Return the improved journal only.
"""

    response = chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]