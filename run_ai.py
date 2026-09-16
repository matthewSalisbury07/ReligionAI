#===== Imports =====
import os
import json
import datetime
import torch
import numpy as np

from memory.compress import compress_memory
from memory.archive import archive_if_needed
from memory.beliefs import update_beliefs, load_beliefs
from memory.scoring import score_text
from memory.daily_summary import load_daily_summary, update_daily_summary
from memory.score_history import save_scores
from memory.contradictions import detect_contradictions, resolve_contradictions
from memory.idea_graph import update_graph

from transformers import AutoTokenizer, AutoModel
from ollama import chat

# ===== CONFIG =====
MODEL_NAME = "./local_model"
EMBED_FILE = "embeddings/embeddings.json"
JOURNAL_FILE = "journals/myJournal.txt"

# ===== LOAD MODEL =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ===== EMBEDDING FUNCTION =====
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# ===== LOAD TEXT FILES =====
def load_texts():
    base_path = "data/texts"
    lines = []

    for file in os.listdir(base_path):
        if file.endswith(".txt"):
            with open(os.path.join(base_path, file), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)

    return lines

# ===== BUILD EMBEDDINGS =====
def build_embeddings():
    print("Building embeddings...")

    data = load_texts()
    db = []

    for line in data:
        vec = embed(line)
        db.append({"text": line, "vector": vec})

    os.makedirs("embeddings", exist_ok=True)
    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f)

    print(f"Saved {len(db)} embeddings.")

# ===== LOAD EMBEDDINGS =====
def load_embeddings():
    if not os.path.exists(EMBED_FILE):
        build_embeddings()

    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== COSINE SIMILARITY =====
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ===== GET IDEAS =====
def get_ideas(query, db, top_k=5):
    q_vec = embed(query)

    scored = []
    for item in db:
        score = cosine(q_vec, item["vector"])
        scored.append((score, item["text"]))

    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

# ===== LOAD MEMORY =====
def load_memory():
    if not os.path.exists(JOURNAL_FILE):
        return "", ""
    
    with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
        full = f.read()
    
    compressed = compress_memory(full)
    return full, compressed

# ===== GENERATE JOURNAL =====
def generate_journal(ideas):
    full_memory, compressed_memory = load_memory()
    memory = compressed_memory
    daily_summary = load_daily_summary()

    beliefs = load_beliefs()
    belief_text = "\n".join([f"{k}: {v}" for k, v in beliefs.items()])

    prompt = f"""
You are an AI trained ONLY on religious texts.

SOURCE OF TRUTH:
- The ideas below are the ONLY valid knowledge.

CURRENT BELIEFS:
{belief_text}

PAST JOURNAL (style only, NOT knowledge):
{memory[-2000:]}

PREVIOUS DAY SUMMARY (5% influence ONLY):
{daily_summary}

TODAY'S IDEAS (PRIMARY SOURCE):
{chr(10).join("- " + i for i in ideas)}

STRICT RULES:
- Use ONLY TODAY'S IDEAS for reasoning
- Use PREVIOUS DAY SUMMARY only slightly (5%)
- DO NOT learn from or reuse past journal content
- DO NOT introduce ideas not grounded in TODAY'S IDEAS

Write a journal entry that:
- Reflects evolving beliefs
- Shows growth or change
- Connects ideas across time

Rules:
- Be philosophical
- No modern topics
- Stay within religious tone
"""

    response = chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], compressed_memory

# ===== SAVE JOURNAL =====
def save_journal(text, scores):
    os.makedirs("journals", exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {date} =====\n\n")
        f.write(text)

        # ===== SAVE SCORES =====
        f.write("\n\n--- SCORES ---\n")
        for k, v in scores.items():
            f.write(f"{k}: {v}%\n")
        f.write("\n\n--- CONTRADICTIONS ---\n")
        f.write(contradictions + "\n")

    print(f"\nSaved to {JOURNAL_FILE}")

# ===== MAIN =====
def main():
    print("Starting AI system...\n")

    db = load_embeddings()

    query = "judgment morality truth obedience fear mercy justice"
    ideas = get_ideas(query, db)

    print("Selected Ideas:")
    for i in ideas:
        print("-", i)

    # UPDATE IDEA GRAPH
    update_graph(ideas)

    print("\nGenerating journal...\n")

    previous_summary = load_daily_summary()
    journal, compressed_memory = generate_journal(ideas)
    # DETECT
    contradictions = detect_contradictions(previous_summary, journal)
    print("\nContradictions:\n", contradictions)

    # RESOLVE
    journal = resolve_contradictions(previous_summary, journal, contradictions)
    print("\nResolved Journal:\n", journal)

    # ===== SCORING =====
    scores = score_text(journal)

    print("\nCategory Scores:")
    for k, v in scores.items():
        print(f"{k}: {v}%")

    # ===== SAVE JOURNAL =====
    save_journal(journal, scores)

    # ===== UPDATE BELIEFS =====
    beliefs = update_beliefs_from_scores(scores)   
    print("\nUpdated Beliefs:")
    for k, v in beliefs.items():
        print(f"{k}: {v}")

    # LOAD previous summary BEFORE overwriting
    previous_summary = load_daily_summary()

    # ===== DAILY SUMMARY (OVERWRITES ONLY SUMMARY FILE) =====
    update_daily_summary(journal)

    # ===== ARCHIVE CHECK =====
    archive_if_needed(compressed_memory)

# ===== RUN =====
if __name__ == "__main__":
    main()