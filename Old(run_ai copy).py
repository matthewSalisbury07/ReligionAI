#===== Imports =====
import os
import json
import datetime
import torch
import numpy as np
from memory.compress import compress_memory
from memory.archive import archive_if_needed
from memory.beliefs import update_beliefs, load_beliefs
from transformers import AutoTokenizer, AutoModel
from ollama import chat
from data.memory.daily_summary import load_daily_summary, update_daily_summary

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
    
    from memory.compress import compress_memory
    compressed = compress_memory(full)
    return full, compressed

# ===== GENERATE JOURNAL =====
def generate_journal(ideas):
    daily_summary = load_daily_summary()
    full_memory, compressed_memory = load_memory()
    memory = compressed_memory
    beliefs = load_beliefs()
    belief_text = "\n".join([f"{k}: {v}" for k, v in beliefs.items()])
    prompt = f"""
You are an AI trained only on religious texts.

You are developing beliefs over time.

CURRENT BELIEFS:
{belief_text}

PAST REFLECTIONS:
{memory}

TODAY'S IDEAS:
{chr(10).join("- " + i for i in ideas)}

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
    return response["message"]["content"]

# ===== SAVE JOURNAL =====
def save_journal(text):
    os.makedirs("journals", exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {date} =====\n\n")
        f.write(text)

    print(f"\nSaved to {JOURNAL_FILE}")

# ===== ARCHIVE CHECK =====
archive_if_needed(compressed_memory)

# ===== MAIN =====
def main():
    print("Starting AI system...\n")
    db = load_embeddings()

    # You can tweak this to guide the AI's focus
    query = "judgment morality truth obedience fear mercy justice"
    ideas = get_ideas(query, db)
    print("Selected Ideas:")
    for i in ideas:
        print("-", i)

    print("\nGenerating journal...\n")
    beliefs = update_beliefs(journal)
    print("\nUpdated Beliefs:")
    for k, v in beliefs.items():
        print(f"{k}: {v}")

# ===== RUN =====
if __name__ == "__main__":
    main()