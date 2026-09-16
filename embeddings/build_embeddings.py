import json
import torch
from transformers import AutoTokenizer, AutoModel
from config import MODEL_NAME
from data.load_texts import load_all_texts

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

data = load_all_texts()

db = []

for line in data:
    vec = embed(line)
    db.append({"text": line, "vector": vec})

json.dump(db, open("embeddings.json", "w"))
