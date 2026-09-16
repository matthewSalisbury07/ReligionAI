import os
from memory.archive import archive_if_needed
def load_all_texts(folder="Religious Texts"):
    all_lines = []
    base_path = os.path.join(os.path.dirname(__file__), folder)

    for file in os.listdir(base_path):
        if file.endswith(".txt"):
            with open(os.path.join(base_path, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines if l.strip()]
                all_lines.extend(lines)
    return all_lines

if __name__ == "__main__":
    texts = load_all_texts()
    print(f"Loaded {len(texts)} lines")