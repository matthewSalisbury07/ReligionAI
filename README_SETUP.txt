SETUP

1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. python download_model.py
5. set TRANSFORMERS_OFFLINE=1
6. ollama 3 8b install
7. torch and other import installs

TEXTS:
Put .txt files in data/texts/

BUILD:
cd embeddings
python build_embeddings.py