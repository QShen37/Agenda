# embedding.py
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_task_text(stem, options=None):
    return stem

def embed_task(task_text):
    emb = _model.encode(task_text, normalize_embeddings=True)
    return emb
