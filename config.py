import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
INDEX_DIR = ROOT / "vector_db"
MAX_INPUT = 6000
