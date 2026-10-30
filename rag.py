import hashlib
import json
import re

import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from config import EMBEDDING_MODEL, INDEX_DIR, OLLAMA_URL, ROOT


def embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


def normalize_text(text):
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def make_chunks(document, method="heading", size=1800, overlap=200):
    if method not in {"heading", "character"}:
        raise ValueError(f"Unknown chunking method: {method}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in document["pages"]:
        text = normalize_text(page["text"])
        # A page boundary is preserved so every chunk has an unambiguous citation.
        sections = re.split(
            r"(?=^(?:MADDE\s+\d+|Madde\s+\d+|[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ /-]{12,})[^\n]*$)",
            text, flags=re.M,
        ) if method == "heading" else [text]
        for section in sections:
            heading = section.split("\n", 1)[0][:160]
            for part in splitter.split_text(section):
                if not part.strip():
                    continue
                chunk_id = hashlib.sha256(
                    f"{document['sha256']}:{page['page']}:{len(chunks)}:{part}".encode()
                ).hexdigest()[:20]
                chunks.append({
                    "id": chunk_id, "text": part, "heading": heading,
                    "document_id": document["document_id"], "title": document["title"],
                    "url": document["final_url"], "page": page["page"],
                    "category": document["category"], "sha256": document["sha256"],
                    "retrieved_at": document["retrieved_at"],
                    "validity_status": document["validity_status"],
                })
    return chunks


class Retriever:
    def __init__(self, directory=INDEX_DIR):
        self.directory = directory
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("Embedding model changed. Rebuild the index first.")
        corpus_hash = hashlib.sha256((ROOT / "data/corpus.json").read_bytes()).hexdigest()
        if self.metadata["corpus_sha256"] != corpus_hash:
            raise ValueError("Selected corpus changed. Rebuild the index first.")
        self.chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index and chunk count differ; rebuild the index.")
        self.embedder = embeddings()

    def search(self, query, k=5):
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        vector = np.asarray([self.embedder.embed_query(query)], dtype="float32")
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Embedding model returned an invalid query vector")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        # Similarity is not a probability or a legal reliability score.
        return [dict(self.chunks[int(i)], similarity=float(score))
                for score, i in zip(scores[0], indices[0]) if i >= 0]


def build_index(method="heading"):
    manifest_path = ROOT / "data" / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = []
    for entry in manifest:
        if entry["status"] != "ok":
            continue
        document = json.loads((ROOT / entry["text_path"]).read_text(encoding="utf-8"))
        chunks.extend(make_chunks(document, method))
    if not chunks:
        raise ValueError("No extracted text available; run collect_documents.py first")
    embedder = embeddings()
    vectors = []
    for start in range(0, len(chunks), 24):
        batch = chunks[start:start + 24]
        vectors.extend(embedder.embed_documents([c["text"] for c in batch]))
        print(f"Embedding {min(start + 24, len(chunks))}/{len(chunks)}", flush=True)
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Embedding model returned invalid document vectors")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    # Publish metadata last; failed embedding never replaces an existing index.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (INDEX_DIR / "metadata.json").write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL, "method": method,
        "documents": len(manifest), "chunks": len(chunks),
        "dimension": matrix.shape[1],
        "corpus_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"Index saved: {INDEX_DIR}")
