"""
Lightweight local RAG pipeline.
- Ingests markdown documents from data/docs/
- Chunks text with overlap
- Builds token-frequency vectors (no external embedding API needed)
- Retrieves by cosine similarity
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List


# ── Config ─────────────────────────────────────────────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
CHUNK_SIZE = 400       # characters
CHUNK_OVERLAP = 80     # characters
TOP_K = 3


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    vector: dict[str, float] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def _tf_idf_vector(tokens: List[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {w: (count / total) * idf.get(w, 1.0) for w, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b + 1e-10)


# ── Vector store ─────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.idf: dict[str, float] = {}
        self._built = False

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_docs(self, docs_dir: str = DOCS_DIR):
        raw_chunks: List[tuple[str, str]] = []  # (text, source)

        for fname in os.listdir(docs_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            # chunk with overlap
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end].strip()
                if chunk_text:
                    raw_chunks.append((chunk_text, fname))
                start += CHUNK_SIZE - CHUNK_OVERLAP

        # build IDF
        doc_freq: dict[str, int] = Counter()
        all_token_lists = []
        for text, _ in raw_chunks:
            tokens = _tokenize(text)
            all_token_lists.append(tokens)
            for w in set(tokens):
                doc_freq[w] += 1

        N = len(raw_chunks) or 1
        self.idf = {w: math.log(N / (df + 1)) + 1 for w, df in doc_freq.items()}

        # build chunks with vectors
        for (text, source), tokens in zip(raw_chunks, all_token_lists):
            vec = _tf_idf_vector(tokens, self.idf)
            self.chunks.append(Chunk(text=text, source=source, vector=vec))

        self._built = True

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        if not self._built:
            self.ingest_docs()

        q_tokens = _tokenize(query)
        q_vec = _tf_idf_vector(q_tokens, self.idf)

        scored = [(chunk, _cosine(q_vec, chunk.vector)) for chunk in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


# Singleton
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.ingest_docs()
    return _store
