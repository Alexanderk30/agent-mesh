"""Shared memory store with similarity search for subtask results."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-numpy cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class _TFIDFEmbedder:
    """Lightweight TF-IDF embedder used when sentence-transformers is unavailable."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0
        self._df: Counter = Counter()

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    def _rebuild_idf(self) -> None:
        self._idf = {
            term: math.log((1 + self._doc_count) / (1 + df)) + 1
            for term, df in self._df.items()
        }

    def add_document(self, text: str) -> None:
        """Register a new document and rebuild IDF weights."""
        tokens = set(self._tokenize(text))
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
            self._df[t] += 1
        self._doc_count += 1
        self._rebuild_idf()

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised TF-IDF vector for *text*."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] = count * self._idf.get(token, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class MemoryStore:
    """Key-value store with vector similarity search over subtask results."""

    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}
        self._embeddings: List[Tuple[str, np.ndarray]] = []

        # Try sentence-transformers first, fall back to TF-IDF
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._strategy = "sentence-transformers"
            self._tfidf: Optional[_TFIDFEmbedder] = None
            print("[MEMORY] Embedding strategy: sentence-transformers (all-MiniLM-L6-v2)")
        except ImportError:
            self._st_model = None
            self._tfidf = _TFIDFEmbedder()
            self._strategy = "tfidf"
            print("[MEMORY] Embedding strategy: TF-IDF fallback")

    def _embed(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            return self._st_model.encode(text, convert_to_numpy=True)
        assert self._tfidf is not None
        return self._tfidf.embed(text)

    def put(self, subtask_id: str, data: dict) -> None:
        """Store *data* under *subtask_id* and index its embedding."""
        self._data[subtask_id] = data
        text = json.dumps(data, default=str)
        if self._tfidf is not None:
            self._tfidf.add_document(text)
        vec = self._embed(text)
        self._embeddings.append((subtask_id, vec))

    def get(self, subtask_id: str) -> Optional[dict]:
        return self._data.get(subtask_id)

    def find_similar(self, query: str, top_k: int = 3) -> List[dict]:
        """Return the *top_k* most similar stored results to *query*."""
        if not self._embeddings:
            return []
        if self._tfidf is not None:
            self._tfidf.add_document(query)
        q_vec = self._embed(query)
        scored: List[Tuple[float, str]] = []
        for sid, vec in self._embeddings:
            # Re-embed stored text if TF-IDF vocab has grown
            if self._tfidf is not None:
                stored_text = json.dumps(self._data[sid], default=str)
                vec = self._tfidf.embed(stored_text)
            sim = _cosine_similarity(q_vec, vec)
            scored.append((sim, sid))
        scored.sort(reverse=True)
        return [self._data[sid] for _, sid in scored[:top_k] if sid in self._data]
