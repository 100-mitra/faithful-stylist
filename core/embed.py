"""Embedder interface (text).

Default is a dependency-light hashing/TF embedder: offline, deterministic across runs
(hashing via hashlib, not Python's per-process-salted hash()), no torch. For a 60-100
item catalog already narrowed by SQL hard-filters, lexical similarity over enriched tags
is adequate, and grounding does not depend on embedding quality. sentence-transformers
(`minilm`) and Voyage (`voyage`) are opt-in behind the same interface.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from core.config import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 2]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, dim) float32, L2-normalized

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


class HashingTfidfEmbedder:
    """Hashing-trick TF embedder with sublinear term frequency + L2 normalization."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-tf-{self._dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self._dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in _tokens(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                idx, sign = self._bucket(tok)
                out[i, idx] += sign * (1.0 + np.log(c))
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    kind = settings.embedder
    if kind == "minilm":
        return _SentenceTransformerEmbedder()
    if kind == "voyage":
        return _VoyageEmbedder(settings.voyage_api_key)
    return HashingTfidfEmbedder()


class _SentenceTransformerEmbedder:  # pragma: no cover - optional heavy extra
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install the 'st' extra: pip install '.[st]'") from exc
        self._model = SentenceTransformer(model_name)
        self._name = f"st-{model_name}"

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class _VoyageEmbedder:  # pragma: no cover - optional API extra
    def __init__(self, api_key: str | None, model_name: str = "voyage-3"):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY required for the voyage embedder")
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError("Install the 'voyage' extra: pip install '.[voyage]'") from exc
        self._client = voyageai.Client(api_key=api_key)
        self._model = model_name

    @property
    def dim(self) -> int:
        return 1024

    @property
    def name(self) -> str:
        return f"voyage-{self._model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        result = self._client.embed(texts, model=self._model)
        vecs = np.asarray(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)
