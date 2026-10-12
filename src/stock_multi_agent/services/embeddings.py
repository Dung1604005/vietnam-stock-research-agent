from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Protocol
from urllib.request import Request, urlopen

import numpy as np


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class LocalHashEmbeddingProvider:
    """Free deterministic text embeddings for an offline vector-search baseline.

    This is feature hashing, not a pretrained language model. Word, word-bigram
    and character n-gram features make it typo/diacritic tolerant while keeping
    the RAG pipeline usable when an Ollama embedding model is unavailable.
    """

    model_id = "local-hash-vi-v1"
    _ALIASES = {
        "doanh thu": ("revenue", "sales"),
        "loi nhuan": ("profit", "earnings"),
        "bao cao": ("report", "annual report", "financial statement"),
        "rui ro": ("risk", "uncertainty"),
        "tang truong": ("growth", "increase"),
        "dong tien": ("cash flow",),
        "co tuc": ("dividend",),
        "no": ("debt", "liability"),
    }

    def __init__(self, dimensions: int | None = None) -> None:
        configured = dimensions or self._env_int("STOCK_AGENT_RAG_VECTOR_DIMENSIONS", 768)
        self.dimensions = max(128, min(4096, configured))

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        normalized = self._normalize(text)
        tokens = re.findall(r"[a-z0-9]{2,}", normalized)
        features: list[str] = [f"w:{token}" for token in tokens]
        features.extend(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        compact = " ".join(tokens)
        for size in (3, 4, 5):
            features.extend(f"c:{compact[index:index + size]}" for index in range(max(0, len(compact) - size + 1)))
        for phrase, aliases in self._ALIASES.items():
            if phrase in normalized or any(alias in normalized for alias in aliases):
                features.extend(f"a:{alias}" for alias in (phrase, *aliases))

        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            index = raw % self.dimensions
            vector[index] += -1.0 if raw & (1 << 63) else 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    @staticmethod
    def _normalize(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text.casefold())
        return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default


class OllamaEmbeddingProvider:
    """Semantic embeddings through Ollama's local `/api/embed` endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.model_id = f"ollama:{self.model}"
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        try:
            self.timeout_seconds = max(5, min(120, int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))))
        except ValueError:
            self.timeout_seconds = 90

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        body = json.dumps({"model": self.model, "input": texts, "truncate": True}).encode("utf-8")
        request = Request(f"{self.base_url}/api/embed", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured local Ollama
            payload = json.loads(response.read())
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        results: list[np.ndarray] = []
        for values in vectors:
            vector = np.asarray(values, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if not norm:
                raise RuntimeError("Ollama returned an empty embedding")
            results.append(vector / norm)
        return results


def configured_embedding_provider() -> EmbeddingProvider:
    provider = os.getenv("STOCK_AGENT_RAG_EMBEDDING_PROVIDER", "local_hash").strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingProvider()
    return LocalHashEmbeddingProvider()
