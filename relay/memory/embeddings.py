"""Embeddings for customer memory.

`HashingEmbedder` is deterministic and dependency-free (signed feature hashing over word
unigrams and bigrams). It is what tests and the stub provider use. `OpenAIEmbedder` is used
when RELAY_PROVIDER=openai; its output is projected to the configured dimension so both fit the
same pgvector column.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = [t.rstrip("s") if len(t) > 3 else t for t in TOKEN_RE.findall(text.lower())]
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return normalize(vector)


class OpenAIEmbedder:
    def __init__(self, dim: int = 256, model: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            from relay.providers.base import MissingDependencyError

            raise MissingDependencyError("openai", "openai") from exc
        self._client = OpenAI()
        self.dim = dim
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text, dimensions=self.dim)
        return normalize(list(response.data[0].embedding))
