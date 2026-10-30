"""Embedding providers.

Two families are supported out of the box:

``hash``
    A deterministic, dependency-free hashing embedder. It mixes word tokens,
    CJK uni/bigrams and light sub-word features into a dense unit vector. It is
    not a neural model, but it works offline, is fast, and gives the daemon a
    useful lexical baseline with zero setup.

``ollama`` / ``openai``
    Real embedding models over a local or remote HTTP API. Point ``base_url``
    at an Ollama instance (``http://127.0.0.1:11434``) or any OpenAI-compatible
    ``/v1/embeddings`` endpoint.

Every provider normalises vectors to unit length so cosine similarity is a dot
product. The ``fingerprint`` identifies the vector space; switching models
automatically invalidates the old query vectors without deleting cached
embeddings (switching back is instant and cheap).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Sequence

import numpy as np

from .util import iter_batches


class EmbeddingError(RuntimeError):
    pass


class BaseEmbedder:
    name: str = "base"
    model: str = "base"
    dim: int = 0
    fingerprint: str = "base"
    provider: str = "base"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embed([text])[0], dtype=np.float32)

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "fingerprint": self.fingerprint,
        }


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)
_LATIN_RE = re.compile(r"^[a-z0-9_]+$")
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def _features(text: str) -> Counter[str]:
    """Tokenise mixed Chinese/English text into weighted-ish features."""
    normalised = unicodedata.normalize("NFKC", text or "").lower()
    counts: Counter[str] = Counter()
    for match in _TOKEN_RE.finditer(normalised):
        token = match.group(0)
        if _CJK_RE.match(token):
            # Unigrams help recall; bigrams carry most of the signal.
            for char in token:
                counts[f"c1:{char}"] += 0.35
            for i in range(len(token) - 1):
                counts[f"c2:{token[i:i + 2]}"] += 1.0
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    counts[f"c3:{token[i:i + 3]}"] += 0.25
        elif _LATIN_RE.match(token):
            if len(token) <= 2:
                continue
            counts[f"w:{token}"] += 1.0
            # Light morphology: "indexing" should still touch "index".
            if len(token) >= 5:
                for i in range(len(token) - 3):
                    counts[f"g:{token[i:i + 4]}"] += 0.15
            parts = re.split(r"[_]", token)
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 2:
                        counts[f"w:{part}"] += 0.5
    return counts


class HashEmbedder(BaseEmbedder):
    """Offline hashing embedder (the default)."""

    provider = "hash"

    def __init__(self, *, dim: int = 512, model: str | None = None):
        if dim < 32:
            raise ValueError("hash embedder dim must be >= 32")
        self.dim = int(dim)
        self.model = model or f"hash-{dim}"
        self.name = self.model
        self.fingerprint = f"hash:{self.model}:{self.dim}:v1"
        self._sign_cache: dict[str, tuple[int, float]] = {}

    def _bucket(self, feature: str) -> tuple[int, float]:
        cached = self._sign_cache.get(feature)
        if cached is not None:
            return cached
        if len(self._sign_cache) > 100_000:
            self._sign_cache.clear()
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        bucket = raw % self.dim
        sign = 1.0 if (raw >> 63) & 1 else -1.0
        self._sign_cache[feature] = (bucket, sign)
        return bucket, sign

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            counts = _features(text)
            for feature, tf in counts.items():
                bucket, sign = self._bucket(feature)
                # Sub-linear term frequency keeps long documents comparable
                # with short queries.
                weight = (1.0 + math.log(tf)) if tf > 1.0 else tf
                vector[bucket] += sign * weight
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector /= norm
            out.append(vector.tolist())
        return out


def _normalise(vec: Sequence[float]) -> list[float]:
    array = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.tolist()


class HTTPEmbedder(BaseEmbedder):
    """Ollama or OpenAI-compatible embeddings over HTTP."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dim: int,
        base_url: str | None,
        timeout: float = 60.0,
        batch_size: int = 32,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.provider = provider
        self.model = model
        self.dim = int(dim)
        self.base_url = (base_url or self._default_base_url(provider)).rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = int(batch_size)
        self.api_key_env = api_key_env
        self.name = f"{provider}/{model}"
        self.fingerprint = f"{provider}:{model}:{self.dim}:v1"

    @staticmethod
    def _default_base_url(provider: str) -> str:
        if provider == "ollama":
            return "http://127.0.0.1:11434"
        if provider == "openai":
            return "https://api.openai.com/v1"
        raise EmbeddingError(f"unsupported embedder provider: {provider}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "embd/0.1"}
        if self.provider == "openai":
            api_key = os.environ.get(self.api_key_env or "OPENAI_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise EmbeddingError(f"{self.name}: HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(
                f"{self.name}: cannot reach {url} ({exc.reason}). "
                "Is the embedding server running?"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise EmbeddingError(f"{self.name}: request to {url} failed: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EmbeddingError(f"{self.name}: invalid JSON from {url}") from exc

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "ollama":
            payload = {"model": self.model, "input": texts}
            data = self._post(f"{self.base_url}/api/embed", payload)
            vectors = data.get("embeddings")
            if vectors is None:
                # Older Ollama versions expose /api/embeddings one at a time.
                vectors = []
                for text in texts:
                    single = self._post(
                        f"{self.base_url}/api/embeddings",
                        {"model": self.model, "prompt": text},
                    )
                    vectors.append(single.get("embedding"))
        else:
            payload = {"model": self.model, "input": texts}
            # Only the v3 family supports a dimensions override; sending it to
            # arbitrary OpenAI-compatible servers is a common 400.
            if self.dim and self.model.startswith("text-embedding-3"):
                payload["dimensions"] = self.dim
            data = self._post(f"{self.base_url}/embeddings", payload)
            ordered = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
            vectors = [row.get("embedding") for row in ordered]

        if not vectors or len(vectors) != len(texts):
            raise EmbeddingError(
                f"{self.name}: expected {len(texts)} embeddings, got {len(vectors or [])}"
            )
        cleaned: list[list[float]] = []
        for vector in vectors:
            if not vector:
                raise EmbeddingError(f"{self.name}: empty embedding returned")
            if len(vector) != self.dim:
                raise EmbeddingError(
                    f"{self.name}: model returned dim={len(vector)}, config says {self.dim}. "
                    "Set embedder.dim to match the model."
                )
            cleaned.append(_normalise(vector))
        return cleaned

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for batch in iter_batches(list(texts), self.batch_size):
            out.extend(self._embed_with_retry(batch))
        return out

    def _embed_with_retry(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self._embed_batch(batch)
            except EmbeddingError as exc:
                last_error = exc
                message = str(exc)
                # A 400 usually means the batch is too large for the server;
                # split it instead of retrying the same request.
                if "HTTP 400" in message and len(batch) > 1:
                    mid = len(batch) // 2
                    return self._embed_with_retry(batch[:mid]) + self._embed_with_retry(batch[mid:])
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        assert last_error is not None
        raise last_error


def make_embedder(embedder_config: dict[str, Any]) -> BaseEmbedder:
    provider = str(embedder_config.get("provider", "hash")).lower()
    model = str(embedder_config.get("model") or "")
    dim = int(embedder_config.get("dim") or 0)

    if provider == "hash":
        if not dim:
            dim = 512
        return HashEmbedder(dim=dim, model=model or None)

    if provider in {"ollama", "openai"}:
        if not model:
            raise ValueError(f"embedder.model is required for provider={provider}")
        if not dim:
            defaults = {"ollama": 768, "openai": 1536}
            dim = defaults[provider]
        return HTTPEmbedder(
            provider=provider,
            model=model,
            dim=dim,
            base_url=embedder_config.get("base_url"),
            timeout=float(embedder_config.get("timeout", 60)),
            batch_size=int(embedder_config.get("batch_size", 32)),
            api_key_env=str(embedder_config.get("api_key_env") or "OPENAI_API_KEY"),
        )

    raise ValueError(f"unknown embedder provider: {provider!r} (expected hash, ollama or openai)")
