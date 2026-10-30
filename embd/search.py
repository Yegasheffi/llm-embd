"""Hybrid retrieval over the local index.

Vector search and FTS5 trigram search run independently, then fuse with
reciprocal-rank fusion. The vector matrix is cached in memory and rebuilt when
the database generation changes, so the CLI and watcher can write concurrently
without serving stale neighbours.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import Config
from .db import Database
from .embedders import BaseEmbedder, EmbeddingError
from .util import shorten


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


@dataclass
class VectorEntry:
    ids: np.ndarray
    matrix: np.ndarray
    meta: list[dict[str, Any]]
    generation: int


class VectorCache:
    def __init__(self, db: Database, *, max_entries: int = 4):
        self.db = db
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str, str | None], VectorEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, collection: str, fingerprint: str, dim: int, path_prefix: str | None) -> VectorEntry:
        generation = self.db.generation()
        key = (collection, fingerprint, path_prefix)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.generation == generation:
                self._entries.move_to_end(key)
                return entry

            rows = self.db.vector_rows(collection, fingerprint, path_prefix)
            ids: list[int] = []
            vectors: list[np.ndarray] = []
            meta: list[dict[str, Any]] = []
            for row in rows:
                vector = np.frombuffer(row["vector"], dtype=np.float32)
                if vector.size != dim:
                    continue
                ids.append(int(row["id"]))
                vectors.append(vector)
                meta.append({
                    "id": int(row["id"]),
                    "path": row["path"],
                    "content_hash": row["content_hash"],
                    "start": int(row["start"]),
                    "ordinal": int(row["ordinal"]),
                })
            matrix = np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
            entry = VectorEntry(
                ids=np.asarray(ids, dtype=np.int64),
                matrix=matrix,
                meta=meta,
                generation=generation,
            )
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class Searcher:
    def __init__(self, config: Config, db: Database, embedder: BaseEmbedder):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.cache = VectorCache(db)
        self._query_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._query_lock = threading.Lock()

    def _query_vector(self, query: str) -> np.ndarray:
        key = (self.embedder.fingerprint, query)
        with self._query_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return cached
        vector = self.embedder.embed_query(query)
        with self._query_lock:
            self._query_cache[key] = vector
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > 256:
                self._query_cache.popitem(last=False)
        return vector

    def search(
        self,
        query: str,
        *,
        collection: str = "default",
        limit: int | None = None,
        mode: str | None = None,
        path_prefix: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = (query or "").strip()
        default_limit = int(self.config.search["default_limit"])
        limit = max(1, min(100, int(limit or default_limit)))
        if mode is None:
            mode = "hybrid" if self.config.search["hybrid"] else "vector"
        mode = mode.lower()
        if mode not in {"hybrid", "vector", "lexical"}:
            raise ValueError("mode must be one of: hybrid, vector, lexical")

        response: dict[str, Any] = {
            "query": query,
            "collection": collection,
            "mode": mode,
            "limit": limit,
            "results": [],
            "count": 0,
            "took_ms": 0.0,
            "model": self.embedder.name,
            "fingerprint": self.embedder.fingerprint,
            "warnings": [],
        }
        if not query:
            response["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response

        candidate_limit = max(limit * 6, int(self.config.search["candidate_limit"]))
        vector_hits: dict[int, float] = {}
        lexical_hits: dict[int, float] = {}
        lexical_ranks: dict[int, int] = {}

        if mode in {"hybrid", "vector"}:
            try:
                query_vector = self._query_vector(query)
                entry = self.cache.get(
                    collection, self.embedder.fingerprint, self.embedder.dim, path_prefix
                )
                if entry.ids.size:
                    # Accelerate BLAS on macOS can emit spurious FP warnings for
                    # perfectly finite matmuls; results are checked below.
                    with np.errstate(all="ignore"):
                        scores = entry.matrix @ query_vector.astype(np.float32)
                    scores = np.nan_to_num(scores, copy=False)
                    if entry.ids.size > candidate_limit:
                        candidate = np.argpartition(scores, -candidate_limit)[-candidate_limit:]
                    else:
                        candidate = np.arange(entry.ids.size)
                    ordered = candidate[np.argsort(scores[candidate])[::-1]]
                    min_score = float(self.config.search.get("min_vector_score", 0.0))
                    for rank, idx in enumerate(ordered, 1):
                        score = float(scores[idx])
                        if score > min_score:
                            vector_hits[int(entry.ids[idx])] = score
                else:
                    response["warnings"].append(
                        "当前模型还没有向量；先用 embd index 建立索引，或检查 embedder 配置。"
                    )
            except EmbeddingError as exc:
                response["warnings"].append(f"向量检索失败，已降级为关键词检索：{shorten(str(exc), 240)}")
                if mode == "vector":
                    mode = "lexical"
                    response["mode"] = mode

        if mode in {"hybrid", "lexical"}:
            fts_query = build_fts_query(query)
            found: list[tuple[int, float]] = []
            if fts_query:
                try:
                    found = self.db.lexical_search(
                        collection, fts_query, candidate_limit, path_prefix
                    )
                except Exception as exc:  # FTS syntax/edge-case fallback
                    response["warnings"].append(f"关键词索引查询失败：{shorten(str(exc), 160)}")
                    found = []
            if not found:
                found = self.db.like_search(
                    collection, query[:80], candidate_limit, path_prefix
                )
            for rank, (chunk_id, bm25) in enumerate(found, 1):
                lexical_hits[chunk_id] = bm25
                lexical_ranks[chunk_id] = rank

        if not vector_hits and not lexical_hits:
            response["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response

        fused: dict[int, float] = {}
        k = 60.0
        vector_weight = float(self.config.search["vector_weight"])
        lexical_weight = float(self.config.search["lexical_weight"])
        if mode == "vector":
            vector_weight, lexical_weight = 1.0, 0.0
        elif mode == "lexical":
            vector_weight, lexical_weight = 0.0, 1.0

        vector_ranks = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(
                sorted(vector_hits.items(), key=lambda item: item[1], reverse=True), 1
            )
        }
        for chunk_id, rank in vector_ranks.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + vector_weight / (k + rank)
        for chunk_id, rank in lexical_ranks.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + lexical_weight / (k + rank)

        ordered_ids = sorted(fused, key=fused.get, reverse=True)[: max(limit * 6, limit)]
        chunks = self.db.fetch_chunks(ordered_ids)
        needle = query.casefold()
        results: list[dict[str, Any]] = []
        per_path: dict[str, int] = {}
        max_per_path = max(1, int(self.config.search.get("max_per_path", 2)))
        for chunk_id in ordered_ids:
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue
            path = chunk["path"]
            if per_path.get(path, 0) >= max_per_path:
                continue
            per_path[path] = per_path.get(path, 0) + 1
            score = fused[chunk_id]
            text = chunk["text"]
            if needle and needle in text.casefold():
                score += 0.01
            results.append({
                "id": chunk_id,
                "path": path,
                "collection": chunk["collection"],
                "ordinal": chunk["ordinal"],
                "start": chunk["start"],
                "end": chunk["end"],
                "content_hash": chunk["content_hash"],
                "score": round(score, 6),
                "vector_score": round(vector_hits.get(chunk_id, 0.0), 6),
                "lexical_rank": lexical_ranks.get(chunk_id),
                "snippet": make_snippet(text, query),
                "text": text,
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        response["results"] = results[:limit]
        response["count"] = len(results[:limit])
        response["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return response


def build_fts_query(query: str) -> str | None:
    normalised = unicodedata.normalize("NFKC", query).lower()
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            terms.append(term)

    for match in _TOKEN_RE.finditer(normalised):
        token = match.group(0)
        if _CJK_RE.match(token):
            if len(token) >= 3:
                # Trigram FTS supports substring search; ORing the windows
                # keeps long Chinese queries from becoming an all-or-nothing
                # phrase.
                add(token)
                for i in range(len(token) - 2):
                    add(token[i:i + 3])
            # 1-2 char CJK terms cannot be served by the trigram index; the
            # caller falls back to LIKE when nothing is found.
        elif len(token) >= 3:
            add(token)
    terms = terms[:24]
    if not terms:
        return None
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def make_snippet(text: str, query: str, limit: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    lower = clean.casefold()
    positions: list[int] = []
    q = query.strip().casefold()
    if q:
        pos = lower.find(q)
        if pos >= 0:
            positions.append(pos)
    for match in _TOKEN_RE.finditer(q):
        pos = lower.find(match.group(0))
        if pos >= 0:
            positions.append(pos)
    if positions:
        centre = min(positions)
        start = max(0, centre - 80)
        end = min(len(clean), start + limit)
        prefix = "…" if start else ""
        suffix = "…" if end < len(clean) else ""
        return prefix + clean[start:end].strip() + suffix
    return clean[: limit - 1].rstrip() + "…"
