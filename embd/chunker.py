"""Content-defined chunking with semantic boundary snapping.

Why this file matters
---------------------
Most RAG pipelines split text into fixed-size windows. Insert one paragraph at
the top of a document and every downstream window shifts, so every embedding
must be recomputed. That is the expensive part.

Here, chunk boundaries are chosen by a FastCDC-style gear hash over the raw
bytes. The hash has a short effective window, so an edit only perturbs nearby
candidate boundaries; after that the cutter re-synchronises with the previous
version. Boundaries are then snapped to a nearby paragraph / line / sentence
break (within a small window) to keep chunks readable for retrieval.

The result is *content-addressed* chunks: a chunk's SHA-256 depends only on its
text, not on its position in the file. The indexer can therefore reuse an
embedding whenever the same text appears again, even in another file.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable


MASK64 = (1 << 64) - 1


def _splitmix64(seed: int) -> int:
    seed = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = seed
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


# A fixed gear table. Regenerating it with the same seed on every run is
# deterministic across platforms and Python versions.
GEAR: tuple[int, ...] = tuple(_splitmix64(i * 0x100000001B3 + 0x1234) for i in range(256))


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    start: int  # byte offset in the UTF-8 encoded source
    end: int
    text: str
    content_hash: str

    @property
    def byte_length(self) -> int:
        return self.end - self.start


def _masks(avg_bytes: int) -> tuple[int, int]:
    bits = max(4, min(30, int(round(math.log2(max(16, avg_bytes))))))
    hard = (1 << bits) - 1
    easy = (1 << max(1, bits - 2)) - 1
    return hard, easy


def _raw_cut(data: bytes, start: int, min_size: int, avg_size: int, max_size: int) -> int:
    """Return a byte offset strictly inside ``(start, start+max_size]``."""
    total = len(data)
    if total - start <= max_size:
        return total

    hard_mask, easy_mask = _masks(avg_size)
    scan_from = start + min_size
    normal_at = start + avg_size
    scan_to = min(start + max_size, total)

    h = 0
    for j in range(start, scan_from):
        h = ((h << 1) + GEAR[data[j]]) & MASK64

    for j in range(scan_from, scan_to):
        h = ((h << 1) + GEAR[data[j]]) & MASK64
        mask = hard_mask if j < normal_at else easy_mask
        if (h & mask) == 0:
            return j + 1
    return scan_to


_SEPARATORS = (
    b"\n\n",
    b"\r\n\r\n",
    b"\n",
    b"\r\n",
    "。".encode(),
    "！".encode(),
    "？".encode(),
    b". ",
    b"! ",
    b"? ",
)


def _snap(data: bytes, start: int, cut: int, min_size: int, max_size: int, window: int) -> int:
    """Nudge a cut towards a human-friendly separator without breaking limits."""
    if window <= 0 or cut >= len(data):
        return cut

    low = max(start + min_size, cut - window)
    high = min(start + max_size, len(data), cut + window)
    if high <= low:
        return cut

    region = data[low:high]
    best: tuple[int, int] | None = None  # (kind_rank, offset)
    for kind, sep in enumerate(_SEPARATORS):
        positions = _all_occurrences(region, sep)
        for p in positions:
            absolute = low + p + len(sep)
            if absolute < start + min_size or absolute > start + max_size:
                continue
            distance = abs(absolute - cut)
            # Prefer strong separators, then proximity. Cap the distance cost
            # so a weak separator right at the cut can still win.
            score = kind * (window // 2) + distance
            if best is None or score < best[0]:
                best = (score, absolute)
    return best[1] if best else cut


def _all_occurrences(haystack: bytes, needle: bytes) -> Iterable[int]:
    if not needle:
        return
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return
        yield idx
        start = idx + 1


def _utf8_safe_cut(data: bytes, cut: int) -> int:
    """Never cut inside a multi-byte UTF-8 sequence."""
    while 0 < cut < len(data) and (data[cut] & 0xC0) == 0x80:
        cut -= 1
    return cut


def chunk_text(
    text: str,
    *,
    min_bytes: int = 384,
    avg_bytes: int = 1200,
    max_bytes: int = 4800,
    snap_bytes: int = 320,
    include_whitespace_only: bool = False,
) -> list[Chunk]:
    """Split ``text`` into deterministic, content-addressed chunks.

    The offsets are byte offsets into ``text.encode("utf-8")``. That keeps
    slicing exact even for CJK text and lets callers map back to a file.
    """
    if min_bytes < 1 or avg_bytes < min_bytes or max_bytes < avg_bytes:
        raise ValueError("chunk sizes must satisfy 0 < min <= avg <= max")

    data = text.encode("utf-8", errors="replace")
    if not data:
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    total = len(data)

    while start < total:
        if total - start <= max_bytes:
            cut = total
        else:
            cut = _raw_cut(data, start, min_bytes, avg_bytes, max_bytes)
            cut = _snap(data, start, cut, min_bytes, max_bytes, snap_bytes)
        cut = _utf8_safe_cut(data, cut)
        if cut <= start:  # defensive; should not happen with sane settings
            cut = min(total, start + max_bytes)
            cut = _utf8_safe_cut(data, cut)

        raw = data[start:cut]
        body = raw.decode("utf-8", errors="replace")
        if include_whitespace_only or body.strip():
            digest = hashlib.sha256(raw).hexdigest()
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    start=start,
                    end=cut,
                    text=body,
                    content_hash=digest,
                )
            )
            ordinal += 1
        start = cut

    return chunks


def chunk_stats(data: bytes, chunks: list[Chunk]) -> dict[str, float | int]:
    sizes = [c.byte_length for c in chunks] or [0]
    return {
        "count": len(chunks),
        "bytes": len(data),
        "min": min(sizes) if chunks else 0,
        "max": max(sizes) if chunks else 0,
        "avg": round(sum(sizes) / len(sizes), 1) if chunks else 0,
    }
