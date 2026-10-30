"""Small shared helpers (formatting, paths, hashing, HTTP JSON)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8", errors="replace"))


def now() -> float:
    return time.time()


def compact_path(path: str, max_len: int = 64) -> str:
    if len(path) <= max_len:
        return path
    head = max_len // 2 - 2
    return path[:head] + "…" + path[-head:]


def canonical_path(path: str | os.PathLike[str]) -> str:
    """Absolute, symlink-resolved path without a trailing slash."""
    return str(Path(path).expanduser().resolve())


def is_within(path: str, root: str) -> bool:
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def dumps(data: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(data, ensure_ascii=False, indent=2, default=json_default)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=json_default)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:.0f}s"


def relative_time(ts: float | None) -> str:
    if not ts:
        return "—"
    delta = max(0.0, time.time() - ts)
    if delta < 10:
        return "刚刚"
    if delta < 60:
        return f"{int(delta)} 秒前"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    if delta < 86400:
        return f"{int(delta // 3600)} 小时前"
    return f"{int(delta // 86400)} 天前"


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def safe_collection_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("-", (name or "").strip()).strip("-.")
    if not cleaned:
        raise ValueError("collection name cannot be empty")
    if len(cleaned) > 64:
        raise ValueError("collection name is too long (max 64)")
    return cleaned


def shorten(text: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def iter_batches(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
