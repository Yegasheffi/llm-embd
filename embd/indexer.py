"""The incremental indexer.

The invalidation ladder, cheapest check first:

1. ``(mtime_ns, size)`` matches the previous index  -> don't even open the file.
2. SHA-256 of the raw bytes matches                  -> update stat only, no chunking.
3. Chunk hashes already exist in the embedding cache -> reuse vectors, embed only
   hashes that are genuinely new. Content-defined boundaries make step 3
   effective after edits in the middle of a file.

All embedding calls happen *before* the database swap, so a provider outage
leaves the previous good index untouched.
"""

from __future__ import annotations

import fnmatch
import os
import stat as stat_module
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .chunker import Chunk, chunk_text
from .config import Config
from .db import Database
from .embedders import BaseEmbedder, EmbeddingError
from .extract import ExtractError, UnsupportedFile, extract_text, is_supported
from .util import canonical_path, is_within, now, sha256_bytes, shorten


ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class IndexSummary:
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_unchanged_hash: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_reused: int = 0
    chunks_removed: int = 0
    embeddings_computed: int = 0
    bytes_read: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=now)
    duration_ms: float = 0.0

    def merge(self, other: "IndexSummary") -> None:
        for key in (
            "files_seen", "files_indexed", "files_unchanged", "files_unchanged_hash",
            "files_skipped", "files_removed", "chunks_added", "chunks_reused",
            "chunks_removed", "embeddings_computed", "bytes_read",
        ):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.errors.extend(other.errors)
        self.skipped.extend(other.skipped)

    def finish(self) -> "IndexSummary":
        self.duration_ms = round((now() - self.started_at) * 1000, 1)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_unchanged": self.files_unchanged,
            "files_unchanged_hash": self.files_unchanged_hash,
            "files_skipped": self.files_skipped,
            "files_removed": self.files_removed,
            "chunks_added": self.chunks_added,
            "chunks_reused": self.chunks_reused,
            "chunks_removed": self.chunks_removed,
            "embeddings_computed": self.embeddings_computed,
            "bytes_read": self.bytes_read,
            "errors": self.errors[:50],
            "skipped": self.skipped[:50],
            "duration_ms": self.duration_ms,
        }


class Indexer:
    def __init__(self, config: Config, db: Database, embedder: BaseEmbedder):
        self.config = config
        self.db = db
        self.embedder = embedder
        self._lock = threading.Lock()

    # --------------------------------------------------------------- discovery
    def collect_files(self, root: str) -> tuple[list[str], list[dict[str, str]], bool]:
        """Collect indexable files under ``root``.

        Returns ``(paths, errors, truncated)``. Directory scans are intentionally
        conservative: hidden directories, VCS metadata, dependency trees and
        binary assets are pruned early.
        """
        root_path = Path(root)
        errors: list[dict[str, str]] = []
        if root_path.is_file():
            return ([str(root_path)] if is_supported(root_path, self.config) else []), errors, False
        if not root_path.is_dir():
            return [], [{"path": root, "error": "path does not exist"}], False

        ignore_dirs = {name.lower() for name in self.config.index["ignore_dirs"]}
        ignore_globs = list(self.config.index["ignore_globs"])
        ignore_globs.extend(_read_embdignore(root_path))
        follow = bool(self.config.index["follow_symlinks"])
        max_files = int(self.config.index["max_files"])

        found: list[str] = []
        truncated = False

        def onerror(exc: OSError) -> None:
            errors.append({"path": getattr(exc, "filename", root) or root, "error": str(exc)})

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=onerror, followlinks=follow):
            dirnames[:] = sorted(
                name for name in dirnames
                if name.lower() not in ignore_dirs
                and not (name.startswith(".") and name not in {".github", ".well-known"})
                and not _ignored(os.path.relpath(os.path.join(dirpath, name), root), name, ignore_globs)
            )
            for filename in sorted(filenames):
                if len(found) >= max_files:
                    truncated = True
                    break
                full = os.path.join(dirpath, filename)
                if filename.startswith(".") and filename not in self.config.index["text_basenames"]:
                    continue
                relative = os.path.relpath(full, root)
                if _ignored(relative, filename, ignore_globs):
                    continue
                try:
                    st = os.stat(full, follow_symlinks=follow)
                except OSError as exc:
                    errors.append({"path": full, "error": str(exc)})
                    continue
                if not stat_module.S_ISREG(st.st_mode):
                    continue
                if st.st_size > int(self.config.index["max_file_bytes"]):
                    continue
                if not is_supported(full, self.config):
                    continue
                found.append(full)
            if truncated:
                break

        return found, errors, truncated

    # ------------------------------------------------------------------ public
    def index_paths(
        self,
        paths: Sequence[str],
        collection: str,
        *,
        force: bool = False,
        watch: bool = True,
        progress: ProgressFn | None = None,
    ) -> IndexSummary:
        summary = IndexSummary()
        with self._lock:
            targets: list[tuple[str, str]] = []
            seen: set[str] = set()

            for raw in paths:
                path = canonical_path(raw)
                if not os.path.exists(path):
                    summary.errors.append({"path": path, "error": "path does not exist"})
                    continue
                if os.path.isdir(path):
                    if watch:
                        self.db.upsert_watch(collection, path, True)
                    files, errors, truncated = self.collect_files(path)
                    summary.errors.extend(errors)
                    if truncated:
                        summary.errors.append(
                            {"path": path, "error": f"directory truncated at {self.config.index['max_files']} files"}
                        )
                    for file_path in files:
                        canonical = canonical_path(file_path)
                        if canonical not in seen:
                            seen.add(canonical)
                            targets.append((canonical, path))
                else:
                    if watch:
                        self.db.upsert_watch(collection, path, False)
                    if path not in seen:
                        seen.add(path)
                        targets.append((path, path))

            total = len(targets)
            for index, (file_path, root) in enumerate(targets, 1):
                if progress:
                    progress({
                        "phase": "indexing",
                        "current": index,
                        "total": total,
                        "path": file_path,
                        "summary": summary.to_dict() | {"duration_ms": 0},
                    })
                file_summary = self.index_file(file_path, collection, root, force=force)
                summary.merge(file_summary)
            return summary.finish()

    def index_file(
        self,
        path: str,
        collection: str,
        root: str,
        *,
        force: bool = False,
    ) -> IndexSummary:
        summary = IndexSummary()
        summary.files_seen = 1
        try:
            st = os.stat(path)
        except OSError as exc:
            summary.files_skipped += 1
            summary.errors.append({"path": path, "error": str(exc)})
            return summary.finish()

        if not stat_module.S_ISREG(st.st_mode):
            summary.files_skipped += 1
            return summary.finish()

        max_bytes = int(self.config.index["max_file_bytes"])
        if st.st_size > max_bytes:
            summary.files_skipped += 1
            summary.skipped.append({"path": path, "reason": f"larger than {max_bytes} bytes"})
            return summary.finish()

        row = self.db.get_file(collection, path)
        if (
            row is not None
            and not force
            and row["status"] == "indexed"
            and int(row["mtime_ns"]) == int(st.st_mtime_ns)
            and int(row["size"]) == int(st.st_size)
        ):
            summary.files_unchanged += 1
            summary.chunks_reused += int(row["chunks"] or 0)
            return summary.finish()

        try:
            with open(path, "rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError as exc:
            summary.errors.append({"path": path, "error": str(exc)})
            self.db.mark_file_error(
                collection=collection, path=path, root=root,
                message=str(exc), mtime_ns=st.st_mtime_ns, size=st.st_size, inode=st.st_ino,
            )
            return summary.finish()

        summary.bytes_read += len(data)
        if len(data) > max_bytes:
            summary.files_skipped += 1
            summary.skipped.append({"path": path, "reason": f"larger than {max_bytes} bytes"})
            return summary.finish()

        content_hash = sha256_bytes(data)

        # Level 2: bytes are identical, only metadata changed (touch, checkout,
        # backup restore with preserved content). No re-chunk, no embedding.
        if row is not None and not force and row["content_hash"] == content_hash and row["status"] == "indexed":
            self.db.touch_file(
                collection=collection, path=path, root=root,
                mtime_ns=st.st_mtime_ns, size=st.st_size, inode=st.st_ino,
                content_hash=content_hash,
            )
            summary.files_unchanged_hash += 1
            summary.chunks_reused += int(row["chunks"] or 0)
            return summary.finish()

        try:
            extracted = extract_text(path, data, self.config)
        except UnsupportedFile as exc:
            summary.files_skipped += 1
            summary.skipped.append({"path": path, "reason": str(exc)})
            return summary.finish()
        except ExtractError as exc:
            summary.errors.append({"path": path, "error": str(exc)})
            self.db.mark_file_error(
                collection=collection, path=path, root=root, message=str(exc),
                mtime_ns=st.st_mtime_ns, size=st.st_size, inode=st.st_ino,
            )
            return summary.finish()

        chunk_config = self.config.chunk
        chunks = chunk_text(
            extracted.text,
            min_bytes=int(chunk_config["min_bytes"]),
            avg_bytes=int(chunk_config["avg_bytes"]),
            max_bytes=int(chunk_config["max_bytes"]),
            snap_bytes=int(chunk_config["snap_bytes"]),
        )

        # Level 3: content-addressed chunk reuse. Only hashes absent from the
        # model's cache need an embedding round-trip.
        unique: dict[str, str] = {}
        for chunk in chunks:
            unique.setdefault(chunk.content_hash, chunk.text)
        cached = self.db.existing_embedding_hashes(self.embedder.fingerprint, unique.keys())
        to_embed = [h for h in unique if h not in cached]

        if to_embed:
            try:
                self._embed_missing(to_embed, unique, summary)
            except (EmbeddingError, ValueError) as exc:
                message = shorten(str(exc), 500)
                summary.errors.append({"path": path, "error": message})
                self.db.mark_file_error(
                    collection=collection, path=path, root=root, message=message,
                    mtime_ns=st.st_mtime_ns, size=st.st_size, inode=st.st_ino,
                )
                return summary.finish()

        previous_chunks = int(row["chunks"] or 0) if row else 0
        self.db.write_file_success(
            collection=collection,
            path=path,
            root=root,
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            inode=st.st_ino,
            content_hash=content_hash,
            chunks=chunks,
            embedded=len(chunks),
        )
        summary.files_indexed += 1
        summary.chunks_added = len(chunks)
        summary.chunks_removed = previous_chunks
        summary.chunks_reused += len(chunks) - len(to_embed)
        return summary.finish()

    def _embed_missing(self, to_embed: list[str], texts: dict[str, str], summary: IndexSummary) -> None:
        batch_size = max(1, int(self.config.embedder["batch_size"]))
        for start in range(0, len(to_embed), batch_size):
            batch_hashes = to_embed[start:start + batch_size]
            vectors = self.embedder.embed([texts[h] for h in batch_hashes])
            if len(vectors) != len(batch_hashes):
                raise EmbeddingError(
                    f"embedder returned {len(vectors)} vectors for {len(batch_hashes)} chunks"
                )
            items: list[tuple[str, bytes]] = []
            for content_hash, vector in zip(batch_hashes, vectors):
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim != 1:
                    raise EmbeddingError("embedder returned a non-1D vector")
                items.append((content_hash, array.tobytes()))
            self.db.store_embeddings(self.embedder.fingerprint, self.embedder.dim, items)
            summary.embeddings_computed += len(items)

    # ------------------------------------------------------------------- watch
    def scan_root(self, watch_row: dict[str, Any], progress: ProgressFn | None = None) -> IndexSummary:
        collection = watch_row["collection"]
        root = watch_row["path"]
        summary = IndexSummary()
        try:
            if watch_row["is_dir"]:
                if not os.path.isdir(root):
                    self.db.remove_watch(collection, root)
                    self.db.watch_failed(collection, root, "directory no longer exists")
                    return summary.finish()
                seen_list, errors, truncated = self.collect_files(root)
                seen = set(seen_list)
                summary.errors.extend(errors)
                if truncated:
                    summary.errors.append({"path": root, "error": "directory scan truncated"})
                stat_map = self.db.file_stat_map(collection, root)
                changed: list[str] = []
                for path in seen_list:
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    row = stat_map.get(path)
                    if (
                        row is None
                        or int(row["mtime_ns"]) != int(st.st_mtime_ns)
                        or int(row["size"]) != int(st.st_size)
                        or row["status"] != "indexed"
                    ):
                        changed.append(path)
                if changed:
                    with self._lock:
                        for index, path in enumerate(changed, 1):
                            if progress:
                                progress({
                                    "phase": "watching",
                                    "current": index,
                                    "total": len(changed),
                                    "path": path,
                                })
                            summary.merge(self.index_file(path, collection, root))
                # Do not infer deletions from a partial scan (permissions,
                # I/O errors): that would silently drop still-valid rows.
                if not errors:
                    removed = self.db.delete_missing_under(collection, root, seen)
                    summary.files_removed += len(removed)
                summary.files_seen += len(seen_list)
            else:
                if not os.path.exists(root):
                    if self.db.delete_file(collection, root):
                        summary.files_removed += 1
                    self.db.remove_watch(collection, root)
                    return summary.finish()
                row = self.db.get_file(collection, root)
                st = os.stat(root)
                if (
                    row is None
                    or int(row["mtime_ns"]) != int(st.st_mtime_ns)
                    or int(row["size"]) != int(st.st_size)
                    or row["status"] != "indexed"
                ):
                    summary.merge(self.index_file(root, collection, root))
            self.db.watch_failed(collection, root, None)
        except Exception as exc:  # watcher must survive one bad root
            message = shorten(f"{type(exc).__name__}: {exc}", 500)
            summary.errors.append({"path": root, "error": message})
            self.db.watch_failed(collection, root, message)
        return summary.finish()

    # ---------------------------------------------------------------- removal
    def remove(self, path: str, collection: str, *, is_dir: bool | None = None) -> dict[str, Any]:
        path = canonical_path(path)
        if is_dir is None:
            is_dir = os.path.isdir(path)
        removed = self.db.delete_files_under(collection, path, is_dir=is_dir)
        self.db.remove_watches_under(collection, path)
        return {"path": path, "collection": collection, "removed": removed, "recursive": is_dir}

    def reindex_collection(self, collection: str, *, force: bool = True) -> IndexSummary:
        summary = IndexSummary()
        rows = self.db.all_file_paths(collection)
        with self._lock:
            for index, row in enumerate(rows, 1):
                if not os.path.exists(row["path"]):
                    if self.db.delete_file(collection, row["path"]):
                        summary.files_removed += 1
                    continue
                summary.merge(
                    self.index_file(row["path"], collection, row["root"], force=force)
                )
                if index % 50 == 0:
                    time.sleep(0)
        return summary.finish()


def _read_embdignore(root: Path) -> list[str]:
    path = root / ".embdignore"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    patterns: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line.lstrip("/"))
    return patterns


def _ignored(relative: str, name: str, patterns: Sequence[str]) -> bool:
    relative = relative.replace(os.sep, "/")
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if "/" in pattern:
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(relative, pattern + "/*"):
                return True
        elif fnmatch.fnmatch(name, pattern):
            return True
    return False
