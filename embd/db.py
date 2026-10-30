"""SQLite storage for metadata, chunks, FTS and vector blobs.

One file holds everything: file fingerprints, content-addressed chunks, an FTS5
trigram index for lexical/hybrid retrieval, and a global embedding cache keyed
by ``(content_hash, model_fingerprint)``. WAL mode lets the CLI and daemon read
while the watcher writes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .util import human_bytes, now


SCHEMA_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    path TEXT NOT NULL,
    root TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    inode INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error TEXT,
    chunks INTEGER NOT NULL DEFAULT 0,
    embedded INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL,
    UNIQUE(collection, path)
);
CREATE INDEX IF NOT EXISTS idx_files_collection ON files(collection, path);
CREATE INDEX IF NOT EXISTS idx_files_root ON files(collection, root);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    path TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection, id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (content_hash, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_fp ON embeddings(fingerprint);

CREATE TABLE IF NOT EXISTS watch_roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    path TEXT NOT NULL,
    is_dir INTEGER NOT NULL,
    created_at REAL NOT NULL,
    last_scan_at REAL,
    last_error TEXT,
    UNIQUE(collection, path)
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self._migrate()

    # ------------------------------------------------------------------ setup
    def _migrate(self) -> None:
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{version} is newer than this embd build "
                f"(v{SCHEMA_VERSION}); upgrade embd or use a new EMBD_HOME"
            )
        if version < 1:
            self.conn.executescript(SCHEMA)
            self.conn.execute("PRAGMA user_version=1")
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('generation', '1')"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('embedded_total', '0')"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('chunks_written_total', '0')"
        )

    @contextmanager
    def _tx(self) -> Iterator[None]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------------- meta
    def _meta_get(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def meta_get(self, key: str, default: str = "") -> str:
        with self.lock:
            return self._meta_get(key, default)

    def meta_set(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def generation(self) -> int:
        with self.lock:
            try:
                return int(self._meta_get("generation", "1"))
            except ValueError:
                return 1

    def _bump_generation(self) -> None:
        self.conn.execute(
            "UPDATE meta SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT) "
            "WHERE key='generation'"
        )

    def bump_generation(self) -> None:
        with self.lock:
            self._bump_generation()

    # ------------------------------------------------------------ collections
    def ensure_collection(self, name: str) -> bool:
        with self._tx():
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO collections(name,created_at,updated_at) VALUES(?,?,?)",
                (name, now(), now()),
            )
            created = cursor.rowcount > 0
            if created:
                self._bump_generation()
            return created

    def collection_exists(self, name: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM collections WHERE name=?", (name,)
            ).fetchone()
            return row is not None

    def collections(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT c.name, c.created_at, c.updated_at,
                       COUNT(f.id) AS files,
                       COALESCE(SUM(f.chunks), 0) AS chunks,
                       COALESCE(SUM(CASE WHEN f.status='error' THEN 1 ELSE 0 END), 0) AS errors,
                       MAX(f.indexed_at) AS last_indexed
                FROM collections c
                LEFT JOIN files f ON f.collection = c.name
                GROUP BY c.name
                ORDER BY c.name
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_collection(self, name: str) -> bool:
        with self._tx():
            files = self.conn.execute(
                "SELECT id FROM files WHERE collection=?", (name,)
            ).fetchall()
            for row in files:
                self.conn.execute("DELETE FROM chunks WHERE file_id=?", (row["id"],))
            self.conn.execute("DELETE FROM files WHERE collection=?", (name,))
            self.conn.execute("DELETE FROM watch_roots WHERE collection=?", (name,))
            cursor = self.conn.execute("DELETE FROM collections WHERE name=?", (name,))
            deleted = cursor.rowcount > 0
            if deleted:
                self._bump_generation()
            return deleted

    # ------------------------------------------------------------------ files
    def get_file(self, collection: str, path: str) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM files WHERE collection=? AND path=?", (collection, path)
            ).fetchone()

    def file_rows(self, collection: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, collection, path, root, size, status, error, chunks,
                       embedded, indexed_at, content_hash
                FROM files WHERE collection=? ORDER BY indexed_at DESC LIMIT ? OFFSET ?
                """,
                (collection, limit, offset),
            ).fetchall()
            return [dict(row) for row in rows]

    def file_stat_map(self, collection: str, root: str | None = None) -> dict[str, sqlite3.Row]:
        with self.lock:
            if root:
                rows = self.conn.execute(
                    "SELECT * FROM files WHERE collection=? AND root=?", (collection, root)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM files WHERE collection=?", (collection,)
                ).fetchall()
            return {row["path"]: row for row in rows}

    def all_file_paths(self, collection: str) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT id, collection, path, root, mtime_ns, size, status FROM files "
                "WHERE collection=? ORDER BY path",
                (collection,),
            ).fetchall()

    def touch_file(
        self,
        *,
        collection: str,
        path: str,
        root: str,
        mtime_ns: int,
        size: int,
        inode: int,
        content_hash: str | None,
    ) -> None:
        with self._tx():
            self.conn.execute(
                """
                UPDATE files
                SET mtime_ns=?, size=?, inode=?, content_hash=?, status='indexed',
                    error=NULL, indexed_at=?
                WHERE collection=? AND path=?
                """,
                (mtime_ns, size, inode, content_hash, now(), collection, path),
            )
            self.conn.execute(
                "UPDATE collections SET updated_at=? WHERE name=?", (now(), collection)
            )
            self._bump_generation()

    def write_file_success(
        self,
        *,
        collection: str,
        path: str,
        root: str,
        mtime_ns: int,
        size: int,
        inode: int,
        content_hash: str,
        chunks: Sequence[Any],
        embedded: int,
    ) -> None:
        """Atomically swap a file's chunks (embedding work happens before this)."""
        timestamp = now()
        with self._tx():
            self.conn.execute(
                "INSERT OR IGNORE INTO collections(name,created_at,updated_at) VALUES(?,?,?)",
                (collection, timestamp, timestamp),
            )
            row = self.conn.execute(
                "SELECT id FROM files WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
            if row:
                file_id = int(row["id"])
                self.conn.execute(
                    """
                    UPDATE files SET root=?, mtime_ns=?, size=?, inode=?, content_hash=?,
                        status='indexed', error=NULL, chunks=?, embedded=?, indexed_at=?
                    WHERE id=?
                    """,
                    (root, mtime_ns, size, inode, content_hash, len(chunks), embedded, timestamp, file_id),
                )
            else:
                cursor = self.conn.execute(
                    """
                    INSERT INTO files(collection,path,root,mtime_ns,size,inode,content_hash,
                                      status,error,chunks,embedded,indexed_at)
                    VALUES(?,?,?,?,?,?,?,'indexed',NULL,?,?,?)
                    """,
                    (
                        collection, path, root, mtime_ns, size, inode, content_hash,
                        len(chunks), embedded, timestamp,
                    ),
                )
                file_id = int(cursor.lastrowid)

            # The FTS delete/insert triggers keep the lexical index in sync.
            self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            self.conn.executemany(
                """
                INSERT INTO chunks(file_id,collection,path,ordinal,start,end,content_hash,text)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        file_id, collection, path, chunk.ordinal, chunk.start,
                        chunk.end, chunk.content_hash, chunk.text,
                    )
                    for chunk in chunks
                ],
            )
            self.conn.execute(
                "UPDATE meta SET value=CAST(CAST(value AS INTEGER)+? AS TEXT) "
                "WHERE key='chunks_written_total'",
                (len(chunks),),
            )
            self.conn.execute(
                "UPDATE collections SET updated_at=? WHERE name=?", (timestamp, collection)
            )
            self._bump_generation()

    def mark_file_error(
        self,
        *,
        collection: str,
        path: str,
        root: str,
        message: str,
        mtime_ns: int = 0,
        size: int = 0,
        inode: int = 0,
    ) -> None:
        timestamp = now()
        with self._tx():
            self.conn.execute(
                "INSERT OR IGNORE INTO collections(name,created_at,updated_at) VALUES(?,?,?)",
                (collection, timestamp, timestamp),
            )
            row = self.conn.execute(
                "SELECT id FROM files WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE files SET status='error', error=?, mtime_ns=?, size=?, inode=?, indexed_at=? "
                    "WHERE id=?",
                    (message[:2000], mtime_ns, size, inode, timestamp, row["id"]),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO files(collection,path,root,mtime_ns,size,inode,status,error,indexed_at)
                    VALUES(?,?,?,?,?,?,'error',?,?)
                    """,
                    (collection, path, root, mtime_ns, size, inode, message[:2000], timestamp),
                )
            self._bump_generation()

    def delete_file(self, collection: str, path: str) -> bool:
        with self._tx():
            row = self.conn.execute(
                "SELECT id FROM files WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
            if not row:
                return False
            self.conn.execute("DELETE FROM chunks WHERE file_id=?", (row["id"],))
            self.conn.execute("DELETE FROM files WHERE id=?", (row["id"],))
            self._bump_generation()
            return True

    def delete_files_under(self, collection: str, path: str, *, is_dir: bool) -> int:
        """Delete index rows for a file, or every file below a directory."""
        with self.lock:
            if is_dir:
                prefix = path.rstrip(os.sep) + os.sep
                rows = self.conn.execute(
                    "SELECT id FROM files WHERE collection=? AND (path=? OR path LIKE ? ESCAPE '\\')",
                    (collection, path, _escape_like(prefix) + "%"),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT id FROM files WHERE collection=? AND path=?", (collection, path)
                ).fetchall()
        deleted = 0
        with self._tx():
            for row in rows:
                # Delete chunks explicitly so the FTS delete trigger runs.
                self.conn.execute("DELETE FROM chunks WHERE file_id=?", (row["id"],))
                self.conn.execute("DELETE FROM files WHERE id=?", (row["id"],))
                deleted += 1
            if deleted:
                self._bump_generation()
        return deleted

    def remove_watches_under(self, collection: str, path: str) -> int:
        prefix = path.rstrip(os.sep) + os.sep
        with self._tx():
            cursor = self.conn.execute(
                "DELETE FROM watch_roots WHERE collection=? AND (path=? OR path LIKE ? ESCAPE '\\')",
                (collection, path, _escape_like(prefix) + "%"),
            )
            return cursor.rowcount

    def delete_missing_under(self, collection: str, root: str, seen_paths: set[str]) -> list[str]:
        """Remove rows previously rooted at ``root`` that disappeared from disk."""
        removed: list[str] = []
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, path FROM files WHERE collection=? AND root=?",
                (collection, root),
            ).fetchall()
        for row in rows:
            if row["path"] not in seen_paths:
                if self.delete_file(collection, row["path"]):
                    removed.append(row["path"])
        return removed

    # ----------------------------------------------------------------- chunks
    def existing_embedding_hashes(self, fingerprint: str, hashes: Iterable[str]) -> set[str]:
        wanted = list(dict.fromkeys(hashes))
        found: set[str] = set()
        with self.lock:
            for i in range(0, len(wanted), 500):
                batch = wanted[i:i + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = self.conn.execute(
                    f"SELECT content_hash FROM embeddings WHERE fingerprint=? "
                    f"AND content_hash IN ({placeholders})",
                    (fingerprint, *batch),
                ).fetchall()
                found.update(row[0] for row in rows)
        return found

    def store_embeddings(self, fingerprint: str, dim: int, items: Sequence[tuple[str, bytes]]) -> int:
        if not items:
            return 0
        timestamp = now()
        with self._tx():
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO embeddings(content_hash,fingerprint,dim,vector,created_at) "
                "VALUES(?,?,?,?,?)",
                [(content_hash, fingerprint, dim, vector, timestamp) for content_hash, vector in items],
            )
            inserted = self.conn.total_changes - before
            if inserted:
                self.conn.execute(
                    "UPDATE meta SET value=CAST(CAST(value AS INTEGER)+? AS TEXT) "
                    "WHERE key='embedded_total'",
                    (inserted,),
                )
                self._bump_generation()
        return inserted

    def embedding_count(self, fingerprint: str | None = None) -> int:
        with self.lock:
            if fingerprint:
                row = self.conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
            else:
                row = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0])

    def vector_rows(
        self, collection: str, fingerprint: str, path_prefix: str | None = None
    ) -> list[sqlite3.Row]:
        query = """
            SELECT c.id, c.path, c.content_hash, c.start, c.ordinal, e.vector
            FROM chunks c
            JOIN embeddings e ON e.content_hash = c.content_hash AND e.fingerprint = ?
            WHERE c.collection = ?
        """
        params: list[Any] = [fingerprint, collection]
        if path_prefix:
            query += " AND c.path LIKE ? ESCAPE '\\'"
            params.append(_like_prefix(path_prefix))
        query += " ORDER BY c.id"
        with self.lock:
            return self.conn.execute(query, params).fetchall()

    def vector_count(self, collection: str, fingerprint: str) -> int:
        with self.lock:
            row = self.conn.execute(
                """
                SELECT COUNT(*) FROM chunks c
                JOIN embeddings e ON e.content_hash=c.content_hash AND e.fingerprint=?
                WHERE c.collection=?
                """,
                (fingerprint, collection),
            ).fetchone()
            return int(row[0])

    def lexical_search(
        self,
        collection: str,
        match_query: str,
        limit: int,
        path_prefix: str | None = None,
    ) -> list[tuple[int, float]]:
        query = """
            SELECT c.id, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ? AND c.collection = ?
        """
        params: list[Any] = [match_query, collection]
        if path_prefix:
            query += " AND c.path LIKE ? ESCAPE '\\'"
            params.append(_like_prefix(path_prefix))
        query += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self.lock:
            rows = self.conn.execute(query, params).fetchall()
            return [(int(row["id"]), float(row["rank"])) for row in rows]

    def like_search(
        self, collection: str, needle: str, limit: int, path_prefix: str | None = None
    ) -> list[tuple[int, float]]:
        query = "SELECT id, 0.0 AS rank FROM chunks WHERE collection=? AND text LIKE ? ESCAPE '\\'"
        params: list[Any] = [collection, f"%{_escape_like(needle)}%"]
        if path_prefix:
            query += " AND path LIKE ? ESCAPE '\\'"
            params.append(_like_prefix(path_prefix))
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self.lock:
            rows = self.conn.execute(query, params).fetchall()
            return [(int(row["id"]), float(row["rank"])) for row in rows]

    def fetch_chunks(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        if not ids:
            return result
        with self.lock:
            for i in range(0, len(ids), 500):
                batch = list(ids[i:i + 500])
                placeholders = ",".join("?" for _ in batch)
                rows = self.conn.execute(
                    f"SELECT id, collection, path, ordinal, start, end, content_hash, text "
                    f"FROM chunks WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    result[int(row["id"])] = dict(row)
        return result

    # ------------------------------------------------------------------ watch
    def upsert_watch(self, collection: str, path: str, is_dir: bool) -> None:
        with self._tx():
            self.conn.execute(
                "INSERT OR IGNORE INTO collections(name,created_at,updated_at) VALUES(?,?,?)",
                (collection, now(), now()),
            )
            self.conn.execute(
                """
                INSERT INTO watch_roots(collection,path,is_dir,created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(collection,path) DO UPDATE SET is_dir=excluded.is_dir
                """,
                (collection, path, 1 if is_dir else 0, now()),
            )

    def remove_watch(self, collection: str, path: str) -> bool:
        with self._tx():
            cursor = self.conn.execute(
                "DELETE FROM watch_roots WHERE collection=? AND path=?", (collection, path)
            )
            return cursor.rowcount > 0

    def list_watch_roots(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM watch_roots ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]

    def watch_failed(self, collection: str, path: str, error: str | None) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE watch_roots SET last_scan_at=?, last_error=? WHERE collection=? AND path=?",
                (now(), (error or "")[:1000] if error else None, collection, path),
            )

    # ------------------------------------------------------------------- gc
    def gc_embeddings(self) -> int:
        with self._tx():
            cursor = self.conn.execute(
                "DELETE FROM embeddings WHERE NOT EXISTS "
                "(SELECT 1 FROM chunks c WHERE c.content_hash = embeddings.content_hash)"
            )
            deleted = cursor.rowcount
            if deleted:
                self._bump_generation()
            return deleted

    def clear_embeddings(self, fingerprint: str | None = None) -> int:
        with self._tx():
            if fingerprint:
                cursor = self.conn.execute(
                    "DELETE FROM embeddings WHERE fingerprint=?", (fingerprint,)
                )
            else:
                cursor = self.conn.execute("DELETE FROM embeddings")
            deleted = cursor.rowcount
            self._bump_generation()
            return deleted

    def clear_embeddings_for_collection(self, fingerprint: str, collection: str) -> int:
        with self._tx():
            cursor = self.conn.execute(
                "DELETE FROM embeddings WHERE fingerprint=? AND content_hash IN "
                "(SELECT content_hash FROM chunks WHERE collection=?)",
                (fingerprint, collection),
            )
            deleted = cursor.rowcount
            if deleted:
                self._bump_generation()
            return deleted

    # ----------------------------------------------------------------- stats
    def stats(self, fingerprint: str | None = None) -> dict[str, Any]:
        with self.lock:
            files = self.conn.execute(
                "SELECT COUNT(*) FROM files WHERE status='indexed'"
            ).fetchone()[0]
            error_files = self.conn.execute(
                "SELECT COUNT(*) FROM files WHERE status='error'"
            ).fetchone()[0]
            chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            distinct = self.conn.execute(
                "SELECT COUNT(DISTINCT content_hash) FROM chunks"
            ).fetchone()[0]
            total_embedded = int(self._meta_get("embedded_total", "0") or 0)
            total_chunks_written = int(self._meta_get("chunks_written_total", "0") or 0)
            watch_roots = self.conn.execute("SELECT COUNT(*) FROM watch_roots").fetchone()[0]
            last_scan = self.conn.execute(
                "SELECT MAX(last_scan_at) FROM watch_roots"
            ).fetchone()[0]
            last_index = self.conn.execute(
                "SELECT MAX(indexed_at) FROM files"
            ).fetchone()[0]

        active_vectors = self.embedding_count(fingerprint) if fingerprint else 0
        try:
            db_size = self.path.stat().st_size
            for suffix in ("-wal", "-shm"):
                side = Path(str(self.path) + suffix)
                if side.exists():
                    db_size += side.stat().st_size
        except OSError:
            db_size = 0

        reuse_saved = max(0, int(total_chunks_written) - int(total_embedded))
        return {
            "files": int(files),
            "error_files": int(error_files),
            "chunks": int(chunks),
            "distinct_chunks": int(distinct),
            "active_vectors": int(active_vectors),
            "embedding_cache_total": self.embedding_count(),
            "lifetime_embeddings": total_embedded,
            "lifetime_chunks": total_chunks_written,
            "reuse_saved": reuse_saved,
            "watch_roots": int(watch_roots),
            "last_scan_at": last_scan,
            "last_indexed_at": last_index,
            "db_bytes": db_size,
            "db_human": human_bytes(db_size),
            "generation": self.generation(),
            "collections": self.collections(),
        }


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like_prefix(prefix: str) -> str:
    return _escape_like(prefix.rstrip("/")) + "/%"
