"""Local HTTP API + web console.

The API is intentionally small and boring. Any app that can make an HTTP request
can use the daemon; no SDK, no Python import, no shared memory.

    POST /v1/index    {path, collection}      build/refresh an index
    POST /v1/search   {query, collection}     hybrid retrieval
    GET  /v1/stats                            counters, model, watcher health
    GET  /v1/files                            indexed files
    POST /v1/reindex                          rebuild a collection
    POST /v1/gc                               drop orphaned vectors

Long operations can run asynchronously and are observable through /v1/jobs.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .config import Config
from .db import Database
from .embedders import BaseEmbedder
from .indexer import Indexer
from .search import Searcher
from .util import canonical_path, dumps, now, safe_collection_name, shorten
from .watcher import Watcher


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    created_at: float = field(default_factory=now)
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "progress": dict(self.progress),
                "result": self.result,
                "error": self.error,
            }


class JobManager:
    def __init__(self, max_jobs: int = 80):
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, work: Callable[[Callable[[dict[str, Any]], None]], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()

        def progress(update: dict[str, Any]) -> None:
            with job.lock:
                job.progress = update

        def runner() -> None:
            with job.lock:
                job.status = "running"
                job.started_at = now()
            try:
                result = work(progress)
                with job.lock:
                    job.status = "done"
                    job.result = result
                    job.finished_at = now()
                    done_total = job.progress.get("total") or 1
                    job.progress = {
                        **job.progress,
                        "phase": "done",
                        "current": done_total,
                        "total": done_total,
                    }
            except Exception as exc:
                with job.lock:
                    job.status = "error"
                    job.error = shorten(f"{type(exc).__name__}: {exc}", 1000)
                    job.finished_at = now()
                traceback.print_exc()

        thread = threading.Thread(target=runner, name=f"embd-job-{job.id}", daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.to_dict() for job in jobs[:limit]]

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        finished = [j for j in self._jobs.values() if j.status in {"done", "error"}]
        finished.sort(key=lambda j: j.created_at)
        for job in finished[: max(0, len(self._jobs) - self.max_jobs)]:
            self._jobs.pop(job.id, None)


class App:
    def __init__(
        self,
        config: Config,
        db: Database,
        embedder: BaseEmbedder,
        *,
        watch: bool = True,
    ):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.started_at = now()
        self.indexer = Indexer(config, db, embedder)
        self.searcher = Searcher(config, db, embedder)
        self.jobs = JobManager()
        self.watcher: Watcher | None = None
        self.model_changed = False

        previous = self.db.meta_get("active_fingerprint", "")
        if not previous:
            self.db.meta_set("active_fingerprint", embedder.fingerprint)
        elif previous != embedder.fingerprint:
            self.model_changed = True
            self.db.meta_set("active_fingerprint", embedder.fingerprint)

        if watch and config.get("watch.enabled", True):
            self.watcher = Watcher(config, db, self.indexer)
            self.watcher.start()

    def stats(self) -> dict[str, Any]:
        stats = self.db.stats(self.embedder.fingerprint)
        stats["model"] = self.embedder.info()
        stats["uptime_seconds"] = round(now() - self.started_at, 1)
        stats["version"] = __version__
        stats["watch_enabled"] = self.watcher is not None
        stats["watcher"] = self.watcher.status if self.watcher else {"running": False}
        stats["model_changed"] = self.model_changed
        stats["needs_reindex"] = bool(
            stats["chunks"] > 0 and stats["active_vectors"] == 0
        )
        stats["jobs_running"] = sum(
            1 for job in self.jobs.list(100) if job["status"] in {"queued", "running"}
        )
        return stats

    def close(self) -> None:
        if self.watcher:
            self.watcher.stop()
            self.watcher.join(timeout=10)
        self.db.close()


_ROUTES = [
    ("GET", "/health", "daemon liveness"),
    ("GET", "/v1/stats", "index, cache and watcher counters"),
    ("GET", "/v1/config", "public configuration"),
    ("GET", "/v1/collections", "list collections"),
    ("POST", "/v1/collections", "create an empty collection"),
    ("GET", "/v1/files", "list indexed files"),
    ("POST", "/v1/index", "index a file or directory"),
    ("POST", "/v1/search", "hybrid search"),
    ("GET", "/v1/search", "query-string convenience search"),
    ("DELETE", "/v1/files", "remove a file/directory from the index"),
    ("POST", "/v1/reindex", "rebuild a collection"),
    ("GET", "/v1/watch", "list watched roots"),
    ("POST", "/v1/watch", "watch a path"),
    ("DELETE", "/v1/watch", "stop watching a path"),
    ("POST", "/v1/gc", "delete orphaned embeddings"),
    ("GET", "/v1/jobs", "recent jobs"),
    ("GET", "/v1/jobs/{id}", "job status"),
]


class Handler(BaseHTTPRequestHandler):
    server_version = "embd/" + __version__
    protocol_version = "HTTP/1.1"
    app: App  # injected by make_handler

    # ---------------------------------------------------------------- plumbing
    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("EMBD_HTTP_LOG"):
            super().log_message(fmt, *args)

    def do_OPTIONS(self) -> None:
        self._send_bytes(HTTPStatus.NO_CONTENT, b"", "text/plain")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            self._send_bytes(HTTPStatus.OK, b"", "text/html; charset=utf-8")
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"code": "not_found", "message": "not found"}})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send_dashboard()
                return
            if path == "/favicon.ico":
                self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                return
            self._guard(path)
            if path == "/health":
                self._json_ok({"version": __version__, "uptime_seconds": round(now() - self.app.started_at, 1), "model": self.app.embedder.info()})
            elif path == "/v1/stats":
                self._json_ok({"stats": self.app.stats()})
            elif path == "/v1/config":
                self._json_ok({"config": self.app.config.to_public_dict()})
            elif path == "/v1/routes":
                self._json_ok({"routes": [{"method": m, "path": p, "description": d} for m, p, d in _ROUTES]})
            elif path == "/v1/collections":
                self._json_ok({"collections": self.app.db.collections()})
            elif path == "/v1/files":
                collection = self._collection(query.get("collection", ["default"])[0])
                limit = _int(query.get("limit", ["100"])[0], 100, 1, 1000)
                offset = _int(query.get("offset", ["0"])[0], 0, 0, 1_000_000)
                self._json_ok({"files": self.app.db.file_rows(collection, limit, offset), "collection": collection})
            elif path == "/v1/search":
                q = query.get("q", query.get("query", [""]))[0]
                collection = self._collection(query.get("collection", ["default"])[0])
                limit = _int(query.get("limit", [str(self.app.config.search["default_limit"])])[0], 10, 1, 100)
                mode = query.get("mode", [None])[0]
                prefix = query.get("path_prefix", [None])[0]
                result = self.app.searcher.search(q, collection=collection, limit=limit, mode=mode, path_prefix=prefix)
                self._json_ok(result)
            elif path == "/v1/watch":
                self._json_ok({"watch": self.app.db.list_watch_roots()})
            elif path == "/v1/jobs":
                self._json_ok({"jobs": self.app.jobs.list(30)})
            elif re.match(r"^/v1/jobs/[A-Za-z0-9]+$", path):
                job_id = path.rsplit("/", 1)[1]
                job = self.app.jobs.get(job_id)
                if not job:
                    raise ApiError(404, "job_not_found", f"unknown job: {job_id}")
                self._json_ok({"job": job.to_dict()})
            elif path == "/v1/openapi.json":
                self._json_ok({"openapi": "3.1.0", "info": {"title": "embd", "version": __version__}, "routes": [{"method": m, "path": p, "description": d} for m, p, d in _ROUTES]})
            else:
                raise ApiError(404, "not_found", f"no route: GET {path}")
        except ApiError as exc:
            self._send_error(exc)
        except ValueError as exc:
            self._send_error(ApiError(400, "invalid_request", str(exc)))
        except Exception as exc:
            traceback.print_exc()
            self._send_error(ApiError(500, "internal_error", shorten(str(exc), 500)))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            self._guard(path)
            body = self._read_json()
            if path == "/v1/index":
                self._handle_index(body)
            elif path == "/v1/search":
                q = str(body.get("query") or body.get("q") or "").strip()
                if not q:
                    raise ApiError(400, "invalid_query", "query is required")
                collection = self._collection(body.get("collection") or "default")
                limit = _int(body.get("limit") or self.app.config.search["default_limit"], 10, 1, 100)
                mode = body.get("mode")
                prefix = body.get("path_prefix")
                if mode is not None and not isinstance(mode, str):
                    raise ApiError(400, "invalid_mode", "mode must be a string")
                if prefix is not None and not isinstance(prefix, str):
                    raise ApiError(400, "invalid_path_prefix", "path_prefix must be a string")
                result = self.app.searcher.search(q, collection=collection, limit=limit, mode=mode, path_prefix=prefix)
                self._json_ok(result)
            elif path == "/v1/reindex":
                self._handle_reindex(body)
            elif path == "/v1/collections":
                name = self._collection(body.get("name") or body.get("collection") or "")
                created = self.app.db.ensure_collection(name)
                self._json_ok({"collection": name, "created": created})
            elif path == "/v1/watch":
                self._handle_add_watch(body)
            elif path == "/v1/gc":
                deleted = self.app.db.gc_embeddings()
                self.app.searcher.cache.clear()
                self._json_ok({"deleted": deleted})
            else:
                raise ApiError(404, "not_found", f"no route: POST {path}")
        except ApiError as exc:
            self._send_error(exc)
        except json.JSONDecodeError as exc:
            self._send_error(ApiError(400, "invalid_json", str(exc)))
        except ValueError as exc:
            self._send_error(ApiError(400, "invalid_request", str(exc)))
        except Exception as exc:
            traceback.print_exc()
            self._send_error(ApiError(500, "internal_error", shorten(str(exc), 500)))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            self._guard(path)
            if path == "/v1/files":
                body = self._read_json(optional=True)
                target = body.get("path") or query.get("path", [None])[0]
                if not target:
                    raise ApiError(400, "invalid_path", "path is required")
                collection = self._collection(body.get("collection") or query.get("collection", ["default"])[0])
                recursive = body.get("recursive")
                result = self.app.indexer.remove(
                    str(target), collection,
                    is_dir=bool(recursive) if recursive is not None else None,
                )
                self.app.searcher.cache.clear()
                if self.app.watcher:
                    self.app.watcher.wake()
                self._json_ok(result)
            elif path == "/v1/watch":
                body = self._read_json(optional=True)
                target = body.get("path") or query.get("path", [None])[0]
                if not target:
                    raise ApiError(400, "invalid_path", "path is required")
                collection = self._collection(body.get("collection") or query.get("collection", ["default"])[0])
                removed = self.app.db.remove_watch(collection, canonical_path(str(target)))
                self._json_ok({"removed": removed})
            elif path.startswith("/v1/collections/"):
                name = self._collection(unquote(path.rsplit("/", 1)[1]))
                deleted = self.app.db.delete_collection(name)
                self.app.searcher.cache.clear()
                self._json_ok({"collection": name, "deleted": deleted})
            else:
                raise ApiError(404, "not_found", f"no route: DELETE {path}")
        except ApiError as exc:
            self._send_error(exc)
        except Exception as exc:
            traceback.print_exc()
            self._send_error(ApiError(500, "internal_error", shorten(str(exc), 500)))

    # ------------------------------------------------------------------ routes
    def _handle_index(self, body: dict[str, Any]) -> None:
        raw = body.get("paths") if "paths" in body else body.get("path")
        if isinstance(raw, str):
            paths = [raw]
        elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            paths = raw
        else:
            raise ApiError(400, "invalid_path", "path or paths is required")
        paths = [canonical_path(p) for p in paths if p.strip()]
        if not paths:
            raise ApiError(400, "invalid_path", "path or paths is required")
        if len(paths) > 1000:
            raise ApiError(400, "too_many_paths", "at most 1000 paths per request")
        collection = self._collection(body.get("collection") or "default")
        force = bool(body.get("force", False))
        watch = bool(body.get("watch", True))
        asynchronous = bool(body.get("async", False))

        def work(progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            summary = self.app.indexer.index_paths(
                paths, collection, force=force, watch=watch, progress=progress
            )
            if self.app.watcher:
                self.app.watcher.wake()
            self.app.searcher.cache.clear()
            return summary.to_dict()

        if asynchronous:
            job = self.app.jobs.submit("index", work)
            self._json_ok({"job_id": job.id, "status": job.status}, status=202)
            return
        summary = work(lambda _: None)
        self._json_ok({"summary": summary, "collection": collection})

    def _handle_reindex(self, body: dict[str, Any]) -> None:
        collection = self._collection(body.get("collection") or "default")
        force_embed = bool(body.get("force_embed", False))
        asynchronous = bool(body.get("async", False))

        def work(progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            if force_embed:
                self.app.db.clear_embeddings_for_collection(self.app.embedder.fingerprint, collection)
                self.app.searcher.cache.clear()
            summary = self.app.indexer.reindex_collection(collection, force=True)
            if self.app.watcher:
                self.app.watcher.wake()
            self.app.searcher.cache.clear()
            self.app.model_changed = False
            result = summary.to_dict()
            result["force_embed"] = force_embed
            return result

        if asynchronous:
            job = self.app.jobs.submit("reindex", work)
            self._json_ok({"job_id": job.id, "status": job.status}, status=202)
            return
        self._json_ok({"summary": work(lambda _: None), "collection": collection})

    def _handle_add_watch(self, body: dict[str, Any]) -> None:
        target = body.get("path")
        if not isinstance(target, str) or not target.strip():
            raise ApiError(400, "invalid_path", "path is required")
        canonical = canonical_path(target)
        if not os.path.exists(canonical):
            raise ApiError(400, "path_not_found", f"path does not exist: {canonical}")
        collection = self._collection(body.get("collection") or "default")
        self.app.db.upsert_watch(collection, canonical, os.path.isdir(canonical))
        if self.app.watcher:
            self.app.watcher.wake()
        self._json_ok({"path": canonical, "collection": collection, "watching": True})

    # ----------------------------------------------------------------- helpers
    def _collection(self, name: Any) -> str:
        try:
            return safe_collection_name(str(name or "default"))
        except ValueError as exc:
            raise ApiError(400, "invalid_collection", str(exc)) from exc

    def _guard(self, path: str) -> None:
        if path in {"/", "/health", "/favicon.ico"}:
            return
        api_key = self.app.config.get("api_key")
        if not api_key:
            return
        provided = self.headers.get("X-API-Key", "")
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        if not provided or not hmac.compare_digest(str(api_key), provided):
            raise ApiError(401, "unauthorized", "missing or invalid API key")

    def _read_json(self, *, optional: bool = False) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        max_bytes = int(self.app.config.get("server.max_body_bytes", 1024 * 1024))
        if length > max_bytes:
            raise ApiError(413, "body_too_large", f"request body exceeds {max_bytes} bytes")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            if optional:
                return {}
            raise ApiError(400, "empty_body", "JSON body is required")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "invalid_json", f"invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ApiError(400, "invalid_json", "JSON body must be an object")
        return value

    def _dashboard_bytes(self) -> bytes:
        path = Path(__file__).parent / "static" / "index.html"
        try:
            return path.read_bytes()
        except OSError:
            return b"<!doctype html><title>embd</title><h1>embd</h1><p>Dashboard asset missing.</p>"

    def _send_dashboard(self) -> None:
        self._send_bytes(HTTPStatus.OK, self._dashboard_bytes(), "text/html; charset=utf-8")

    def _json_ok(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_json(status, {"ok": True, **payload})

    def _send_error(self, exc: ApiError) -> None:
        self._send_json(exc.status, {"ok": False, "error": {"code": exc.code, "message": exc.message}})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(status, dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            origin = self.app.config.get("server.cors_origin", "*")
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


class Server:
    def __init__(self, app: App, host: str, port: int):
        self.app = app
        handler = type("BoundHandler", (Handler,), {"app": app})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.host = host
        self.port = int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def serve(app: App, host: str, port: int) -> None:
    Server(app, host, port).serve_forever()
