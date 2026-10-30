"""Command line interface.

Most commands work in two modes:

* local (default) — open the SQLite index directly, no daemon required
* remote (``--remote URL``) — call a running daemon over HTTP

That makes the CLI useful in scripts and CI before anyone starts a background
service, while still letting a GUI app share the exact same index.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import sqlite3
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .bench import format_benchmark, run_benchmark
from .config import Config, load_config
from .db import Database
from .embedders import BaseEmbedder, make_embedder
from .indexer import Indexer, IndexSummary
from .search import Searcher
from .server import App, Server
from .util import dumps, human_bytes, human_duration, relative_time, safe_collection_name, shorten
from .watcher import Watcher


class CliError(RuntimeError):
    pass


# --------------------------------------------------------------------- output
class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def _style(*, no_color: bool = False) -> Style:
    return Style(
        enabled=sys.stdout.isatty()
        and not no_color
        and not os.environ.get("NO_COLOR")
    )


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _width(text))
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], style: Style) -> str:
    widths = [_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _width(str(cell)))
    lines = [
        "  ".join(style.dim(_pad(h, widths[i])) for i, h in enumerate(headers)),
        style.dim("  ".join("─" * widths[i] for i in range(len(headers)))),
    ]
    for row in rows:
        lines.append("  ".join(_pad(str(cell), widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _truncate_path(path: str, limit: int = 60) -> str:
    if len(path) <= limit:
        return path
    head = max(12, limit // 2 - 2)
    return path[:head] + "…" + path[-(limit - head - 1):]


def _summary_line(summary: dict[str, Any], style: Style) -> str:
    bits = [
        f"{style.green(str(summary['files_indexed']))} 个文件更新",
        f"{summary['files_unchanged'] + summary['files_unchanged_hash']} 个未变",
        f"{summary['chunks_added']} 块写入",
        f"{style.cyan(str(summary['chunks_reused']))} 块复用",
        f"{style.cyan(str(summary['embeddings_computed']))} 次 embedding",
        f"{summary['duration_ms']:.0f} ms",
    ]
    if summary.get("files_removed"):
        bits.append(f"{summary['files_removed']} 个移除")
    if summary.get("errors"):
        bits.append(style.red(f"{len(summary['errors'])} 个错误"))
    if summary.get("skipped"):
        bits.append(style.yellow(f"{len(summary['skipped'])} 个跳过"))
    return " · ".join(bits)


# --------------------------------------------------------------------- remote
def _request(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json", "User-Agent": f"embd/{__version__}"}
    if payload is not None:
        body = dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            message = parsed.get("error", {}).get("message", detail)
        except json.JSONDecodeError:
            message = detail
        raise CliError(f"HTTP {exc.code}: {shorten(message, 500)}") from exc
    except urllib.error.URLError as exc:
        raise CliError(f"无法连接 {url}: {exc.reason}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"服务返回了无效 JSON: {exc}") from exc
    if not data.get("ok"):
        message = data.get("error", {}).get("message", "request failed")
        raise CliError(str(message))
    return data


def _remote_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


# ---------------------------------------------------------------------- local
class Local:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)
        self.embedder = make_embedder(config.embedder)
        self.indexer = Indexer(config, self.db, self.embedder)
        self.searcher = Searcher(config, self.db, self.embedder)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Local":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -------------------------------------------------------------------- commands
def cmd_serve(args: argparse.Namespace, config: Config, style: Style) -> int:
    host = args.host or config.get("host")
    port = int(args.port if args.port is not None else config.get("port"))
    app = App(config, Database(config.db_path), make_embedder(config.embedder), watch=not args.no_watch)
    try:
        server = Server(app, host, port)
    except OSError as exc:
        app.close()
        raise CliError(f"无法监听 {host}:{port}：{exc}") from exc

    if not args.json:
        print(style.bold(f"embd {__version__}"))
        print(f"  model   {style.cyan(app.embedder.name)}  {style.dim(app.embedder.fingerprint)}")
        print(f"  home    {config.home}")
        print(f"  db      {config.db_path}")
        print(f"  api     {style.cyan(server.url + '/v1')}")
        print(f"  console {style.cyan(server.url)}")
        if app.watcher:
            print(f"  watch   every {config.get('watch.interval_seconds')}s · {len(app.db.list_watch_roots())} root(s)")
        if app.model_changed:
            print(style.yellow("  ! embedding model changed; run `embd reindex` to refresh vectors"))
        if app.watcher is None:
            print(style.dim("  watch disabled (--no-watch)"))
        print(style.dim("  Ctrl-C to stop"))

    stopped = threading.Event()

    def shutdown(signum: int | None = None, frame: Any = None) -> None:
        if stopped.is_set():
            return
        stopped.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        try:
            server.httpd.server_close()
        except OSError:
            pass
    return 0


def cmd_index(args: argparse.Namespace, config: Config, style: Style) -> int:
    collection = safe_collection_name(args.collection)
    if args.async_ and not args.remote:
        raise CliError("--async 需要配合 --remote 指向运行中的 daemon")
    if args.remote:
        data = _request(
            _remote_url(args.remote, "/v1/index"),
            "POST",
            {
                "paths": args.paths,
                "collection": collection,
                "force": args.force,
                "watch": not args.no_watch,
                "async": args.async_,
            },
            api_key=args.api_key,
        )
        if args.async_:
            job = data
            if args.json:
                print(dumps(job, pretty=True))
            else:
                print(f"job {style.cyan(str(job.get('job_id')))} queued; poll with: embd jobs {job.get('job_id')} --remote {args.remote}")
            return 0
        summary = data["summary"]
    else:
        with Local(config) as local:
            if not args.json and sys.stdout.isatty():
                def progress(update: dict[str, Any]) -> None:
                    current = update.get("current", 0)
                    total = update.get("total", 0)
                    path = _truncate_path(str(update.get("path", "")), 48)
                    sys.stdout.write(f"\r{style.dim(f'索引 {current}/{total}')}  {path:<48}")
                    sys.stdout.flush()

                summary_obj = local.indexer.index_paths(
                    args.paths, collection, force=args.force,
                    watch=not args.no_watch, progress=progress,
                )
                sys.stdout.write("\r" + " " * 100 + "\r")
            else:
                summary_obj = local.indexer.index_paths(
                    args.paths, collection, force=args.force, watch=not args.no_watch
                )
            summary = summary_obj.to_dict()

    if args.json:
        print(dumps(summary, pretty=True))
    else:
        print(_summary_line(summary, style))
        for error in summary.get("errors", [])[:8]:
            print(style.red(f"  ✗ {shorten(error['path'], 72)}: {shorten(error['error'], 140)}"))
        for skipped in summary.get("skipped", [])[:8]:
            print(style.yellow(f"  - {shorten(skipped['path'], 72)}: {shorten(skipped['reason'], 140)}"))
    return 0 if not summary.get("errors") else 1


def cmd_search(args: argparse.Namespace, config: Config, style: Style) -> int:
    collection = safe_collection_name(args.collection)
    query = " ".join(args.query).strip()
    if not query:
        raise CliError("query cannot be empty")
    if args.remote:
        data = _request(
            _remote_url(args.remote, "/v1/search"),
            "POST",
            {
                "query": query,
                "collection": collection,
                "limit": args.limit,
                "mode": args.mode,
                "path_prefix": args.path_prefix,
            },
            api_key=args.api_key,
        )
    else:
        with Local(config) as local:
            data = local.searcher.search(
                query, collection=collection, limit=args.limit,
                mode=args.mode, path_prefix=args.path_prefix,
            )
    if args.json:
        print(dumps(data, pretty=True))
        return 0
    results = data.get("results", [])
    if not results:
        print(style.dim("没有命中。先建立索引，或换一个关键词。"))
        for warning in data.get("warnings", []):
            print(style.yellow(f"  ! {warning}"))
        return 0
    print(
        style.dim(
            f"{data['count']} 条结果 · {data['took_ms']:.1f} ms · {data['mode']} · {data['model']}"
        )
    )
    for index, item in enumerate(results, 1):
        score = f"{item['score']:.4f}"
        print(f"\n{style.dim(str(index).rjust(2))}. {style.cyan(score)}  {item['path']}")
        print(f"    {item['snippet']}")
        extras = []
        if item.get("vector_score"):
            extras.append(f"vec {item['vector_score']:.3f}")
        if item.get("lexical_rank"):
            extras.append(f"kw #{item['lexical_rank']}")
        extras.append(f"chunk {item['ordinal']}")
        print(style.dim("    " + " · ".join(extras)))
    for warning in data.get("warnings", []):
        print(style.yellow(f"\n! {warning}"))
    return 0


def cmd_stats(args: argparse.Namespace, config: Config, style: Style) -> int:
    if args.remote:
        data = _request(_remote_url(args.remote, "/v1/stats"), api_key=args.api_key)["stats"]
    else:
        with Local(config) as local:
            data = local.db.stats(local.embedder.fingerprint)
            data["model"] = local.embedder.info()
    if args.json:
        print(dumps(data, pretty=True))
        return 0
    model = data.get("model", {})
    reuse_base = data.get("lifetime_chunks", 0)
    reuse = data.get("reuse_saved", 0)
    reuse_pct = (reuse / reuse_base * 100) if reuse_base else 0.0
    print(style.bold("embd stats"))
    print(f"  model          {model.get('name') or model.get('model', '?')}  {style.dim(model.get('fingerprint', ''))}")
    errors_label = style.dim("(errors %s)" % data.get("error_files", 0))
    distinct_label = style.dim("(distinct %s)" % data.get("distinct_chunks", 0))
    print(f"  files          {data.get('files', 0)}  {errors_label}")
    print(f"  chunks         {data.get('chunks', 0)}  {distinct_label}")
    print(f"  vectors        {data.get('active_vectors', 0)}")
    print(f"  reuse          {reuse} / {reuse_base} chunks  {style.green(f'{reuse_pct:.1f}%')} saved")
    print(f"  watch roots    {data.get('watch_roots', 0)}")
    print(f"  last index     {relative_time(data.get('last_indexed_at'))}")
    print(f"  database       {data.get('db_human', human_bytes(data.get('db_bytes', 0)))}  {style.dim(str(config.db_path))}")
    if data.get("warnings") or data.get("needs_reindex"):
        print(style.yellow("  ! model has no vectors for this index; run `embd reindex`"))
    collections = data.get("collections") or []
    if collections:
        print()
        print(_table(
            ["collection", "files", "chunks", "last indexed"],
            [
                [c["name"], str(c["files"]), str(c["chunks"]), relative_time(c.get("last_indexed"))]
                for c in collections
            ],
            style,
        ))
    return 0


def cmd_collections(args: argparse.Namespace, config: Config, style: Style) -> int:
    if args.remote:
        rows = _request(_remote_url(args.remote, "/v1/collections"), api_key=args.api_key)["collections"]
    else:
        with Local(config) as local:
            rows = local.db.collections()
    if args.json:
        print(dumps(rows, pretty=True))
        return 0
    if not rows:
        print(style.dim("还没有 collection。运行 `embd index <path>`。"))
        return 0
    print(_table(
        ["collection", "files", "chunks", "errors", "last indexed"],
        [[r["name"], str(r["files"]), str(r["chunks"]), str(r["errors"]), relative_time(r.get("last_indexed"))] for r in rows],
        style,
    ))
    return 0


def cmd_files(args: argparse.Namespace, config: Config, style: Style) -> int:
    collection = safe_collection_name(args.collection)
    if args.remote:
        query = urllib.parse.urlencode({"collection": collection, "limit": args.limit})
        rows = _request(_remote_url(args.remote, f"/v1/files?{query}"), api_key=args.api_key)["files"]
    else:
        with Local(config) as local:
            rows = local.db.file_rows(collection, args.limit, 0)
    if args.json:
        print(dumps(rows, pretty=True))
        return 0
    if not rows:
        print(style.dim(f"collection `{collection}` 还没有文件。"))
        return 0
    print(_table(
        ["path", "status", "chunks", "size", "indexed"],
        [
            [_truncate_path(r["path"], 64), r["status"], str(r["chunks"]),
             human_bytes(r["size"]), relative_time(r.get("indexed_at"))]
            for r in rows
        ],
        style,
    ))
    return 0


def cmd_remove(args: argparse.Namespace, config: Config, style: Style) -> int:
    collection = safe_collection_name(args.collection)
    if args.remote:
        data = _request(
            _remote_url(args.remote, "/v1/files"), "DELETE",
            {"path": args.path, "collection": collection, "recursive": True if args.recursive else None},
            api_key=args.api_key,
        )
    else:
        with Local(config) as local:
            data = local.indexer.remove(args.path, collection, is_dir=True if args.recursive else None)
    if args.json:
        print(dumps(data, pretty=True))
    else:
        print(f"已从索引移除 {data['removed']} 个文件  {style.dim(data['path'])}")
    return 0


def cmd_reindex(args: argparse.Namespace, config: Config, style: Style) -> int:
    collection = safe_collection_name(args.collection)
    if args.async_ and not args.remote:
        raise CliError("--async 需要配合 --remote 指向运行中的 daemon")
    if args.remote:
        data = _request(
            _remote_url(args.remote, "/v1/reindex"), "POST",
            {"collection": collection, "force_embed": args.force_embed, "async": args.async_},
            api_key=args.api_key,
            timeout=1800.0,
        )
        if args.async_:
            print(dumps(data, pretty=True) if args.json else f"job {data.get('job_id')} queued")
            return 0
        summary = data["summary"]
    else:
        with Local(config) as local:
            if args.force_embed:
                local.db.clear_embeddings_for_collection(local.embedder.fingerprint, collection)
            summary = local.indexer.reindex_collection(collection, force=True).to_dict()
    if args.json:
        print(dumps(summary, pretty=True))
    else:
        print(_summary_line(summary, style))
    return 0 if not summary.get("errors") else 1


def cmd_watch(args: argparse.Namespace, config: Config, style: Style) -> int:
    with Local(config) as local:
        roots = local.db.list_watch_roots()
        if not roots:
            print(style.yellow("没有 watch root。先运行 `embd index <path>` 或 `embd watch` 前用 API 添加。"))
            return 1
        watcher = Watcher(
            config, local.db, local.indexer,
            interval=args.interval or config.get("watch.interval_seconds"),
            on_event=lambda event: print(style.dim(f"[watch] {dumps(event)}")),
        )
        watcher.start()
        print(f"watching {len(roots)} root(s), interval {watcher.interval}s. Ctrl-C to stop.")
        try:
            while watcher.is_alive():
                time.sleep(0.4)
        except KeyboardInterrupt:
            pass
        finally:
            watcher.stop()
            watcher.join(timeout=5)
    return 0


def cmd_gc(args: argparse.Namespace, config: Config, style: Style) -> int:
    if args.remote:
        data = _request(_remote_url(args.remote, "/v1/gc"), "POST", {}, api_key=args.api_key)
    else:
        with Local(config) as local:
            data = {"deleted": local.db.gc_embeddings()}
    if args.json:
        print(dumps(data, pretty=True))
    else:
        print(f"清理 {data['deleted']} 条孤立向量")
    return 0


def cmd_jobs(args: argparse.Namespace, config: Config, style: Style) -> int:
    if not args.remote:
        raise CliError("`embd jobs` 需要 --remote（任务只存在于守护进程中）")
    if args.job_id:
        data = _request(_remote_url(args.remote, f"/v1/jobs/{args.job_id}"), api_key=args.api_key)
        job = data["job"]
        if args.json:
            print(dumps(job, pretty=True))
        else:
            print(f"{job['id']}  {job['kind']}  {job['status']}")
            progress = job.get("progress") or {}
            if progress:
                print(style.dim(dumps(progress)))
            if job.get("error"):
                print(style.red(job["error"]))
            if job.get("result"):
                print(_summary_line(job["result"], style))
        return 0
    jobs = _request(_remote_url(args.remote, "/v1/jobs"), api_key=args.api_key)["jobs"]
    if args.json:
        print(dumps(jobs, pretty=True))
        return 0
    if not jobs:
        print(style.dim("没有任务。"))
        return 0
    print(_table(
        ["id", "kind", "status", "progress", "created"],
        [[j["id"], j["kind"], j["status"],
          f"{j.get('progress', {}).get('current', 0)}/{j.get('progress', {}).get('total', 0)}",
          relative_time(j.get("created_at"))] for j in jobs],
        style,
    ))
    return 0


def cmd_doctor(args: argparse.Namespace, config: Config, style: Style) -> int:
    checks: list[tuple[bool, str, str]] = []
    checks.append((sys.version_info >= (3, 10), "Python", f"{sys.version.split()[0]}"))
    checks.append((True, "SQLite", sqlite3.sqlite_version))
    fts = False
    trigram = False
    try:
        mem = sqlite3.connect(":memory:")
        mem.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(x)")
        mem.execute("CREATE VIRTUAL TABLE trgm_probe USING fts5(x, tokenize='trigram')")
        fts = trigram = True
        mem.close()
    except sqlite3.Error as exc:
        checks.append((False, "FTS5", str(exc)))
    checks.append((fts, "FTS5", "available" if fts else "missing"))
    checks.append((trigram, "trigram tokenizer", "available" if trigram else "missing"))

    try:
        config.home.mkdir(parents=True, exist_ok=True)
        probe = config.home / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append((True, "home writable", str(config.home)))
    except OSError as exc:
        checks.append((False, "home writable", str(exc)))

    db: Database | None = None
    try:
        db = Database(config.db_path)
        embedder = make_embedder(config.embedder)
        stats = db.stats(embedder.fingerprint)
        checks.append((True, "database", f"{config.db_path} · {stats['db_human']}"))
        checks.append((True, "model", f"{embedder.name} · dim {embedder.dim}"))
        checks.append((stats["active_vectors"] > 0 or stats["chunks"] == 0,
                       "vectors", f"{stats['active_vectors']} active / {stats['chunks']} chunks"))
        checks.append((True, "indexed files", str(stats["files"])))
        checks.append((True, "watch roots", str(stats["watch_roots"])))
    except Exception as exc:
        checks.append((False, "database", shorten(str(exc), 200)))
    finally:
        if db:
            db.close()

    pdf = importlib.util.find_spec("pypdf") is not None or importlib.util.find_spec("PyPDF2") is not None
    checks.append((True, "PDF support", "pypdf installed" if pdf else "optional; install pypdf for PDFs"))

    api_url = args.remote or f"http://127.0.0.1:{config.get('port')}"
    try:
        health = _request(_remote_url(api_url, "/health"), timeout=1.5)
        checks.append((True, "daemon", f"online · {health.get('version')} · {health.get('model', {}).get('name')}"))
    except CliError:
        checks.append((True, "daemon", "offline (CLI still works locally)"))

    if args.json:
        print(dumps([{"ok": ok, "check": name, "detail": detail} for ok, name, detail in checks], pretty=True))
    else:
        print(style.bold("embd doctor"))
        for ok, name, detail in checks:
            mark = style.green("✓") if ok else style.red("✗")
            print(f"  {mark} {_pad(name, 18)} {style.dim(detail)}")
        if not pdf:
            print(style.dim("\n  PDF 可选依赖：pip install pypdf"))
        if any(not ok for ok, _, _ in checks):
            return 1
    return 0


def cmd_bench(args: argparse.Namespace, config: Config, style: Style) -> int:
    report = run_benchmark(docs=args.docs, base_config=config)
    if args.json:
        print(dumps(report, pretty=True))
    else:
        print(format_benchmark(report))
    return 0


def cmd_version(args: argparse.Namespace, config: Config, style: Style) -> int:
    print(f"embd {__version__}")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--home", default=argparse.SUPPRESS, help="data directory (default: ~/.embd or $EMBD_HOME)")
    common.add_argument("--config", default=argparse.SUPPRESS, help="path to config.json")
    common.add_argument("--remote", default=argparse.SUPPRESS, help="call a running daemon at this URL")
    common.add_argument("--api-key", default=argparse.SUPPRESS, help="API key for --remote (or $EMBD_API_KEY)")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        prog="embd",
        description="embd — local incremental embedding daemon for RAG",
        parents=[common],
        epilog=(
            "examples:\n"
            "  embd serve --open\n"
            "  embd index ~/Documents/notes --collection notes\n"
            "  embd search '分块对齐' --collection notes\n"
            "  embd bench\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"embd {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("serve", parents=[common], help="start the HTTP daemon + web console")
    p.add_argument("--host", help="bind address (default from config: 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 47821)")
    p.add_argument("--no-watch", action="store_true", help="disable the background watcher")
    p.add_argument("--open", action="store_true", help="open the console in a browser")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("index", parents=[common], help="index files or directories")
    p.add_argument("paths", nargs="+", help="file or directory paths")
    p.add_argument("-c", "--collection", default="default")
    p.add_argument("-f", "--force", action="store_true", help="ignore the stat fast path")
    p.add_argument("--no-watch", action="store_true", help="index once, do not add a watch root")
    p.add_argument("--async", dest="async_", action="store_true", help="queue in the daemon and return a job id")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", parents=[common], help="hybrid semantic + keyword search")
    p.add_argument("query", nargs="+")
    p.add_argument("-c", "--collection", default="default")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--mode", choices=["hybrid", "vector", "lexical"], default=None)
    p.add_argument("--path-prefix", help="restrict results to a path prefix")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", parents=[common], help="show index and cache statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("collections", parents=[common], help="list collections")
    p.set_defaults(func=cmd_collections)

    p = sub.add_parser("files", parents=[common], help="list indexed files")
    p.add_argument("-c", "--collection", default="default")
    p.add_argument("-n", "--limit", type=int, default=50)
    p.set_defaults(func=cmd_files)

    p = sub.add_parser("remove", parents=[common], help="remove a path from the index (not from disk)")
    p.add_argument("path")
    p.add_argument("-c", "--collection", default="default")
    p.add_argument("--recursive", action="store_true", help="treat path as a directory even if it no longer exists")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("reindex", parents=[common], help="rebuild an entire collection")
    p.add_argument("-c", "--collection", default="default")
    p.add_argument("--force-embed", action="store_true", help="drop this collection's vectors first")
    p.add_argument("--async", dest="async_", action="store_true")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("watch", parents=[common], help="run the watcher in the foreground")
    p.add_argument("--interval", type=float, help="poll interval in seconds")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("gc", parents=[common], help="delete cached vectors no longer referenced by a chunk")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("jobs", parents=[common], help="inspect daemon jobs (requires --remote)")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("doctor", parents=[common], help="check the environment and index health")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("bench", parents=[common], help="run the incremental-index benchmark")
    p.add_argument("-n", "--docs", type=int, default=60)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("version", parents=[common], help="print the version")
    p.set_defaults(func=cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    for flag in ("json", "no_color"):
        if not hasattr(args, flag):
            setattr(args, flag, False)
    for flag in ("home", "config", "remote", "api_key"):
        if not hasattr(args, flag):
            setattr(args, flag, None)

    json_mode = bool(getattr(args, "json", False))
    style = _style(no_color=bool(getattr(args, "no_color", False)))
    try:
        config = load_config(home=getattr(args, "home", None), config_path=getattr(args, "config", None))
        if not getattr(args, "api_key", None):
            args.api_key = os.environ.get("EMBD_API_KEY") or config.get("api_key")
        if not getattr(args, "remote", None):
            args.remote = os.environ.get("EMBD_URL")
        return int(args.func(args, config, style) or 0)
    except CliError as exc:
        if json_mode:
            print(dumps({"ok": False, "error": str(exc)}))
        else:
            print(style.red(f"error: {exc}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if os.environ.get("EMBD_DEBUG"):
            raise
        if json_mode:
            print(dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        else:
            print(style.red(f"error: {type(exc).__name__}: {exc}"), file=sys.stderr)
            print(style.dim("set EMBD_DEBUG=1 for a traceback"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
