"""A small polling watcher.

Native file-system events differ across platforms and add dependencies. The
indexer's mtime fast path is cheap enough that a periodic stat-only scan is a
good default for a local daemon; content is only read when stat changes. A
future FSEvents/inotify backend can call the same ``scan_once`` without touching
the indexing logic.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .config import Config
from .db import Database
from .indexer import Indexer
from .util import now, shorten


class Watcher(threading.Thread):
    def __init__(
        self,
        config: Config,
        db: Database,
        indexer: Indexer,
        *,
        interval: float | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(name="embd-watcher", daemon=True)
        self.config = config
        self.db = db
        self.indexer = indexer
        self.interval = float(interval or config.get("watch.interval_seconds", 5.0))
        self.on_event = on_event
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._run_lock = threading.Lock()
        self.status: dict[str, Any] = {
            "running": False,
            "cycles": 0,
            "last_cycle_at": None,
            "last_error": None,
            "last_changes": 0,
            "interval_seconds": self.interval,
        }

    def run(self) -> None:
        self.status["running"] = True
        self._emit({"type": "watcher_started", "interval": self.interval})
        # Let the HTTP server bind before the first potentially expensive scan.
        self._stop_event.wait(0.5)
        while not self._stop_event.is_set():
            self.scan_once()
            self._wake_event.wait(self.interval)
            self._wake_event.clear()
        self.status["running"] = False
        self._emit({"type": "watcher_stopped"})

    def scan_once(self) -> dict[str, Any]:
        started = now()
        with self._run_lock:
            roots = self.db.list_watch_roots()
            changes = 0
            errors: list[dict[str, str]] = []
            for root in roots:
                if self._stop_event.is_set():
                    break
                summary = self.indexer.scan_root(root)
                changes += summary.files_indexed + summary.files_removed
                errors.extend(summary.errors)
            self.status.update({
                "cycles": int(self.status["cycles"]) + 1,
                "last_cycle_at": now(),
                "last_changes": changes,
                "last_error": shorten(errors[0]["error"], 240) if errors else None,
                "last_duration_ms": round((now() - started) * 1000, 1),
                "roots": len(roots),
            })
            if changes or errors:
                self._emit({
                    "type": "scan",
                    "changes": changes,
                    "roots": len(roots),
                    "errors": errors[:5],
                })
            return {"changes": changes, "roots": len(roots), "errors": errors}

    def wake(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass
