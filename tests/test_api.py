from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from embd.config import load_config
from embd.db import Database
from embd.embedders import HashEmbedder
from embd.server import App, Server
from tests.helpers import paragraph, temp_config


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="embd-api-")
        self.config = temp_config(self.tmp.name, **{"embedder.dim": 128, "watch.enabled": False})
        self.root = Path(self.tmp.name) / "docs"
        self.root.mkdir()
        (self.root / "a.md").write_text("".join(paragraph(i) for i in range(12)), encoding="utf-8")
        self.app = App(self.config, Database(self.config.db_path), HashEmbedder(dim=128), watch=False)
        self.server = Server(self.app, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.app.close()
        self.tmp.cleanup()

    def request(self, path: str, method: str = "GET", body=None, headers=None):
        data = None
        hdrs = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None, dict(response.headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw) if raw else None, dict(exc.headers)

    def test_health_and_dashboard(self) -> None:
        status, payload, _ = self.request("/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"]["dim"], 128)
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("embd", html)
        self.assertIn("变化的部分", html)

    def test_index_search_stats_and_files(self) -> None:
        status, payload, _ = self.request(
            "/v1/index", "POST", {"path": str(self.root), "collection": "notes"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["files_indexed"], 1)
        self.assertGreater(payload["summary"]["chunks_added"], 0)

        status, payload, _ = self.request(
            "/v1/search", "POST", {"query": "增量索引", "collection": "notes", "limit": 3}
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(payload["count"], 1)
        self.assertIn("a.md", payload["results"][0]["path"])

        status, payload, _ = self.request("/v1/stats")
        self.assertEqual(status, 200)
        self.assertEqual(payload["stats"]["files"], 1)
        self.assertGreater(payload["stats"]["chunks"], 0)
        self.assertEqual(payload["stats"]["model"]["dim"], 128)

        status, payload, _ = self.request("/v1/files?collection=notes")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["files"]), 1)

    def test_async_job(self) -> None:
        status, payload, _ = self.request(
            "/v1/index", "POST", {"path": str(self.root), "collection": "async", "async": True}
        )
        self.assertEqual(status, 202)
        job_id = payload["job_id"]
        for _ in range(50):
            status, payload, _ = self.request(f"/v1/jobs/{job_id}")
            if payload["job"]["status"] in {"done", "error"}:
                break
            time.sleep(0.05)
        self.assertEqual(payload["job"]["status"], "done", payload)
        self.assertEqual(payload["job"]["result"]["files_indexed"], 1)

    def test_gc_and_delete(self) -> None:
        self.request("/v1/index", "POST", {"path": str(self.root), "collection": "notes"})
        status, payload, _ = self.request(
            "/v1/files", "DELETE", {"path": str(self.root / "a.md"), "collection": "notes"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["removed"], 1)
        status, payload, _ = self.request(
            "/v1/search", "POST", {"query": "增量索引", "collection": "notes"}
        )
        self.assertEqual(payload["count"], 0)

    def test_bad_requests(self) -> None:
        status, payload, _ = self.request("/v1/index", "POST", {})
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        status, payload, _ = self.request("/v1/nope")
        self.assertEqual(status, 404)
        status, payload, _ = self.request("/v1/search", "POST", {"query": ""})
        self.assertEqual(status, 400)

    def test_api_key(self) -> None:
        config = load_config(
            home=str(Path(self.tmp.name) / "home-secure"),
            overrides={"embedder.dim": 128, "api_key": "s3cret"},
        )
        app = App(config, Database(config.db_path), HashEmbedder(dim=128), watch=False)
        server = Server(app, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.port}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                self.assertEqual(response.status, 200)
            request = urllib.request.Request(base + "/v1/stats")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(ctx.exception.code, 401)
            request = urllib.request.Request(base + "/v1/stats", headers={"X-API-Key": "s3cret"})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            app.close()


if __name__ == "__main__":
    unittest.main()
