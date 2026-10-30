from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embd.embedders import EmbeddingError, HTTPEmbedder


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "ollama"

    def log_message(self, fmt, *args):  # noqa: N802
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        inputs = payload.get("input") or [payload.get("prompt", "")]
        if self.mode == "ollama":
            body = {"embeddings": [[1.0, 2.0, 2.0] for _ in inputs]}
        else:
            body = {
                "data": [
                    {"index": i, "embedding": [3.0, 4.0, 0.0]}
                    for i in range(len(inputs))
                ]
            }
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class HTTPEmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def test_ollama_batch_and_normalisation(self) -> None:
        _Handler.mode = "ollama"
        embedder = HTTPEmbedder(provider="ollama", model="nomic", dim=3, base_url=self.base)
        vectors = embedder.embed(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(sum(v * v for v in vectors[0]), 1.0, places=5)
        self.assertIn("ollama:nomic", embedder.fingerprint)

    def test_openai_shape_and_order(self) -> None:
        _Handler.mode = "openai"
        embedder = HTTPEmbedder(
            provider="openai", model="text-embedding-3-small", dim=3,
            base_url=self.base, api_key_env="NO_SUCH_KEY",
        )
        vectors = embedder.embed(["a"])
        self.assertEqual(len(vectors[0]), 3)
        self.assertAlmostEqual(vectors[0][0], 0.6, places=5)

    def test_dimension_mismatch_is_clear(self) -> None:
        _Handler.mode = "ollama"
        embedder = HTTPEmbedder(provider="ollama", model="nomic", dim=4, base_url=self.base)
        with self.assertRaises(EmbeddingError) as ctx:
            embedder.embed(["a"])
        self.assertIn("dim=3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
