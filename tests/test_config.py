from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from embd.config import load_config


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(home=tmp)
            self.assertEqual(config.get("port"), 47821)
            self.assertEqual(config.get("embedder.provider"), "hash")
            self.assertEqual(config.chunk["avg_bytes"], 1200)
            self.assertEqual(config.db_path, Path(tmp).resolve() / "index.db")

    def test_file_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home.mkdir(exist_ok=True)
            (home / "config.json").write_text(
                json.dumps({"port": 5001, "chunk": {"avg_bytes": 900}}), encoding="utf-8"
            )
            with mock.patch.dict(os.environ, {"EMBD_PORT": "5002", "EMBD_DIM": "256"}):
                config = load_config(home=tmp)
            self.assertEqual(config.get("port"), 5002)
            self.assertEqual(config.get("embedder.dim"), 256)
            self.assertEqual(config.get("chunk.avg_bytes"), 900)

    def test_cli_overrides_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(home=tmp, overrides={"port": 6000, "embedder.dim": 64})
            self.assertEqual(config.get("port"), 6000)
            self.assertEqual(config.get("embedder.dim"), 64)

    def test_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(home=tmp, config_path=str(path))

    def test_invalid_chunk_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_config(
                    home=tmp,
                    overrides={"chunk.min_bytes": 500, "chunk.avg_bytes": 100},
                )

    def test_public_config_hides_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(home=tmp, overrides={"api_key": "top-secret"})
            public = config.to_public_dict()
            self.assertNotIn("api_key", public)
            self.assertNotIn("top-secret", json.dumps(public))


if __name__ == "__main__":
    unittest.main()
