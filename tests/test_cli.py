from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="embd-cli-test-")
        self.docs = Path(self.tmp.name) / "docs"
        self.docs.mkdir()
        (self.docs / "a.md").write_text(
            "# 增量索引\n\n只重建变化的部分。\n\n## 分块对齐\n\n内容定义分块。\n",
            encoding="utf-8",
        )
        self.env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "NO_COLOR": "1",
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "embd", *args],
            cwd=ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            self.fail(f"CLI failed ({result.returncode}): {result.stderr}\n{result.stdout}")
        return result

    def test_version(self) -> None:
        result = self.run_cli("version")
        self.assertIn("embd 0.1.0", result.stdout)

    def test_index_and_search_json(self) -> None:
        home = str(Path(self.tmp.name) / "home")
        result = self.run_cli("--home", home, "index", str(self.docs), "-c", "notes", "--json")
        summary = json.loads(result.stdout)
        self.assertEqual(summary["files_indexed"], 1)
        self.assertGreater(summary["chunks_added"], 0)

        result = self.run_cli(
            "search", "增量索引", "-c", "notes", "--home", home, "--json"
        )
        payload = json.loads(result.stdout)
        self.assertGreaterEqual(payload["count"], 1)
        self.assertIn("a.md", payload["results"][0]["path"])

    def test_stats_does_not_use_wrong_home(self) -> None:
        home = str(Path(self.tmp.name) / "home")
        self.run_cli("index", str(self.docs), "-c", "notes", "--home", home)
        result = self.run_cli("stats", "--home", home, "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["files"], 1)
        self.assertEqual(payload["chunks"], 1)

    def test_doctor_json(self) -> None:
        home = str(Path(self.tmp.name) / "home")
        result = self.run_cli("doctor", "--home", home, "--json")
        checks = json.loads(result.stdout)
        self.assertTrue(any(item["check"] == "FTS5" and item["ok"] for item in checks))

    def test_bad_path_exit_code(self) -> None:
        home = str(Path(self.tmp.name) / "home")
        result = self.run_cli("index", "/no/such/path/embd", "--home", home, "--json", check=False)
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
