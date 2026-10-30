from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from embd.db import Database
from embd.embedders import HashEmbedder
from embd.indexer import Indexer
from embd.search import Searcher, build_fts_query
from tests.helpers import temp_config


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="embd-search-")
        self.config = temp_config(self.tmp.name, **{"embedder.dim": 128})
        self.db = Database(self.config.db_path)
        self.embedder = HashEmbedder(dim=128)
        self.indexer = Indexer(self.config, self.db, self.embedder)
        self.searcher = Searcher(self.config, self.db, self.embedder)
        root = Path(self.tmp.name) / "docs"
        (root / "notes").mkdir(parents=True)
        (root / "code").mkdir()
        (root / "notes" / "index.md").write_text(
            "# 增量索引\n\n文件改动后只重建变化的部分。内容定义分块对齐边界。\n\n"
            "唯一标记 marker-alpha。\n",
            encoding="utf-8",
        )
        (root / "notes" / "daemon.md").write_text(
            "# 本地守护进程\n\n把索引做成常驻服务，通过 HTTP API 给任何 App 使用。\n",
            encoding="utf-8",
        )
        (root / "code" / "watcher.py").write_text(
            'def scan(root):\n    """Poll mtime and reindex changed files."""\n    pass\n',
            encoding="utf-8",
        )
        self.indexer.index_paths([str(root)], "notes", watch=False)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_chinese_hybrid(self) -> None:
        result = self.searcher.search("增量索引", collection="notes", limit=5)
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("index.md", result["results"][0]["path"])
        self.assertGreater(result["results"][0]["score"], 0)

    def test_english_keyword(self) -> None:
        result = self.searcher.search("mtime", collection="notes", mode="lexical")
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("watcher.py", result["results"][0]["path"])

    def test_vector_mode(self) -> None:
        result = self.searcher.search("分块对齐", collection="notes", mode="vector")
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("index.md", result["results"][0]["path"])

    def test_path_prefix_filter(self) -> None:
        result = self.searcher.search(
            "marker-alpha", collection="notes", path_prefix=str((Path(self.tmp.name) / "docs" / "notes").resolve())
        )
        self.assertGreaterEqual(result["count"], 1)
        self.assertTrue(all("/notes/" in item["path"] for item in result["results"]))

    def test_short_cjk_falls_back_to_like(self) -> None:
        result = self.searcher.search("索引", collection="notes", mode="lexical")
        self.assertGreaterEqual(result["count"], 1)

    def test_no_results(self) -> None:
        result = self.searcher.search("zzzzzzzz-no-such-term", collection="notes")
        self.assertEqual(result["count"], 0)

    def test_empty_query(self) -> None:
        result = self.searcher.search("", collection="notes")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["results"], [])

    def test_fts_query_builder(self) -> None:
        self.assertIsNone(build_fts_query("a"))
        query = build_fts_query("增量索引")
        self.assertIn("增量索引", query or "")
        self.assertIn("增量索", query or "")

    def test_result_shape(self) -> None:
        result = self.searcher.search("增量", collection="notes")
        item = result["results"][0]
        for key in ("id", "path", "score", "snippet", "text", "content_hash", "vector_score"):
            self.assertIn(key, item)
        self.assertEqual(result["model"], self.embedder.name)


if __name__ == "__main__":
    unittest.main()
