from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from embd.config import Config
from embd.db import Database
from embd.embedders import HashEmbedder
from embd.indexer import Indexer
from embd.search import Searcher
from tests.helpers import paragraph, temp_config


class IncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="embd-test-")
        self.config = temp_config(self.tmp.name, **{"embedder.dim": 128})
        self.db = Database(self.config.db_path)
        self.embedder = HashEmbedder(dim=128)
        self.indexer = Indexer(self.config, self.db, self.embedder)
        self.searcher = Searcher(self.config, self.db, self.embedder)
        self.root = Path(self.tmp.name) / "docs"
        self.root.mkdir()
        self.path = self.root / "guide.md"
        self.path.write_text("".join(paragraph(i) for i in range(24)), encoding="utf-8")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _index(self, **kwargs):
        return self.indexer.index_paths([str(self.root)], "notes", watch=False, **kwargs)

    def _bump_mtime(self) -> None:
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    def test_three_levels_of_incrementality(self) -> None:
        first = self._index()
        self.assertEqual(first.files_indexed, 1)
        self.assertGreater(first.chunks_added, 4)
        self.assertEqual(first.embeddings_computed, first.chunks_added)

        second = self._index()
        self.assertEqual(second.files_unchanged, 1)
        self.assertEqual(second.embeddings_computed, 0)
        self.assertEqual(second.bytes_read, 0)

        self._bump_mtime()
        third = self._index()
        self.assertEqual(third.files_unchanged_hash, 1)
        self.assertEqual(third.embeddings_computed, 0)
        self.assertEqual(third.bytes_read, len(self.path.read_bytes()))

        original = self.path.read_text(encoding="utf-8")
        changed = original.replace("第 10 段", "第 10 段（修改后的内容）", 1)
        self.path.write_text(changed, encoding="utf-8")
        self._bump_mtime()
        fourth = self._index()
        self.assertEqual(fourth.files_indexed, 1)
        self.assertGreater(fourth.chunks_reused, 0)
        self.assertLess(fourth.embeddings_computed, fourth.chunks_added)
        self.assertGreaterEqual(fourth.embeddings_computed, 1)

    def test_insert_at_top_only_reembeds_local_chunks(self) -> None:
        self._index()
        original = self.path.read_text(encoding="utf-8")
        self.path.write_text("## 新增前言\n\n这是插入到开头的一段全新内容。\n\n" + original, encoding="utf-8")
        self._bump_mtime()
        summary = self._index()
        self.assertGreater(summary.chunks_reused, 0)
        self.assertLess(summary.embeddings_computed, max(3, summary.chunks_added // 2))

    def test_duplicate_content_reuses_embedding_across_files(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        (self.root / "copy.md").write_text(text, encoding="utf-8")
        summary = self._index()
        self.assertEqual(summary.files_indexed, 2)
        self.assertEqual(summary.embeddings_computed, summary.chunks_added // 2)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["chunks"], summary.chunks_added)

    def test_delete_removes_chunks_and_fts_rows(self) -> None:
        self._index()
        self.assertGreater(self.searcher.search("增量索引", collection="notes")["count"], 0)
        self.path.unlink()
        watch = self.db.list_watch_roots()
        self.assertEqual(watch, [])  # watch=False above
        removed = self.db.delete_missing_under("notes", str(self.root.resolve()), set())
        self.assertEqual(len(removed), 1)
        self.assertEqual(self.searcher.search("增量索引", collection="notes")["count"], 0)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["chunks"], 0)

    def test_model_change_needs_reindex_but_keeps_old_vectors(self) -> None:
        self._index()
        old_vectors = self.db.stats(self.embedder.fingerprint)["active_vectors"]
        self.assertGreater(old_vectors, 0)
        other = HashEmbedder(dim=128, model="hash-alt")
        other_indexer = Indexer(self.config, self.db, other)
        other_searcher = Searcher(self.config, self.db, other)
        result = other_searcher.search("增量索引", collection="notes", mode="vector")
        self.assertEqual(result["count"], 0)
        self.assertTrue(result["warnings"])
        other_indexer.reindex_collection("notes", force=True)
        healed = other_searcher.search("增量索引", collection="notes", mode="vector")
        self.assertGreater(healed["count"], 0)
        # The old vector space is still cached, switching back is instant.
        self.assertGreater(self.db.stats(self.embedder.fingerprint)["active_vectors"], 0)

    def test_binary_file_is_skipped(self) -> None:
        binary = self.root / "blob.txt"
        binary.write_bytes(b"\x00\x01\x02binary" * 100)
        summary = self.indexer.index_paths([str(binary)], "notes", watch=False)
        self.assertEqual(summary.files_skipped, 1)
        self.assertEqual(summary.files_indexed, 0)
        self.assertEqual(len(self.db.file_rows("notes")), 0)

    def test_empty_file_clears_chunks(self) -> None:
        self._index()
        self.path.write_text("", encoding="utf-8")
        self._bump_mtime()
        summary = self._index()
        self.assertEqual(summary.files_indexed, 1)
        self.assertEqual(summary.chunks_added, 0)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["chunks"], 0)

    def test_touch_only_does_not_call_embedder(self) -> None:
        self._index()
        calls = {"n": 0}
        original = self.embedder.embed

        def counting(texts):
            calls["n"] += 1
            return original(texts)

        self.embedder.embed = counting  # type: ignore[method-assign]
        self._bump_mtime()
        self._index()
        self.assertEqual(calls["n"], 0)

    def test_watcher_scan_indexes_change_and_removal(self) -> None:
        self.indexer.index_paths([str(self.root)], "notes", watch=True)
        watches = self.db.list_watch_roots()
        self.assertEqual(len(watches), 1)
        self.assertTrue(watches[0]["is_dir"])

        original = self.path.read_text(encoding="utf-8")
        self.path.write_text(original + "\n\n追加的新段落 marker-new。\n", encoding="utf-8")
        self._bump_mtime()
        summary = self.indexer.scan_root(watches[0])
        self.assertEqual(summary.files_indexed, 1)
        self.assertGreater(summary.embeddings_computed, 0)
        self.assertIsNotNone(self.db.list_watch_roots()[0]["last_scan_at"])

        self.path.unlink()
        removed = self.indexer.scan_root(watches[0])
        self.assertEqual(removed.files_removed, 1)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["files"], 0)

    def test_gc_drops_orphaned_vectors(self) -> None:
        self._index()
        self.assertGreater(self.db.embedding_count(self.embedder.fingerprint), 0)
        self.path.unlink()
        self.db.delete_missing_under("notes", str(self.root.resolve()), set())
        self.assertGreater(self.db.gc_embeddings(), 0)
        self.assertEqual(self.db.embedding_count(self.embedder.fingerprint), 0)

    def test_delete_collection_cleans_fts(self) -> None:
        self._index()
        self.assertGreater(self.searcher.search("增量索引", collection="notes")["count"], 0)
        self.assertTrue(self.db.delete_collection("notes"))
        self.assertEqual(self.searcher.search("增量索引", collection="notes")["count"], 0)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["files"], 0)
        self.assertEqual(self.db.stats(self.embedder.fingerprint)["chunks"], 0)

    def test_embdignore_and_hidden_dirs(self) -> None:
        (self.root / ".embdignore").write_text("secret.md\nprivate/\n", encoding="utf-8")
        (self.root / "secret.md").write_text("should not be indexed", encoding="utf-8")
        (self.root / "private").mkdir()
        (self.root / "private" / "x.md").write_text("hidden", encoding="utf-8")
        (self.root / ".hidden").mkdir()
        (self.root / ".hidden" / "y.md").write_text("hidden", encoding="utf-8")
        summary = self._index()
        paths = [Path(row["path"]).relative_to(self.root.resolve()) for row in self.db.file_rows("notes")]
        self.assertFalse(any(path.name == "secret.md" for path in paths))
        self.assertFalse(any(path.parts[0] == "private" for path in paths))
        self.assertFalse(any(path.parts[0] == ".hidden" for path in paths))
        self.assertGreaterEqual(summary.files_indexed, 1)

    def test_missing_path_reports_error(self) -> None:
        summary = self.indexer.index_paths([str(self.root / "nope")], "notes", watch=False)
        self.assertEqual(len(summary.errors), 1)
        self.assertEqual(summary.files_indexed, 0)


if __name__ == "__main__":
    unittest.main()
