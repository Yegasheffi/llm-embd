from __future__ import annotations

import unittest

from embd.chunker import chunk_text
from tests.helpers import paragraph


class ChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "".join(paragraph(i) for i in range(80))

    def test_deterministic_and_offsets(self) -> None:
        first = chunk_text(self.text, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        second = chunk_text(self.text, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        self.assertEqual([c.content_hash for c in first], [c.content_hash for c in second])
        self.assertGreater(len(first), 5)
        data = self.text.encode("utf-8")
        for chunk in first:
            self.assertLessEqual(chunk.byte_length, 520)
            self.assertEqual(data[chunk.start:chunk.end].decode("utf-8"), chunk.text)
            self.assertEqual(len(chunk.content_hash), 64)

    def test_empty_and_small(self) -> None:
        self.assertEqual(chunk_text(""), [])
        chunks = chunk_text("hello")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "hello")

    def test_insert_at_top_preserves_most_chunks(self) -> None:
        base = chunk_text(self.text, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        inserted = "全新的一段。\n\n" + self.text
        changed = chunk_text(inserted, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        base_hashes = {c.content_hash for c in base}
        changed_hashes = {c.content_hash for c in changed}
        overlap = len(base_hashes & changed_hashes) / max(1, len(base_hashes))
        self.assertGreaterEqual(overlap, 0.7, f"only {overlap:.2f} chunks survived")

    def test_middle_edit_is_local(self) -> None:
        base = chunk_text(self.text, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        edited_text = self.text.replace("第 40 段", "第 40 段（已编辑）", 1)
        self.assertNotEqual(edited_text, self.text)
        edited = chunk_text(edited_text, min_bytes=64, avg_bytes=180, max_bytes=520, snap_bytes=48)
        base_hashes = {c.content_hash for c in base}
        edited_hashes = {c.content_hash for c in edited}
        overlap = len(base_hashes & edited_hashes)
        self.assertGreaterEqual(overlap, len(base_hashes) - 4)
        self.assertGreater(overlap, 0)

    def test_invalid_sizes(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("x", min_bytes=100, avg_bytes=50, max_bytes=200)


if __name__ == "__main__":
    unittest.main()
