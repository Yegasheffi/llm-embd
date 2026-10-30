from __future__ import annotations

import unittest

import numpy as np

from embd.embedders import HashEmbedder, make_embedder


class EmbedderTests(unittest.TestCase):
    def test_hash_embedder_is_deterministic_and_normalised(self) -> None:
        embedder = HashEmbedder(dim=256)
        a = np.asarray(embedder.embed(["增量索引与分块对齐"])[0])
        b = np.asarray(embedder.embed(["增量索引与分块对齐"])[0])
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertEqual(a.shape, (256,))
        self.assertIn("hash:hash-256", embedder.fingerprint)

    def test_related_chinese_text_scores_higher(self) -> None:
        embedder = HashEmbedder(dim=512)
        query = np.asarray(embedder.embed(["增量索引 分块对齐"])[0])
        related = np.asarray(embedder.embed(["文件改动后只重建变化的部分，分块对齐只影响局部"])[0])
        unrelated = np.asarray(embedder.embed(["apple banana cherry orchard"])[0])
        self.assertGreater(float(query @ related), float(query @ unrelated))

    def test_make_embedder_defaults(self) -> None:
        embedder = make_embedder({"provider": "hash", "model": "hash-128", "dim": 128})
        self.assertEqual(embedder.dim, 128)
        self.assertEqual(embedder.name, "hash-128")

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            make_embedder({"provider": "nope", "model": "x", "dim": 128})


if __name__ == "__main__":
    unittest.main()
