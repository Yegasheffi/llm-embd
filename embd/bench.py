"""A self-contained benchmark for the incremental claim.

It creates a synthetic corpus, indexes it, edits a subset, reindexes, and
reports how many embeddings had to be recomputed. Run it with ``embd bench``.
No network or external model is used, so numbers are deterministic and
comparable across machines.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .db import Database
from .embedders import HashEmbedder
from .indexer import Indexer
from .search import Searcher
from .util import canonical_path, human_duration


_TOPICS = [
    ("incremental indexing", "增量索引", "只重建变化的部分", "mtime 与内容哈希"),
    ("chunk alignment", "分块对齐", "内容定义分块", "插入不扰动后续向量"),
    ("local daemon", "本地守护进程", "HTTP API", "任何应用都能调用"),
    ("hybrid retrieval", "混合检索", "向量与关键词", "倒数排名融合"),
    ("sqlite storage", "SQLite 存储", "WAL 模式", "单文件即可迁移"),
    ("embedding cache", "向量缓存", "内容寻址", "跨文件复用"),
    ("polling watcher", "轮询监听", "stat 快路径", "文件落盘后自动刷新"),
    ("privacy", "隐私", "数据不离开本机", "本地模型可选"),
    ("developer experience", "开发体验", "命令行与面板", "零依赖启动"),
    ("observability", "可观测性", "命中率统计", "任务与错误可见"),
]


def _document(rng: random.Random, topic: tuple[str, ...], marker: str) -> str:
    title, cn, phrase, detail = topic
    paragraphs = [
        f"# {title} / {cn}\n",
        f"本文讨论 {cn}：{phrase}。{detail}。\n",
    ]
    for i in range(36):
        paragraphs.append(
            f"## {cn} 第 {i + 1} 节\n\n"
            f"{phrase} 的核心是让系统在文件变化后只处理真正改变的部分。"
            f"这一节记录一个唯一标记 {marker}-{i:02d}，用于验证检索是否命中。"
            f"同时补充一些自然语言上下文：{title} 需要稳定的边界、可解释的分数、"
            f"以及足够低的维护成本。\n"
        )
    rng.shuffle(paragraphs)
    return "\n".join(paragraphs)


def run_benchmark(
    *,
    docs: int = 60,
    seed: int = 7,
    base_config: Config | None = None,
) -> dict[str, Any]:
    docs = max(6, min(500, int(docs)))
    rng = random.Random(seed)
    started = time.perf_counter()

    source = base_config or load_config()
    data = copy.deepcopy(source.data)
    data["embedder"] = {
        **data["embedder"],
        "provider": "hash",
        "model": f"hash-{data['embedder'].get('dim', 512)}",
    }
    data["watch"]["enabled"] = False

    with tempfile.TemporaryDirectory(prefix="embd-bench-") as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        config = Config(data, home=home)
        db = Database(config.db_path)
        embedder = HashEmbedder(dim=int(data["embedder"].get("dim", 512)))
        indexer = Indexer(config, db, embedder)
        searcher = Searcher(config, db, embedder)
        fs_root = Path(canonical_path(Path(tmp) / "corpus"))
        fs_root.mkdir()
        root = str(fs_root)

        markers: dict[str, str] = {}
        for i in range(docs):
            topic = _TOPICS[i % len(_TOPICS)]
            marker = f"mk{i:03d}"
            markers[str(fs_root / f"doc-{i:03d}.md")] = marker
            (fs_root / f"doc-{i:03d}.md").write_text(
                _document(rng, topic, marker), encoding="utf-8"
            )

        first = indexer.index_paths([root], "bench", watch=False)
        first_vectors = first.embeddings_computed
        first_chunks = first.chunks_added

        # Simulate realistic edits: some prepends (worst case for fixed-size
        # chunking), some inserts, some appends.
        edited = 0
        for i, (path, marker) in enumerate(sorted(markers.items())):
            if i % 5 != 0:
                continue
            edited += 1
            original = Path(path).read_text(encoding="utf-8")
            if i % 10 == 0:
                updated = f"## 新增前言\n\n这是一个插入到文档开头的段落 {marker}-new。\n\n{original}"
            else:
                updated = original + f"\n\n## 新增结语\n\n追加的结论段落 {marker}-end。\n"
            Path(path).write_text(updated, encoding="utf-8")

        second = indexer.index_paths([root], "bench", watch=False)
        reused = second.chunks_reused
        recomputed = second.embeddings_computed
        reduction = 1.0 - (recomputed / first_vectors) if first_vectors else 0.0

        queries = list(markers.values())[: min(30, len(markers))]
        hits = 0
        for marker in queries:
            result = searcher.search(marker, collection="bench", limit=3)
            if any(marker in item["text"] for item in result["results"]):
                hits += 1
        hit_rate = hits / len(queries) if queries else 0.0

        elapsed = time.perf_counter() - started
        stats = db.stats(embedder.fingerprint)
        db.close()

    return {
        "docs": docs,
        "edited_docs": edited,
        "first": {
            "chunks": first_chunks,
            "embeddings": first_vectors,
            "duration_ms": first.duration_ms,
        },
        "second": {
            "files_indexed": second.files_indexed,
            "chunks_added": second.chunks_added,
            "chunks_reused": reused,
            "embeddings": recomputed,
            "duration_ms": second.duration_ms,
        },
        "embedding_reduction": round(reduction, 4),
        "reuse_ratio": round(reused / max(1, reused + recomputed), 4),
        "search_hit_rate": round(hit_rate, 4),
        "vector_cache": stats["active_vectors"],
        "total_seconds": round(elapsed, 3),
        "total_human": human_duration(elapsed),
    }


def format_benchmark(report: dict[str, Any]) -> str:
    lines = [
        "embd incremental benchmark",
        "─" * 40,
        f"documents            {report['docs']} (edited {report['edited_docs']})",
        f"initial chunks       {report['first']['chunks']}",
        f"initial embeddings   {report['first']['embeddings']}",
        f"after edits          {report['second']['chunks_added']} chunks written, "
        f"{report['second']['embeddings']} embeddings recomputed",
        f"chunks reused        {report['second']['chunks_reused']} "
        f"({report['reuse_ratio'] * 100:.1f}%)",
        f"embedding reduction  {report['embedding_reduction'] * 100:.1f}%",
        f"search hit rate      {report['search_hit_rate'] * 100:.1f}%",
        f"elapsed              {report['total_human']}",
    ]
    return "\n".join(lines)
