from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from embd.config import Config, load_config


SMALL_CHUNKS = {
    "chunk.min_bytes": 64,
    "chunk.avg_bytes": 180,
    "chunk.max_bytes": 520,
    "chunk.snap_bytes": 48,
}


def temp_config(root: str | Path, **overrides: Any) -> Config:
    home = Path(root) / "home"
    values = {**SMALL_CHUNKS, **overrides}
    return load_config(home=str(home), overrides=values)


def paragraph(index: int) -> str:
    return (
        f"## 第 {index} 节\n\n"
        f"这是第 {index} 段关于增量索引与分块对齐的说明。"
        f"唯一标记 marker-{index:03d} 用来验证检索命中。"
        f"内容定义分块让插入只影响局部，而不是整篇重算。\n\n"
    )
