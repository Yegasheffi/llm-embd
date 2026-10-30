#!/usr/bin/env python3
"""Minimal embd client using only the Python standard library."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:47821"


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    collection = sys.argv[2] if len(sys.argv) > 2 else "default"

    summary = post("/v1/index", {"path": path, "collection": collection})["summary"]
    print(f"indexed {summary['files_indexed']} files, {summary['chunks_added']} chunks, "
          f"{summary['embeddings_computed']} embeddings in {summary['duration_ms']} ms")

    result = post("/v1/search", {
        "query": "增量索引 分块对齐",
        "collection": collection,
        "limit": 5,
    })
    for rank, hit in enumerate(result["results"], 1):
        print(f"{rank}. {hit['score']:.4f}  {hit['path']}")
        print(f"   {hit['snippet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
