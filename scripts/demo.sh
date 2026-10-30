#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO="$(mktemp -d "${TMPDIR:-/tmp}/embd-demo-XXXX")"
KEEP="${KEEP:-0}"

cleanup() {
  if [[ "$KEEP" == "1" ]]; then
    echo "kept demo at $DEMO"
  else
    rm -rf "$DEMO"
  fi
}
trap cleanup EXIT

mkdir -p "$DEMO/notes"
cat > "$DEMO/notes/incremental-index.md" <<'MD'
# 增量索引

文件改动后只需要重建变化的部分。先比 mtime，再比内容哈希，最后用分块哈希复用向量。

## 分块对齐

固定窗口会让插入点之后的边界全部平移。内容定义分块（CDC）只扰动局部。
MD

cat > "$DEMO/notes/local-daemon.md" <<'MD'
# 本地守护进程

把索引层做成常驻的本地 HTTP 服务。任何 App 通过 `/v1/search` 使用，不需要 SDK。

## 隐私

默认只绑定 127.0.0.1，数据和模型都不离开本机。
MD

cat > "$DEMO/notes/hybrid-search.md" <<'MD'
# Hybrid retrieval

Vector recall and trigram keyword recall are fused with reciprocal rank fusion.
This combines semantic recall with exact keyword precision.

## Ranking

RRF avoids comparing raw scores from two different retrieval systems.
MD

export EMBD_HOME="$DEMO/home"
PYTHONPATH="$ROOT" python3 -m embd index "$DEMO/notes" --collection demo
PYTHONPATH="$ROOT" python3 -m embd search "分块对齐" --collection demo
PYTHONPATH="$ROOT" python3 -m embd stats

printf '\nNow edit a file and run the same index command again:\n'
printf '  EMBD_HOME=%s PYTHONPATH=%s python3 -m embd index %s/notes -c demo\n\n' "$DEMO" "$ROOT" "$DEMO"
PYTHONPATH="$ROOT" python3 -m embd bench -n 20
