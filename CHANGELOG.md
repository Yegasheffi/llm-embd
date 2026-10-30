# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-30

### Added

- Local embedding daemon with an HTTP JSON API (`/v1/index`, `/v1/search`,
  `/v1/stats`, `/v1/files`, `/v1/watch`, `/v1/reindex`, `/v1/gc`, `/v1/jobs`).
- Three-level incremental invalidation: `mtime/size` stat fast path, whole-file
  SHA-256, and content-addressed chunk hashes.
- FastCDC-style content-defined chunking with paragraph/line/sentence boundary
  snapping and UTF-8-safe cuts.
- SQLite WAL storage with FTS5 `trigram` keyword search, vector blobs, and a
  model-fingerprinted embedding cache.
- Hybrid retrieval with reciprocal rank fusion, per-file result capping, and
  path-prefix filters.
- Background polling watcher with debounced stat scans and deletion handling.
- `hash-512` offline embedder plus Ollama and OpenAI-compatible HTTP providers.
- Text extraction for Markdown, source code, HTML, `.docx`, `.pptx`, `.xlsx`,
  `.ipynb`, and optional PDF via `pypdf`.
- CLI (`serve`, `index`, `search`, `stats`, `files`, `watch`, `reindex`,
  `remove`, `gc`, `doctor`, `jobs`, `bench`).
- Zero-build local web console with search, index jobs, collection and watcher
  status, and embedding-reuse metrics.
- Optional API key authentication, CORS headers, and graceful SIGINT/SIGTERM
  shutdown.
- `unittest` suite (58 tests) and an incremental benchmark (`embd bench`).
