# Contributing to embd

Thanks for considering a contribution. embd is deliberately small: a Python
package, SQLite, a local HTTP daemon, and a dependency-free web console. The
most useful contributions are correctness fixes around incremental invalidation,
new text formats, new embedding providers, and integrations.

## Before you start

- Search existing issues and pull requests first.
- For a large feature or a public API change, open an issue before writing code.
- Security issues must **not** be reported in public issues. See
  [SECURITY.md](SECURITY.md).
- By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Requires Python 3.10+ and SQLite with FTS5 (the Python builds from
python.org, Homebrew, and GitHub Actions all include it).

```bash
git clone <repo-url>
cd embd
python3 -m pip install -e ".[dev]"

python3 -m embd doctor
python3 -m unittest discover -v
python3 scripts/preflight.py
python3 -m embd bench -n 60
```

The runtime dependency footprint is intentionally small: **numpy only**.
Everything else (HTTP server, SQLite access, file extraction, chunker, CLI,
web UI) uses the standard library. Please discuss new runtime dependencies in
an issue before adding them.

## Where things live

| Path | Responsibility |
|---|---|
| `embd/chunker.py` | FastCDC gear hash, separator snapping, UTF-8 boundaries |
| `embd/indexer.py` | Three-level invalidation, atomic file swap, watcher scans |
| `embd/embedders.py` | `hash`, Ollama, OpenAI-compatible providers |
| `embd/extract.py` | Plain text and Office/HTML/PDF extraction |
| `embd/db.py` | SQLite schema, FTS5, vector cache, WAL transactions |
| `embd/search.py` | Vector + trigram retrieval, RRF fusion |
| `embd/server.py` | HTTP API, jobs, auth, CORS |
| `embd/cli.py` | CLI surface |
| `embd/static/` | Web console |
| `tests/` | `unittest` suite |

## Tests

- Add a regression test for every behavior change.
- Incremental behavior tests belong in `tests/test_incremental.py`.
- Chunk-boundary behavior tests belong in `tests/test_chunker.py`.
- API changes need a test in `tests/test_api.py`.
- Prefer small, deterministic corpora over fixtures that depend on the host.
- Never commit an index database or a real user document.

Run everything before opening a PR:

```bash
make check                 # preflight + full unittest suite
git diff --check
```

## Code style

- Python 3.10-compatible syntax, four-space indentation, type hints where useful.
- Keep the core importable without a running daemon.
- Keep writes transactional and preserve the previous index when embedding
  fails.
- Avoid logging document contents or API keys.
- UI copy is currently Chinese-first; keep it consistent and concise.
- Comments should explain *why*, especially around invalidation and ordering.

## Adding a text format

1. Implement extraction in `embd/extract.py`; return readable text, not a
   serialized object.
2. Add the extension to `DEFAULT` config in `embd/config.py`.
3. Add a focused test in `tests/test_extract.py`.
4. Note the format and any optional dependency in README.

## Adding an embedding provider

1. Subclass `BaseEmbedder` and normalise vectors to unit length.
2. Include provider, model, dimension and schema version in `fingerprint`.
3. Validate the returned dimension and batch length; raise `EmbeddingError`
   with an actionable message.
4. Add a mock-HTTP test in `tests/test_http_embedder.py` (never call a paid API
   in CI).
5. Document the configuration in README.

## Pull requests

- Keep pull requests focused; separate refactors from behavior changes.
- Explain the failure mode or user problem in the description.
- Include commands and output used for verification.
- Update `README.md` and `CHANGELOG.md` for user-visible changes.
- Make sure `scripts/preflight.py` passes and no secrets or private paths are
  included.

## Release process

Maintainers follow [RELEASING.md](RELEASING.md).
