"""Configuration loading.

Resolution order (later wins):

1. built-in defaults
2. ``$EMBD_HOME/config.json`` (or ``$EMBD_CONFIG``)
3. environment variables
4. explicit CLI overrides

Keeping this boring is intentional: the daemon has to be trivially embeddable
in other projects, and a single optional JSON file is easier to reason about
than a layered settings framework.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 47821,
    "api_key": None,
    "embedder": {
        "provider": "hash",
        "model": "hash-512",
        "dim": 512,
        "batch_size": 32,
        "timeout": 60,
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "chunk": {
        "min_bytes": 384,
        "avg_bytes": 1200,
        "max_bytes": 4800,
        "snap_bytes": 320,
    },
    "index": {
        "max_file_bytes": 2 * 1024 * 1024,
        "max_files": 200_000,
        "follow_symlinks": False,
        "ignore_dirs": [
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            "target",
            ".next",
            ".nuxt",
            ".cache",
            ".idea",
            ".vscode",
            ".DS_Store",
            "Library",
        ],
        "ignore_globs": [
            "*.pyc",
            "*.pyo",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.exe",
            "*.bin",
            "*.lock",
            "*.min.js",
            "*.min.css",
            "*.map",
        ],
        "text_extensions": [
            ".md", ".markdown", ".mdx", ".txt", ".text", ".rst", ".org",
            ".html", ".htm", ".xml", ".css", ".scss", ".sass", ".less",
            ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte",
            ".py", ".pyi", ".rb", ".go", ".rs", ".java", ".kt", ".kts",
            ".swift", ".m", ".mm", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
            ".cs", ".php", ".pl", ".lua", ".r", ".jl", ".scala", ".clj",
            ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".graphql",
            ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini",
            ".cfg", ".conf", ".env", ".properties", ".gradle", ".csv",
            ".tsv", ".tex", ".bib", ".log", ".ipynb", ".docx", ".pptx",
            ".pdf",
        ],
        "text_basenames": [
            "Makefile", "Dockerfile", "Containerfile", "Procfile", "Rakefile",
            "Gemfile", "Cargo.lock", "go.mod", "go.sum", "requirements.txt",
            ".gitignore", ".dockerignore", ".editorconfig", ".env", "LICENSE", "NOTICE",
        ],
    },
    "watch": {
        "enabled": True,
        "interval_seconds": 5.0,
    },
    "search": {
        "default_limit": 10,
        "candidate_limit": 60,
        "hybrid": True,
        "vector_weight": 0.65,
        "lexical_weight": 0.35,
        "max_per_path": 2,
        "min_vector_score": 0.12,
    },
    "server": {
        "cors_origin": "*",
        "max_body_bytes": 1024 * 1024,
    },
}

_ENV_MAP = {
    "EMBD_HOST": ("host", str),
    "EMBD_PORT": ("port", int),
    "EMBD_API_KEY": ("api_key", str),
    "EMBD_PROVIDER": ("embedder.provider", str),
    "EMBD_MODEL": ("embedder.model", str),
    "EMBD_DIM": ("embedder.dim", int),
    "EMBD_BASE_URL": ("embedder.base_url", str),
    "EMBD_BATCH_SIZE": ("embedder.batch_size", int),
}


def default_home() -> Path:
    return Path(os.environ.get("EMBD_HOME", "~/.embd")).expanduser()


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _validate(data: dict[str, Any]) -> None:
    if not (1 <= int(data["port"]) <= 65535):
        raise ValueError("port must be between 1 and 65535")
    chunk = data["chunk"]
    if not (0 < int(chunk["min_bytes"]) <= int(chunk["avg_bytes"]) <= int(chunk["max_bytes"])):
        raise ValueError("chunk sizes must satisfy 0 < min <= avg <= max")
    if int(data["embedder"]["dim"]) < 8:
        raise ValueError("embedder.dim must be >= 8")
    if int(data["embedder"]["batch_size"]) < 1:
        raise ValueError("embedder.batch_size must be >= 1")


class Config:
    """A validated, read-only-ish configuration object."""

    def __init__(self, data: dict[str, Any], *, home: Path, path: Path | None = None):
        self.data = _validate_and_return(data)
        self.home = home.expanduser().resolve()
        self.path = path

    @property
    def db_path(self) -> Path:
        return self.home / "index.db"

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def embedder(self) -> dict[str, Any]:
        return self.data["embedder"]

    @property
    def chunk(self) -> dict[str, Any]:
        return self.data["chunk"]

    @property
    def index(self) -> dict[str, Any]:
        return self.data["index"]

    @property
    def search(self) -> dict[str, Any]:
        return self.data["search"]

    def to_public_dict(self) -> dict[str, Any]:
        """Config safe to expose over HTTP (never includes API secrets)."""
        data = copy.deepcopy(self.data)
        data.pop("api_key", None)
        if data.get("embedder", {}).get("api_key_env"):
            data["embedder"]["api_key_env"] = data["embedder"]["api_key_env"]
        return data


def _validate_and_return(data: dict[str, Any]) -> dict[str, Any]:
    _validate(data)
    return data


def load_config(
    *,
    home: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    resolved_home = (
        Path(home).expanduser()
        if home
        else Path(os.environ["EMBD_HOME"]).expanduser()
        if os.environ.get("EMBD_HOME")
        else default_home()
    )
    resolved_home = resolved_home.resolve()

    explicit = config_path or os.environ.get("EMBD_CONFIG")
    path = Path(explicit).expanduser().resolve() if explicit else resolved_home / "config.json"

    data = copy.deepcopy(DEFAULTS)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid config JSON at {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"config at {path} must contain a JSON object")
        data = _deep_merge(data, loaded)

    for env_name, (dotted, cast) in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            try:
                _set_dotted(data, dotted, cast(raw))
            except ValueError as exc:
                raise ValueError(f"invalid {env_name}: {raw!r}") from exc

    if overrides:
        for dotted, value in overrides.items():
            if value is not None:
                _set_dotted(data, dotted, value)

    return Config(data, home=resolved_home, path=path if path.exists() else None)
