"""TOML config loaders.

Configs are split along stability axis:

  configs/corpora/<name>.toml — what files to test, how to sample
  configs/models/<name>.toml  — model identifier and per-model knobs

This way an N×M comparison (N corpora, M models) needs only N+M files rather
than N*M. The two are stitched together at run time:

    python3 bench.py run --corpus jquery --model qwen36-35b

Either flag accepts a config name, an explicit path, or a `.toml` path.

Corpus schema:

    [files]
    directory = "fixtures"        # required, relative to cwd or absolute
    glob      = "*.js"            # required
    limit     = 1                 # optional cap, sorted lexically

    [sample]
    k    = 16
    seed = 42

Model schema (flat — one model per file):

    name              = "qwen3.6-35b-a3b"   # required, model identifier the server knows
    base_url          = "http://localhost:1234"
    api_key           = "not-needed"
    temperature       = 0.0
    max_tokens        = 6000
    timeout           = 600.0
    suppress_thinking = true
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .client import ClientConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPORA_DIR = REPO_ROOT / "configs" / "corpora"
MODELS_DIR = REPO_ROOT / "configs" / "models"


@dataclass
class CorpusConfig:
    name: str            # config stem, used in result filename
    directory: Path
    glob: str
    limit: int | None
    sample_k: int
    sample_seed: int


@dataclass
class ModelConfig:
    name: str            # config stem, used in result filename
    client: ClientConfig
    suppress_thinking: bool


# --- resolution -----------------------------------------------------------


def _resolve_path(name_or_path: str | Path, search_dir: Path) -> Path:
    """Find a config file by name (lookup in `search_dir`) or by literal path."""
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    candidate = search_dir / f"{name_or_path}.toml"
    if candidate.is_file():
        return candidate.resolve()
    if p.suffix and p.suffix == ".toml" and p.is_file():
        return p.resolve()
    raise FileNotFoundError(
        f"config not found: tried '{name_or_path}', '{candidate}'"
    )


# --- corpus ---------------------------------------------------------------


def load_corpus(name_or_path: str | Path) -> CorpusConfig:
    path = _resolve_path(name_or_path, CORPORA_DIR)
    raw = tomllib.loads(path.read_text())

    files_raw = raw.get("files") or {}
    if "directory" not in files_raw or "glob" not in files_raw:
        raise ValueError(f"{path}: [files] requires both `directory` and `glob`")
    directory = Path(files_raw["directory"])
    if not directory.is_absolute():
        directory = Path.cwd() / directory

    sample_raw = raw.get("sample") or {}
    return CorpusConfig(
        name=path.stem,
        directory=directory,
        glob=files_raw["glob"],
        limit=files_raw.get("limit"),
        sample_k=int(sample_raw.get("k", 16)),
        sample_seed=int(sample_raw.get("seed", 42)),
    )


# --- model ----------------------------------------------------------------


def load_model_from_file(path: Path) -> ModelConfig:
    raw = tomllib.loads(path.read_text())
    if "name" not in raw:
        raise ValueError(f"{path}: required field `name` (model identifier) is missing")
    stop_raw = raw.get("stop")
    if stop_raw is not None and not isinstance(stop_raw, list):
        raise ValueError(f"{path}: `stop` must be a list of strings if set")
    client = ClientConfig(
        base_url=raw.get("base_url", "http://localhost:1234"),
        model=raw["name"],
        api_key=raw.get("api_key", "not-needed"),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 6000)),
        timeout=float(raw.get("timeout", 600.0)),
        reasoning_effort=raw.get("reasoning_effort"),
        prefill_no_think=bool(raw.get("prefill_no_think", False)),
        stop=stop_raw,
    )
    return ModelConfig(
        name=path.stem,
        client=client,
        suppress_thinking=bool(raw.get("suppress_thinking", True)),
    )


def load_model(name_or_path: str | Path) -> tuple[ModelConfig, bool]:
    """Resolve `name_or_path` to a ModelConfig.

    Returns (config, from_file). If a matching file exists, loads it. Otherwise
    treats the input as a raw model identifier and returns sane defaults — the
    caller should print a note so the user knows we fell back.
    """
    p = Path(name_or_path)
    candidate = MODELS_DIR / f"{name_or_path}.toml"
    if p.is_file():
        return load_model_from_file(p.resolve()), True
    if candidate.is_file():
        return load_model_from_file(candidate.resolve()), True

    # Fallback: treat as raw model identifier with reasonable defaults.
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name_or_path))
    return (
        ModelConfig(
            name=safe_stem,
            client=ClientConfig(
                base_url="http://localhost:1234",
                model=str(name_or_path),
                max_tokens=6000,
            ),
            suppress_thinking=True,
        ),
        False,
    )


# --- output naming --------------------------------------------------------


def auto_dump_path(corpus: CorpusConfig, model: ModelConfig, results_dir: Path) -> Path:
    return results_dir / f"{corpus.name}__{model.name}.json"
