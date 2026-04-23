"""TOML config schema and loader.

A config bundles a corpus selection (glob over a directory) with sample size
and model settings, so a benchmark run is one command:

    python3 bench.py run --config configs/jquery.toml

Schema (all keys optional unless noted):

    [files]
    directory = "fixtures"     # required, relative to repo root or absolute
    glob = "*.js"              # required
    limit = 1                  # optional cap on files matched (sorted lexically)

    [sample]
    k = 16                     # how many functions to sample
    seed = 42

    [model]
    base_url = "http://localhost:1234"
    name = "qwen3.6-35b-a3b"   # required
    api_key = "not-needed"
    temperature = 0.0
    max_tokens = 6000
    timeout = 600.0
    suppress_thinking = true
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .client import ClientConfig


@dataclass
class FilesConfig:
    directory: Path
    glob: str
    limit: int | None = None


@dataclass
class SampleConfig:
    k: int = 16
    seed: int = 42


@dataclass
class BenchConfig:
    name: str            # derived from config file stem
    files: FilesConfig
    sample: SampleConfig = field(default_factory=SampleConfig)
    model: ClientConfig = field(default_factory=lambda: ClientConfig(base_url="http://localhost:1234", model=""))
    suppress_thinking: bool = True


def load_config(path: Path) -> BenchConfig:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = tomllib.loads(path.read_text())

    files_raw = raw.get("files") or {}
    if "directory" not in files_raw or "glob" not in files_raw:
        raise ValueError(f"{path}: [files] requires both `directory` and `glob`")
    directory = Path(files_raw["directory"])
    if not directory.is_absolute():
        # resolve relative to the config file's parent's parent (repo root)
        # actually: relative to current working directory, which is the natural choice
        directory = Path.cwd() / directory
    files = FilesConfig(
        directory=directory,
        glob=files_raw["glob"],
        limit=files_raw.get("limit"),
    )

    sample_raw = raw.get("sample") or {}
    sample = SampleConfig(
        k=int(sample_raw.get("k", 16)),
        seed=int(sample_raw.get("seed", 42)),
    )

    model_raw = raw.get("model") or {}
    if "name" not in model_raw:
        raise ValueError(f"{path}: [model] requires `name`")
    model = ClientConfig(
        base_url=model_raw.get("base_url", "http://localhost:1234"),
        model=model_raw["name"],
        api_key=model_raw.get("api_key", "not-needed"),
        temperature=float(model_raw.get("temperature", 0.0)),
        max_tokens=int(model_raw.get("max_tokens", 6000)),
        timeout=float(model_raw.get("timeout", 600.0)),
    )
    suppress_thinking = bool(model_raw.get("suppress_thinking", True))

    return BenchConfig(
        name=path.stem,
        files=files,
        sample=sample,
        model=model,
        suppress_thinking=suppress_thinking,
    )


def sanitize_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def auto_dump_path(cfg: BenchConfig, results_dir: Path) -> Path:
    model_part = sanitize_for_filename(cfg.model.model)
    return results_dir / f"{cfg.name}__{model_part}.json"
