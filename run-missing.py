#!/usr/bin/env python3
"""Run every corpus × model combination that doesn't already have a result.

Auto-loads each local LM Studio model before its runs (and unloads it
afterwards to free RAM for the next). Hosted models — anything with a non-
localhost `base_url` (api.openai.com, api.anthropic.com, …) — skip the load
step entirely.

A combination is considered "done" if `results/<corpus>__<model>.json` exists.
Delete a result file to force its combination to re-run.

Usage:
    ./run-missing.py                  # run missing combinations
    ./run-missing.py --dry-run        # list what would run, don't execute
    ./run-missing.py --context 65536  # load with smaller context (default: 131072)
    ./run-missing.py --keep-loaded    # don't unload after the last model

Iteration order: outer loop = models, inner loop = corpora. This way each
model is loaded exactly once, runs every needed corpus, then is unloaded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

CORPORA_DIR = REPO_ROOT / "configs" / "corpora"
MODELS_DIR = REPO_ROOT / "configs" / "models"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_CONTEXT = 131072


def pick_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else "python3"


def is_local_server(base_url: str) -> bool:
    """True when base_url points at a localhost server (LM Studio, llama.cpp, Ollama)."""
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", ""}


def lms(*args: str, capture: bool = False) -> tuple[int, str]:
    """Invoke `lms` and return (exit_code, stdout). Logs the command for visibility."""
    cmd = ["lms", *args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    r = subprocess.run(cmd)
    return r.returncode, ""


def lms_load(model_id: str, context: int) -> bool:
    """Load `model_id` in LM Studio. Returns True on success."""
    rc, _ = lms(
        "load", model_id,
        "-c", str(context),
        "--gpu", "max",
        "--ttl", "99999",
        "-y",
    )
    return rc == 0


def lms_unload(model_id: str) -> None:
    lms("unload", model_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would run, don't execute")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                    help=f"context size for local models (default: {DEFAULT_CONTEXT})")
    ap.add_argument("--keep-loaded", action="store_true",
                    help="don't unload the last model when done")
    args = ap.parse_args()

    from bench.config import load_model

    # Discover by stem.
    corpus_stems = sorted(p.stem for p in CORPORA_DIR.glob("*.toml"))
    model_stems = sorted(p.stem for p in MODELS_DIR.glob("*.toml"))
    if not corpus_stems or not model_stems:
        print("error: no configs found in configs/corpora/ or configs/models/", file=sys.stderr)
        return 1

    total = len(corpus_stems) * len(model_stems)
    print(f"discovered {len(corpus_stems)} corpora × {len(model_stems)} models = {total} combinations\n")

    # Build a plan: per-model list of missing corpora. Skips models with nothing to do.
    plan: list[tuple[str, list[str]]] = []
    skipped = 0
    for ms in model_stems:
        missing = []
        for cs in corpus_stems:
            if (RESULTS_DIR / f"{cs}__{ms}.json").is_file():
                skipped += 1
            else:
                missing.append(cs)
        if missing:
            plan.append((ms, missing))

    if not plan:
        print("all combinations already have results — nothing to run.")
        return 0

    print(f"already have results for {skipped} combinations.")
    print("missing combinations:")
    for ms, css in plan:
        for cs in css:
            print(f"  - {cs} × {ms}")
    print()

    if args.dry_run:
        return 0

    python = pick_python()
    print(f"interpreter: {python}\n")

    new_count = 0
    failed: list[tuple[str, str, int]] = []
    failed_load: list[str] = []
    current_loaded_id: str | None = None  # the actual lms model id (e.g. "qwen3.6-35b-a3b")

    for model_stem, missing_corpora in plan:
        try:
            model_cfg, _ = load_model(model_stem)
        except Exception as e:
            print(f"\n⚠ couldn't load model config '{model_stem}': {e}", file=sys.stderr)
            failed_load.append(model_stem)
            continue

        local = is_local_server(model_cfg.client.base_url)

        if local:
            target_id = model_cfg.client.model
            if current_loaded_id and current_loaded_id != target_id:
                print(f"\n--- unloading {current_loaded_id} ---")
                lms_unload(current_loaded_id)
                current_loaded_id = None
            if current_loaded_id != target_id:
                print(f"\n--- loading {target_id} (context={args.context}) ---")
                if not lms_load(target_id, args.context):
                    print(f"⚠ failed to load {target_id}; skipping its {len(missing_corpora)} run(s)\n",
                          file=sys.stderr)
                    failed_load.append(model_stem)
                    continue
                current_loaded_id = target_id
        else:
            host = urlparse(model_cfg.client.base_url).hostname
            print(f"\n--- {model_stem} is hosted ({host}) — skipping lms load ---")

        for corpus_stem in missing_corpora:
            print(f"\n===== {corpus_stem} × {model_stem} =====")
            result_path = RESULTS_DIR / f"{corpus_stem}__{model_stem}.json"
            cmd = [python, "bench.py", "run", "--corpus", corpus_stem, "--model", model_stem]
            r = subprocess.run(cmd)
            if result_path.is_file():
                new_count += 1
            else:
                failed.append((corpus_stem, model_stem, r.returncode))
                print(f"⚠ no result file produced (exit {r.returncode}); continuing", file=sys.stderr)

    if current_loaded_id and not args.keep_loaded:
        print(f"\n--- unloading {current_loaded_id} ---")
        lms_unload(current_loaded_id)

    # Summary
    print("\n===== summary =====")
    print(f"  newly ran:                {new_count}")
    print(f"  already had result:       {skipped}")
    print(f"  failed runs:              {len(failed)}")
    print(f"  models we couldn't load:  {len(failed_load)}")
    for cs, ms, ec in failed:
        print(f"    fail: {cs} × {ms} (exit {ec})")
    for ms in failed_load:
        print(f"    no-load: {ms}")

    return 0 if not failed and not failed_load else 1


if __name__ == "__main__":
    sys.exit(main())
