#!/usr/bin/env python3
"""Positional recall benchmark — CLI entry.

Tests an LLM's ability to reproduce the first N lines of a named function
inside a large source corpus loaded into context. Measures positional recall,
not just named-entity lookup.

Usage modes:
  - Config-driven (recommended):  python3 bench.py run --config configs/X.toml
  - Single file:                  python3 bench.py run --file path/to/source.py --model X
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _load_source_for_command(args: argparse.Namespace):
    """Resolve --config or --file into a Source. Used by extract and run."""
    from bench.extract import load_source_glob
    from bench.runner import source_from_single_file

    if getattr(args, "config", None):
        from bench.config import load_config

        cfg = load_config(Path(args.config))
        src = load_source_glob(cfg.files.directory, cfg.files.glob, cfg.files.limit)
        return src, cfg
    if not getattr(args, "file", None):
        raise SystemExit("error: pass either --config CONFIG.toml or --file PATH")
    return source_from_single_file(Path(args.file)), None


def cmd_extract(args: argparse.Namespace) -> int:
    from bench.extract import stratified_sample

    source, _bench_cfg = _load_source_for_command(args)

    if args.show:
        match = next((t for t in source.targets if t.name == args.show), None)
        if match is None:
            print(f"function {args.show!r} not found")
            return 1
        loc = f"  ({match.source_path})" if match.source_path else ""
        print(f"# {match.name} — start_line={match.start_line}  body_lines={len(match.body_lines)}{loc}")
        print(f"# -- primary (first {len(match.primary_lines)}) --")
        for i, l in enumerate(match.primary_lines, 1):
            print(f"{i:>3}| {l}")
        if match.bonus_lines:
            print(f"# -- bonus (next {len(match.bonus_lines)}) --")
            for i, l in enumerate(match.bonus_lines, len(match.primary_lines) + 1):
                print(f"{i:>3}| {l}")
        return 0

    total_lines = source.text.count("\n") + 1
    print(
        f"{len(source.targets)} function(s) with ≥20 body lines across "
        f"{len(source.files)} file(s) ({len(source.text):,} chars, {total_lines:,} lines)"
    )
    if args.all:
        chosen = source.targets
    else:
        chosen = stratified_sample(source.targets, total_lines, k=args.k, seed=args.seed)
        print(f"stratified sample of {len(chosen)}:")
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(f"  {t.name:<40}  line={t.start_line:>6}  body_lines={len(t.body_lines)}{loc}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from bench.client import ClientConfig
    from bench.config import auto_dump_path
    from bench.runner import run_benchmark

    source, bench_cfg = _load_source_for_command(args)

    if bench_cfg is not None:
        # Config-driven: model settings come from config; CLI flags override
        model_cfg = bench_cfg.model
        if args.model:
            model_cfg.model = args.model
        if args.base_url:
            model_cfg.base_url = args.base_url
        if args.api_key:
            model_cfg.api_key = args.api_key
        if args.temperature is not None:
            model_cfg.temperature = args.temperature
        if args.max_tokens is not None:
            model_cfg.max_tokens = args.max_tokens
        if args.timeout is not None:
            model_cfg.timeout = args.timeout
        k = args.k if args.k is not None else bench_cfg.sample.k
        seed = args.seed if args.seed is not None else bench_cfg.sample.seed
        suppress_thinking = bench_cfg.suppress_thinking and not args.think

        if args.dump:
            dump_path = Path(args.dump)
        else:
            DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            dump_path = auto_dump_path(bench_cfg, DEFAULT_RESULTS_DIR)
    else:
        # Single-file mode: model required from flags
        if not args.model:
            raise SystemExit("error: --model is required when not using --config")
        model_cfg = ClientConfig(
            base_url=args.base_url or "http://localhost:1234",
            model=args.model,
            api_key=args.api_key or "not-needed",
            temperature=args.temperature if args.temperature is not None else 0.0,
            max_tokens=args.max_tokens if args.max_tokens is not None else 6000,
            timeout=args.timeout if args.timeout is not None else 600.0,
        )
        k = args.k if args.k is not None else 16
        seed = args.seed if args.seed is not None else 42
        suppress_thinking = not args.think
        dump_path = Path(args.dump) if args.dump else None

    fn_filter = args.function if args.function else None
    scores = run_benchmark(
        source=source,
        cfg=model_cfg,
        k=k,
        seed=seed,
        dump_path=dump_path,
        function_filter=fn_filter,
        suppress_thinking=suppress_thinking,
    )
    passed = sum(1 for s in scores if s.passed)
    return 0 if passed == len(scores) else 1


def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-score a previous run's dump without re-querying the model."""
    import json

    from bench.extract import load_source_glob
    from bench.report import render_function, render_summary
    from bench.runner import source_from_single_file
    from bench.scorer import score

    dump = json.loads(Path(args.dump).read_text())
    if args.config:
        from bench.config import load_config

        cfg = load_config(Path(args.config))
        source = load_source_glob(cfg.files.directory, cfg.files.glob, cfg.files.limit)
    elif args.file:
        source = source_from_single_file(Path(args.file))
    else:
        # try the dump itself
        files = dump.get("files") or ([dump["source"]] if dump.get("source") else [])
        if len(files) == 1:
            source = source_from_single_file(Path(files[0]))
        else:
            raise SystemExit(
                "error: original corpus had multiple files; pass --config or --file to re-locate them"
            )

    targets = {t.name: t for t in source.targets}
    scores = []
    for r in dump["results"]:
        t = targets.get(r["function"])
        if t is None:
            print(f"skip: {r['function']} not found in source", file=sys.stderr)
            continue
        sc = score(t.name, t.primary_lines, t.bonus_lines, r.get("response", ""))
        scores.append(sc)
        print(render_function(sc))
    print(render_summary(scores))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ------------------------------------------------------------
    p_ex = sub.add_parser("extract", help="list functions the extractor would test")
    src_grp = p_ex.add_mutually_exclusive_group()
    src_grp.add_argument("--config", help="TOML config (configs/<name>.toml)")
    src_grp.add_argument("--file", help="single source file")
    p_ex.add_argument("file_pos", nargs="?", help=argparse.SUPPRESS)  # legacy positional
    p_ex.add_argument("-k", type=int, default=16)
    p_ex.add_argument("--seed", type=int, default=42)
    p_ex.add_argument("--all", action="store_true", help="list every extracted function, not a sample")
    p_ex.add_argument("--show", metavar="NAME", help="print expected primary+bonus lines for one function")
    p_ex.set_defaults(func=cmd_extract)

    # --- run ----------------------------------------------------------------
    p_run = sub.add_parser("run", help="run the benchmark against an OpenAI-compatible endpoint")
    src_grp = p_run.add_mutually_exclusive_group()
    src_grp.add_argument("--config", help="TOML config (configs/<name>.toml)")
    src_grp.add_argument("--file", help="single source file (legacy mode)")
    p_run.add_argument("file_pos", nargs="?", help=argparse.SUPPRESS)  # legacy positional
    p_run.add_argument("--base-url", default=None, help="overrides config")
    p_run.add_argument("--model", default=None, help="overrides config; required in --file mode")
    p_run.add_argument("--api-key", default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("-k", type=int, default=None, help="overrides config")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument(
        "--dump",
        default=None,
        help="JSON path for full results (default: results/<config-stem>__<model>.json)",
    )
    p_run.add_argument("--function", action="append", help="repeatable; overrides sampling")
    p_run.add_argument("--think", action="store_true", help="allow chain-of-thought (default: suppress)")
    p_run.set_defaults(func=cmd_run)

    # --- rescore ------------------------------------------------------------
    p_rs = sub.add_parser("rescore", help="re-score a previous --dump without re-querying")
    p_rs.add_argument("dump", help="path to JSON dump from a prior `run`")
    src_grp = p_rs.add_mutually_exclusive_group()
    src_grp.add_argument("--config", help="re-locate corpus via this config")
    src_grp.add_argument("--file", help="re-locate corpus from a single file")
    p_rs.set_defaults(func=cmd_rescore)

    return p


def _normalize_legacy_positional(args: argparse.Namespace) -> None:
    """Allow `bench.py extract path/to/file` (legacy) to set --file path."""
    pos = getattr(args, "file_pos", None)
    if pos and not getattr(args, "file", None) and not getattr(args, "config", None):
        args.file = pos


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _normalize_legacy_positional(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
