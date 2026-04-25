#!/usr/bin/env python3
"""Positional recall benchmark — CLI entry.

Tests an LLM's ability to reproduce the first N lines of a named function
inside a large source corpus loaded into context. Measures positional recall,
not just named-entity lookup.

Source selection (extract / run / rescore):
    --corpus NAME      a config under configs/corpora/, or a path to one
    --file PATH        single source file (.js/.mjs/.cjs/.py)

Model selection (run only):
    --model NAME       a config under configs/models/, OR a raw model identifier
                       (raw names get sane defaults; create a config for control)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# --- source resolution ---------------------------------------------------


def _resolve_source(args: argparse.Namespace):
    """Return (Source, CorpusConfig|None) from --corpus or --file."""
    from bench.extract import load_source_glob
    from bench.runner import source_from_single_file

    if getattr(args, "corpus", None):
        from bench.config import load_corpus

        corpus = load_corpus(args.corpus)
        src = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
        return src, corpus
    if getattr(args, "file", None):
        return source_from_single_file(Path(args.file)), None
    raise SystemExit("error: pass either --corpus NAME or --file PATH")


# --- extract -------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    from bench.extract import stratified_sample

    source, corpus = _resolve_source(args)

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
    k = args.k if args.k is not None else (corpus.sample_k if corpus else 16)
    seed = args.seed if args.seed is not None else (corpus.sample_seed if corpus else 42)
    if args.all:
        chosen = source.targets
    else:
        chosen = stratified_sample(source.targets, total_lines, k=k, seed=seed)
        print(f"stratified sample of {len(chosen)}:")
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(f"  {t.name:<40}  line={t.start_line:>6}  body_lines={len(t.body_lines)}{loc}")
    return 0


# --- run ------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from bench.config import auto_dump_path, load_model
    from bench.runner import run_benchmark

    source, corpus = _resolve_source(args)

    if not args.model:
        raise SystemExit("error: --model is required (a name in configs/models/, a path, or a raw model id)")
    model, model_from_file = load_model(args.model)
    if not model_from_file:
        print(
            f"  (no model config '{args.model}' found; using as raw model identifier with defaults)",
            file=sys.stderr,
        )

    # CLI overrides — applied on top of whichever source the model came from.
    if args.base_url:
        model.client.base_url = args.base_url
    if args.api_key:
        model.client.api_key = args.api_key
    if args.temperature is not None:
        model.client.temperature = args.temperature
    if args.max_tokens is not None:
        model.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model.client.timeout = args.timeout
    suppress_thinking = model.suppress_thinking and not args.think

    if corpus is not None:
        k = args.k if args.k is not None else corpus.sample_k
        seed = args.seed if args.seed is not None else corpus.sample_seed
    else:
        k = args.k if args.k is not None else 16
        seed = args.seed if args.seed is not None else 42

    if args.dump:
        dump_path = Path(args.dump)
    elif corpus is not None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(corpus, model, DEFAULT_RESULTS_DIR)
    else:
        # --file mode: derive corpus stem from filename
        from bench.config import CorpusConfig

        synthetic_corpus = CorpusConfig(
            name=Path(args.file).stem,
            directory=Path(args.file).parent,
            glob=Path(args.file).name,
            limit=1,
            sample_k=k,
            sample_seed=seed,
        )
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(synthetic_corpus, model, DEFAULT_RESULTS_DIR)

    fn_filter = args.function if args.function else None
    scores = run_benchmark(
        source=source,
        cfg=model.client,
        k=k,
        seed=seed,
        dump_path=dump_path,
        function_filter=fn_filter,
        suppress_thinking=suppress_thinking,
        skip_preflight=args.skip_preflight,
        fail_fast_after=None if args.no_fail_fast else args.fail_fast_after,
    )
    passed = sum(1 for s in scores if s.passed)
    return 0 if passed == len(scores) else 1


# --- rescore --------------------------------------------------------------


def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-score a previous run's dump without re-querying the model."""
    import json

    from bench.extract import load_source_glob
    from bench.report import render_function, render_summary
    from bench.runner import source_from_single_file
    from bench.scorer import score

    dump = json.loads(Path(args.dump).read_text())
    if args.corpus:
        from bench.config import load_corpus

        corpus = load_corpus(args.corpus)
        source = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
    elif args.file:
        source = source_from_single_file(Path(args.file))
    else:
        files = dump.get("files") or ([dump["source"]] if dump.get("source") else [])
        if len(files) == 1 and Path(files[0]).is_file():
            source = source_from_single_file(Path(files[0]))
        else:
            raise SystemExit(
                "error: dump references a missing or multi-file corpus; "
                "pass --corpus NAME or --file PATH to re-locate it"
            )

    targets = {t.name: t for t in source.targets}
    scores = []
    for r in dump["results"]:
        t = targets.get(r["function"])
        if t is None:
            print(f"skip: {r['function']} not found in source", file=sys.stderr)
            continue
        sc = score(t.name, t.primary_lines, t.bonus_lines, r.get("response", ""))
        if r.get("error"):
            sc.error = r["error"]
        scores.append(sc)
        print(render_function(sc))
    print(render_summary(scores))
    return 0


# --- argparse -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ------------------------------------------------------------
    p_ex = sub.add_parser("extract", help="list functions the extractor would test")
    src_grp = p_ex.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_ex.add_argument("-k", type=int, default=None, help="override corpus sample.k")
    p_ex.add_argument("--seed", type=int, default=None, help="override corpus sample.seed")
    p_ex.add_argument("--all", action="store_true", help="list every extracted function, not a sample")
    p_ex.add_argument("--show", metavar="NAME", help="print expected primary+bonus lines for one function")
    p_ex.set_defaults(func=cmd_extract)

    # --- run ----------------------------------------------------------------
    p_run = sub.add_parser("run", help="run the benchmark against an OpenAI-compatible endpoint")
    src_grp = p_run.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_run.add_argument(
        "--model", required=True,
        help="model config name (configs/models/<name>.toml), a path, or a raw model identifier",
    )
    p_run.add_argument("--base-url", default=None, help="overrides model config")
    p_run.add_argument("--api-key", default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("-k", type=int, default=None, help="overrides corpus.sample.k")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument(
        "--dump", default=None,
        help="JSON path for full results (default: results/<corpus>__<model>.json)",
    )
    p_run.add_argument("--function", action="append", help="repeatable; overrides sampling")
    p_run.add_argument("--think", action="store_true", help="allow chain-of-thought (default: suppress)")
    p_run.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the context-fit pre-flight probe (not recommended)",
    )
    p_run.add_argument(
        "--fail-fast-after", type=int, default=2, metavar="N",
        help="abort the run after N consecutive ERROR results (default: 2)",
    )
    p_run.add_argument(
        "--no-fail-fast", action="store_true",
        help="disable fail-fast; run every query even if they're all erroring",
    )
    p_run.set_defaults(func=cmd_run)

    # --- rescore ------------------------------------------------------------
    p_rs = sub.add_parser("rescore", help="re-score a previous --dump without re-querying")
    p_rs.add_argument("dump", help="path to JSON dump from a prior `run`")
    src_grp = p_rs.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="re-locate corpus via this config")
    src_grp.add_argument("--file", help="re-locate corpus from a single file")
    p_rs.set_defaults(func=cmd_rescore)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
