"""Orchestrate a full benchmark run: extract targets, query the model, score, report."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .client import ClientConfig, chat_complete
from .extract import Source, extract, load_source_glob, stratified_sample
from .report import render_function, render_summary
from .scorer import FunctionScore, score


# Keeping the file FIRST and the tiny task suffix LAST is deliberate:
# llama.cpp / LM Studio / Ollama all reuse the KV cache for common prefix tokens,
# so across the 16 queries only the tail re-processes. Move the file and the
# cache is invalidated every request.
PROMPT_TEMPLATE = (
    "{file_contents}\n"
    "\n"
    "---\n"
    "\n"
    "Task: reproduce verbatim the first {n} lines of the body of the function named "
    "`{name}`{file_qualifier} from the source above — i.e., the {n} lines {anchor_phrase}.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY those lines, one per line, in original order.\n"
    "- Preserve original indentation and characters exactly.\n"
    "- Do NOT output the function signature or the line containing `{signature_marker}`.\n"
    "- Do NOT add commentary, line numbers, or markdown code fences.\n"
    "- If there are blank lines in the body, include them as blank lines.\n"
    "{thinking_suffix}"
)
# Per-language anchor phrasing — the source has no opening brace in Python,
# so saying "following the opening brace" confuses the model and produces
# off-by-N-line drift (emits the signature line, emits class-attr lines before
# the def, etc.). Pin the anchor to a marker the language actually has.
ANCHOR_PHRASE = {
    "js": "starting immediately after the line containing `function {name}(` "
          "or the assignment that introduces it (the line with the opening "
          "brace `{{`)",
    "py": "starting with the first body line after the `def {name}(...):` "
          "signature (including the docstring if present)",
}
SIGNATURE_MARKER = {
    "js": "function {name}(",
    "py": "def {name}(",
}
# Qwen3 (and other reasoning-enabled models) treat `/no_think` as a directive
# to skip chain-of-thought. Ignored by non-reasoning models. For a pure recall
# benchmark, reasoning wastes tokens and risks drift — so suppress by default.
NO_THINK_SUFFIX = "\n/no_think\n"


@dataclass
class _Run:
    function: str
    source_path: str | None
    prompt_chars: int
    response: str
    latency_s: float
    error: str | None = None


def _build_prompt(target, text: str, multi_file: bool, suppress_thinking: bool) -> str:
    anchor = ANCHOR_PHRASE[target.language].format(name=target.name)
    sig_marker = SIGNATURE_MARKER[target.language].format(name=target.name)
    file_qualifier = (
        f" in file `{target.source_path}`" if multi_file and target.source_path else ""
    )
    return PROMPT_TEMPLATE.format(
        file_contents=text,
        name=target.name,
        file_qualifier=file_qualifier,
        n=len(target.primary_lines),
        anchor_phrase=anchor,
        signature_marker=sig_marker,
        thinking_suffix=NO_THINK_SUFFIX if suppress_thinking else "",
    )


def _preflight_context_check(prompt: str, cfg: ClientConfig) -> str | None:
    """Send the actual prompt with max_tokens=1 to detect context-too-small.

    Returns None on success, an error message string otherwise. Cheap because
    no real generation happens — the model only ingests the prompt and emits
    a single token. As a side benefit it warms the server's prefix KV cache
    for the rest of the run.
    """
    probe_cfg = ClientConfig(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        temperature=0.0,
        max_tokens=1,
        timeout=cfg.timeout,
    )
    try:
        chat_complete(probe_cfg, system=None, user=prompt)
        return None
    except Exception as e:
        return str(e)


def _is_context_error(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in ("context length", "n_ctx", "n_keep", "too long", "exceeds"))


def run_benchmark(
    source: Source,
    cfg: ClientConfig,
    k: int = 16,
    seed: int = 42,
    dump_path: Path | None = None,
    function_filter: list[str] | None = None,
    suppress_thinking: bool = True,
    skip_preflight: bool = False,
    fail_fast_after: int | None = 2,
) -> list[FunctionScore]:
    text = source.text
    total_lines = text.count("\n") + 1
    print(
        f"Source: {source.display_name}  ({len(text):,} chars, {total_lines:,} lines, "
        f"{len(source.files)} file(s))",
        flush=True,
    )
    print(
        f"Extracted {len(source.targets)} named functions with ≥20 body lines",
        flush=True,
    )

    if function_filter:
        wanted = {n for n in function_filter}
        chosen = [t for t in source.targets if t.name in wanted]
        missing = wanted - {t.name for t in chosen}
        if missing:
            print(f"WARNING: requested but not found: {sorted(missing)}", flush=True)
    else:
        chosen = stratified_sample(source.targets, total_lines, k=k, seed=seed)

    print(f"Selected {len(chosen)} target function(s):", flush=True)
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(
            f"  - {t.name}  line {t.start_line}  body_lines={len(t.body_lines)}{loc}",
            flush=True,
        )

    multi_file = len(source.files) > 1

    # Pre-flight: send the first real prompt with max_tokens=1 to check that
    # the loaded context is big enough. Misleading FAILs from context-too-small
    # are the easiest mistake to make with LM Studio (TTL-driven JIT reload at
    # default 4K context). Better to abort up front.
    if not skip_preflight:
        probe_prompt = _build_prompt(chosen[0], text, multi_file, suppress_thinking)
        print(
            f"\nPre-flight: probing context fit with a {len(probe_prompt):,}-char prompt "
            f"(max_tokens=1)...",
            flush=True,
        )
        err = _preflight_context_check(probe_prompt, cfg)
        if err is None:
            print("Pre-flight OK.", flush=True)
        elif _is_context_error(err):
            print(f"\n❌ pre-flight failed (context too small):\n   {err}\n", flush=True)
            print("The loaded model context is smaller than the prompt. The most common", flush=True)
            print("cause is LM Studio JIT-reloading at default 4K context after its TTL", flush=True)
            print("expired. Force-reload at the size you need:", flush=True)
            print(f"\n   lms unload {cfg.model}", flush=True)
            print(f"   lms load {cfg.model} --context-length 131072 --gpu max -y\n", flush=True)
            print("Re-run after the model is loaded. (Pass --skip-preflight to override.)", flush=True)
            raise SystemExit(2)
        else:
            print(f"\n❌ pre-flight failed: {err}\n", flush=True)
            print("The server is reachable but rejected the request for a non-context reason.", flush=True)
            print("Fix the server-side error or pass --skip-preflight to push past this check.", flush=True)
            raise SystemExit(2)

    scores: list[FunctionScore] = []
    runs: list[_Run] = []
    consecutive_errors = 0
    for i, t in enumerate(chosen, 1):
        prompt = _build_prompt(t, text, multi_file, suppress_thinking)
        print(
            f"\n[{i}/{len(chosen)}] `{t.name}` — prompt {len(prompt):,} chars, waiting on model...",
            flush=True,
        )
        start = time.monotonic()
        request_error: str | None = None
        try:
            resp = chat_complete(cfg, system=None, user=prompt)
        except Exception as e:
            request_error = str(e)
            print(f"  ERROR: {request_error}", flush=True)
            resp = ""
        latency = time.monotonic() - start
        print(f"  response: {len(resp)} chars in {latency:.1f}s", flush=True)

        # Empty content with no exception = HTTP 200 but the model produced
        # nothing. On reasoning models that's typically the CoT eating the
        # entire max_tokens budget. Treat as a non-recall error so it shows
        # as ERROR, not FAIL.
        score_error = request_error
        if resp.strip() == "" and score_error is None:
            score_error = (
                "empty response (200 OK but no content; reasoning models often need "
                "more max_tokens — try --max-tokens 8000)"
            )
            print(f"  ⚠ {score_error}", flush=True)

        sc = score(t.name, t.primary_lines, t.bonus_lines, resp)
        if score_error:
            sc.error = score_error
        scores.append(sc)
        runs.append(
            _Run(
                function=t.name,
                source_path=str(t.source_path) if t.source_path else None,
                prompt_chars=len(prompt),
                response=resp,
                latency_s=latency,
                error=score_error,
            )
        )
        print(render_function(sc), flush=True)

        # Fail-fast: if N queries in a row error, the rest will too. Bail.
        if score_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if (
            fail_fast_after is not None
            and consecutive_errors >= fail_fast_after
            and i < len(chosen)
        ):
            remaining = len(chosen) - i
            print(
                f"\n⚠ {consecutive_errors} consecutive ERROR results — aborting the "
                f"remaining {remaining} queries.",
                flush=True,
            )
            print(
                "  Same prompt size + same model + same params → same outcome. "
                "Likely fixes:",
                flush=True,
            )
            print(
                "    • Reasoning model burning the budget? bump --max-tokens (try 8000–12000)",
                flush=True,
            )
            print(
                "    • Server-side error? check logs and the per-query message above",
                flush=True,
            )
            print(
                "  Pass --no-fail-fast to disable this check and run every query anyway.",
                flush=True,
            )
            break

    print(render_summary(scores), flush=True)

    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "files": [str(p) for p in source.files],
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "results": [
                {
                    "function": sc.name,
                    "source_file": r.source_path,
                    "passed": sc.passed,
                    "error": sc.error,
                    "primary_matched": sc.primary_matched,
                    "primary_total": sc.primary_total,
                    "hallucinated": sc.hallucinated,
                    "bonus_matched": sc.bonus_matched,
                    "latency_s": r.latency_s,
                    "prompt_chars": r.prompt_chars,
                    "response": r.response,
                }
                for sc, r in zip(scores, runs)
            ],
        }
        dump_path.write_text(json.dumps(payload, indent=2))
        print(f"\nResults dumped to {dump_path}", flush=True)

    return scores


def source_from_single_file(path: Path) -> Source:
    """Convenience: build a Source from one file (for backwards-compat with the file CLI)."""
    targets = extract(path)
    text = path.read_text()
    from .extract import language_of
    return Source(files=[path], text=text, targets=targets, language=language_of(path))
