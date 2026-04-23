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


def run_benchmark(
    source: Source,
    cfg: ClientConfig,
    k: int = 16,
    seed: int = 42,
    dump_path: Path | None = None,
    function_filter: list[str] | None = None,
    suppress_thinking: bool = True,
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
    scores: list[FunctionScore] = []
    runs: list[_Run] = []
    for i, t in enumerate(chosen, 1):
        anchor = ANCHOR_PHRASE[t.language].format(name=t.name)
        sig_marker = SIGNATURE_MARKER[t.language].format(name=t.name)
        file_qualifier = (
            f" in file `{t.source_path}`" if multi_file and t.source_path else ""
        )
        prompt = PROMPT_TEMPLATE.format(
            file_contents=text,
            name=t.name,
            file_qualifier=file_qualifier,
            n=len(t.primary_lines),
            anchor_phrase=anchor,
            signature_marker=sig_marker,
            thinking_suffix=NO_THINK_SUFFIX if suppress_thinking else "",
        )
        print(
            f"\n[{i}/{len(chosen)}] `{t.name}` — prompt {len(prompt):,} chars, waiting on model...",
            flush=True,
        )
        start = time.monotonic()
        try:
            resp = chat_complete(cfg, system=None, user=prompt)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            resp = ""
        latency = time.monotonic() - start
        print(f"  response: {len(resp)} chars in {latency:.1f}s", flush=True)
        if resp.strip() == "":
            print(
                "  ⚠ empty response — likely causes: (1) reasoning model burned the "
                "entire max_tokens budget on chain-of-thought (try --max-tokens 8000); "
                "(2) loaded context size is smaller than the prompt (check `lms ps` and "
                "force-reload with --context-length).",
                flush=True,
            )

        sc = score(t.name, t.primary_lines, t.bonus_lines, resp)
        scores.append(sc)
        runs.append(
            _Run(
                function=t.name,
                source_path=str(t.source_path) if t.source_path else None,
                prompt_chars=len(prompt),
                response=resp,
                latency_s=latency,
            )
        )
        print(render_function(sc), flush=True)

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
