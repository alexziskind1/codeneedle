# Positional Recall Benchmark

Reproduces the benchmark from the YouTube video (see `benchmark_plan.md`):
stuff a large source corpus into an LLM's context, then ask it to reproduce
the first N lines of specific named functions verbatim. Measures positional
recall under long context, not just named-entity lookup.

## Install

```
pip install -r requirements.txt
```

## Quick start

```
# 1. (LM Studio only) make sure your model is loaded with enough context.
#    Defaults can silently sit at 4K. Force-reload at 128K:
lms unload qwen3.6-35b-a3b
lms load qwen3.6-35b-a3b --context-length 131072 --gpu max -y

# 2. Pick a config and run it:
python3 bench.py run --config configs/http_server.toml

# 3. Result is auto-saved as results/<config>__<model>.json.
```

## Layout

```
configs/    TOML configs that bundle corpus + sample + model settings
fixtures/   source files to test against (jquery.js, http_server.py, …)
results/    JSON dumps from every run, auto-named <config>__<model>.json
bench/      package internals
bench.py    CLI entry
```

## Configs

A config selects a corpus via glob, fixes the sample, and pins model settings.
The two examples shipped:

- `configs/http_server.toml` — single ~50KB Python file. Fits any context, fast.
- `configs/jquery.toml` — ~280KB / ~80K-token JS. Closest to the video's setup.
  Needs ≥100K loaded context.

Schema:

```toml
[files]
directory = "fixtures"   # required
glob      = "*.js"       # required
limit     = 1            # optional cap on matched files (sorted lexically)

[sample]
k    = 16                # number of functions to test
seed = 42

[model]
name              = "qwen3.6-35b-a3b"      # required (model name as the server knows it)
base_url          = "http://localhost:1234"
api_key           = "not-needed"           # optional
temperature       = 0.0
max_tokens        = 6000                   # leave room for reasoning models
timeout           = 600.0
suppress_thinking = true                   # appends /no_think (harmless when ignored)
```

If `glob` matches multiple files, they're concatenated with comment-marker
headers (`# === path ===` / `// === path ===`) so the model can see boundaries.
Function-name collisions across files are deduplicated (first occurrence wins),
and the prompt qualifies by file path when more than one file is in play.

## Commands

```
# Run the benchmark
python3 bench.py run --config configs/http_server.toml

# Override anything from the CLI
python3 bench.py run --config configs/jquery.toml --model qwen/qwen3-4b -k 8 --max-tokens 8000

# Test only specific functions (skips sampling)
python3 bench.py run --config configs/http_server.toml \
    --function is_cgi --function translate_path

# Single-file mode (no config)
python3 bench.py run --file fixtures/http_server.py \
    --model qwen3.6-35b-a3b --base-url http://localhost:1234

# See what would be tested
python3 bench.py extract --config configs/http_server.toml          # sampled
python3 bench.py extract --config configs/http_server.toml --all    # all extractable
python3 bench.py extract --config configs/http_server.toml --show is_cgi   # ground truth

# Re-score a prior dump without re-querying
python3 bench.py rescore results/http_server__qwen3.6-35b-a3b.json
```

Supported source languages: `.js`, `.mjs`, `.cjs` (esprima), `.py` (`ast`).

## Reading the output

Per-function diff uses colors matching the video:

- **gray**       — matched line (expected + produced at correct position)
- **orange**     — expected but missing from the output
- **yellow**     — hallucinated / mangled line
- **blue/cyan**  — extra correct lines past the primary 20 (bonus)

Pass threshold per function: ≥ 8 of the 20 expected lines matched.

## Server setup notes

For fair comparison matching the video:

- **llama.cpp**: `--ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0`,
  prompt caching on (default in recent builds).
- **LM Studio**: set context length to cover the file, enable "KV cache quantization"
  → Q8. Prefix cache is automatic.
- **Ollama**: set `num_ctx` via Modelfile or per-request; no KV quant yet, so
  comparison isn't apples-to-apples.

Keep temperature at 0. Default `max_tokens=6000` to leave room for reasoning models.

### LM Studio gotchas we hit (read before debugging)

1. **`lms ps` lies about context size after JIT loads.** If large prompts fail
   with a 400 "context length" error despite `lms ps` showing a big number,
   force-reload:
   ```
   lms unload <model>
   lms load <model> --context-length 131072 --gpu max -y
   ```
2. **Auto-unload by idle TTL** (default ~60 min). After it expires, the next
   request triggers a JIT reload at *default settings*, silently dropping your
   large context. Either disable TTL in the LM Studio UI or re-load before
   each session.
3. **Reasoning models** (qwen3.5, qwen3.6, …) do not honor `/no_think`,
   `enable_thinking: false`, `reasoning_effort: "none"`, or any other API toggle
   we tested. The benchmark still appends `/no_think` (harmless if ignored), but
   you must give the budget for chain-of-thought *plus* the answer. Default
   `max_tokens=6000`; bump to 8000+ if responses come back empty.

## Module map

- `benchmark_plan.md` — analysis of what the benchmark measures and why
- `bench.py` — CLI entry
- `bench/config.py` — TOML config loader
- `bench/extract.py` — function extraction + multi-file source aggregation
- `bench/client.py` — tiny OpenAI-compatible client
- `bench/scorer.py` — LCS alignment, line classification, pass/fail
- `bench/report.py` — ANSI color rendering
- `bench/runner.py` — orchestration: prompt assembly, query, score, dump
- `smoke_test.py` — end-to-end sanity check without an LLM
