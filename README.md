# Positional Recall Benchmark

Reproduces the benchmark from the YouTube video (see `benchmark_plan.md`):
stuff a large source corpus into an LLM's context, then ask it to reproduce
the first N lines of specific named functions verbatim. Measures positional
recall under long context, not just named-entity lookup.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management.

```
# Create a venv in .venv/ and install the deps from requirements.txt
uv venv
uv pip install -r requirements.txt
```

Run any project script via `uv run` (no `source .venv/bin/activate` needed):

```
uv run python bench.py run --corpus http_server --model qwen36-35b
uv run python analysis/visualize.py
uv run python smoke_test.py
```

If you'd rather activate the venv:

```
source .venv/bin/activate
python3 bench.py run --corpus http_server --model qwen36-35b
```

The rest of this README writes commands as `python3 …` for brevity — prepend
`uv run ` if your venv isn't active.

## Quick start

```
# 1. (LM Studio only) make sure your model is loaded with enough context.
#    Defaults can silently sit at 4K. Force-reload at 128K:
lms unload qwen3.6-35b-a3b
lms load qwen3.6-35b-a3b --context-length 131072 --gpu max -y

# 2. Pick a corpus + a model and run them (assumes .venv is active; otherwise prepend `uv run`):
python3 bench.py run --corpus http_server --model qwen36-35b

# 3. Result is auto-saved as results/<corpus>__<model>.json.
```

## Layout

```
configs/
  corpora/        what files to test, sample size — one TOML per corpus
  models/         model identifier and per-model knobs — one TOML per model
fixtures/         source files to test against (jquery.js, http_server.py, …)
results/          JSON dumps from every run, auto-named <corpus>__<model>.json
analysis/
  visualize.py    Plotly dashboard builder
  charts/         generated HTML output (gitignored)
  VIZ_README.md   chart-by-chart explanation + how to extend
bench/            package internals
bench.py          CLI entry
```

## Configs

The split is by axis-of-change. You rarely change which files to test, but
you constantly compare different models — so an N×M comparison needs only
N+M files, not N*M.

> Field-by-field reference for every TOML key, plus recipes for adding a new
> corpus or model, lives in [`configs/CONFIG_README.md`](configs/CONFIG_README.md).

### Corpora — `configs/corpora/<name>.toml`

```toml
[files]
directory = "fixtures"   # required
glob      = "*.js"       # required
limit     = 1            # optional cap on matched files (sorted lexically)

[sample]
k    = 16                # number of functions to test
seed = 42
```

Shipped:
- `http_server` — single ~50KB Python file, fits any context, fast iteration
- `jquery` — ~280KB / ~80K-token JS, closest to the video's setup (needs ≥100K loaded context)

If `glob` matches multiple files, they're concatenated with comment-marker
headers (`# === path ===` / `// === path ===`) so the model sees file
boundaries. Cross-file name collisions are deduplicated (first occurrence
wins), and the prompt qualifies by file path when more than one file is in play.

### Models — `configs/models/<name>.toml`

```toml
name              = "qwen3.6-35b-a3b"      # required (model id the server knows)
base_url          = "http://localhost:1234"
api_key           = "not-needed"           # optional
temperature       = 0.0
max_tokens        = 6000                   # leave room for reasoning models
timeout           = 600.0
suppress_thinking = true                   # appends /no_think (harmless when ignored)
```

Shipped:
- `qwen3-4b` — small, honors `/no_think`, `max_tokens=1500` is fine
- `qwen36-35b` — reasoning-on-by-default, ignores every thinking-disable knob; needs `max_tokens=6000`

If you pass `--model FOO` and there's no matching config file, FOO is treated
as a raw model identifier with sane defaults — so you don't *have* to write a
config to do a one-off run, but for repeated use it's worth pinning the knobs.

### How the two configs combine at run time

Every `run` invocation needs **one corpus** (`--corpus NAME` or `--file PATH`)
and **one model** (`--model NAME`). They're resolved independently and stitched
together — there is no shared parent file or inheritance.

**Resolution order**, for both flags:
1. If the value points to an existing file on disk, load it.
2. Otherwise look it up by name under `configs/corpora/<name>.toml` or
   `configs/models/<name>.toml`.
3. (`--model` only) If still not found, treat the value as a raw model
   identifier and use built-in defaults. A note is printed so you know the
   fallback was taken.

**Override layering**, applied in order (later wins):
1. defaults baked into the loader (`max_tokens=6000`, `temperature=0`, …)
2. fields set in the **model config** file
3. CLI overrides — `--base-url`, `--max-tokens`, `--temperature`, `--timeout`,
   `--api-key`
4. sampling overrides (`-k`, `--seed`) layer over the **corpus config**'s
   `[sample]` the same way

This means model knobs can come from anywhere on the chain. A typical config
sets the model-specific defaults (e.g. `max_tokens=6000` for a reasoning model)
and you override per-run knobs (`--max-tokens 8000` for a hard case) without
editing the file.

**`--think`** flips one bit: it inverts `suppress_thinking` so chain-of-thought
is left on. Useful when you specifically want to compare reasoning vs.
no-reasoning recall on a model that supports both.

**Output filename** is `results/<corpus.name>__<model.name>.json`, where each
`name` is the **config stem** (filename without `.toml`). Raw-model fallback
sanitizes the identifier (`/` → `_`). Override the whole path with `--dump`.

Mental model: corpus = *what to ask*, model = *who to ask and how*. Keep them
orthogonal.

## Commands

```
# Run a benchmark
python3 bench.py run --corpus http_server --model qwen36-35b

# Compare models on the same corpus
python3 bench.py run --corpus jquery --model qwen3-4b
python3 bench.py run --corpus jquery --model qwen36-35b

# Override anything from the CLI
python3 bench.py run --corpus jquery --model qwen36-35b -k 8 --max-tokens 8000

# Test only specific functions (skips sampling)
python3 bench.py run --corpus http_server --model qwen36-35b \
    --function is_cgi --function translate_path

# Use a raw model identifier (no config file needed)
python3 bench.py run --corpus http_server --model "qwen/qwen3-4b"

# Single-file mode (no corpus config)
python3 bench.py run --file fixtures/http_server.py --model qwen36-35b

# See what would be tested
python3 bench.py extract --corpus http_server          # sampled
python3 bench.py extract --corpus http_server --all    # every extractable function
python3 bench.py extract --corpus http_server --show is_cgi   # ground truth

# Re-score a prior dump without re-querying
python3 bench.py rescore results/http_server__qwen36-35b.json

# Build Plotly dashboards comparing every run in results/
python3 analysis/visualize.py
# -> analysis/charts/<corpus>.html + analysis/charts/index.html
# (see analysis/VIZ_README.md for what each chart shows)
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
- `analysis/visualize.py` — builds Plotly HTML dashboards from `results/*.json`
  (see [`analysis/VIZ_README.md`](analysis/VIZ_README.md) for chart-by-chart details)
- `smoke_test.py` — end-to-end sanity check without an LLM
