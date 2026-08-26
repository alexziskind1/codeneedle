# Positional Recall Benchmark

Reproduces the benchmark from the YouTube video (see `benchmark_plan.md`):
stuff a large source corpus into an LLM's context, then ask it to reproduce
the first N lines of specific named functions verbatim. Measures positional
recall under long context, not just named-entity lookup.

[video walkthrough](https://youtu.be/zBYfzecY5ww)

> ### Scoring changed since the video
>
> If you follow the video and get different numbers, that's expected — and the
> repo's are the corrected ones. Issues reported by viewers turned up three
> defects, all since fixed. **Scores from before these changes are not
> comparable with scores after them.**
>
> | What was wrong | Effect on the old numbers |
> |---|---|
> | Blank lines counted as recalled ([#4]) | 21.9% of all "matched" credit was whitespace. 26 function-scores were built *entirely* from blank lines. |
> | Re-indented lines counted as hallucinations ([#4]) | 21% of reported hallucinations were correct content, indented differently. For several models it was nearly all of them. |
> | The JS prompt anchored on text that isn't there ([#4]) | 5 of 16 jQuery targets were asked about via `function <name>(`, which doesn't exist for `val: function( value ) {`. Those five averaged 59% vs 79% for the rest — in every model tested. |
>
> The third is the one worth understanding: the benchmark was **measuring a
> defect in its own question**, and charging it to the model. Depth in the file
> doesn't explain the gap (mean start line 5,261 vs 5,281), and it isn't one
> bad target (15-point gap remains with the worst one removed).
>
> Two smaller corrections came with them: targets whose name *and* signature are
> duplicated are now excluded, because no prompt can identify them; and the pass
> threshold is a ratio rather than a fixed 8, since blank lines no longer inflate
> the denominator. On a 20-line all-code window that is exactly the original 8/20.
>
> Thanks to [@AliakseiLasevich], [@tleschinski], [@tcclaviger], [@akierum],
> [@mazar] and [@QuantForgeSoftware] for the reports that surfaced these.

[#4]: https://github.com/alexziskind1/codeneedle/issues/4
[@AliakseiLasevich]: https://github.com/AliakseiLasevich
[@tleschinski]: https://github.com/tleschinski
[@tcclaviger]: https://github.com/tcclaviger
[@akierum]: https://github.com/akierum
[@mazar]: https://github.com/mazar
[@QuantForgeSoftware]: https://github.com/QuantForgeSoftware

## Install

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management.

```
# Create a venv in .venv/ and install the deps from requirements.txt
uv venv
uv pip install -r requirements.txt
```

### Tests

```
uv pip install -r requirements-dev.txt
uv run pytest              # 106 tests, ~9s, no model or network needed
```

The suite pins the extractor against the real fixtures (every body line must
equal the actual source line at that number), covers the scoring policy,
multi-file corpora, chart generation, and runs the full pipeline against an
in-process mock server. `smoke_test.py` is retained as a dependency-free
alternative.

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

## Docker (Optional)

Start interactive bash session with all dependencies already pre-installed

```sh
docker compose run --rm app
```

Now you can use either `uv run` or `python` directly

Close interactive shell by pressing `CTRL-d` or typing `exit` plus `RETURN`

> **macOS/Windows note:** the compose file uses `network_mode: host` so the
> container can reach a local LM Studio server at `localhost:1234`. On Docker
> Desktop this must be enabled first (Settings → Resources → Network →
> "Enable host networking"), or drop that line and pass
> `--base-url http://host.docker.internal:1234` to your runs.

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
tests/            pytest suite (see "Tests" above)
analysis/
  visualize.py    Plotly dashboard builder
  charts/         generated HTML output (gitignored)
  VIZ_README.md   chart-by-chart explanation + how to extend
.secrets/         API keys for hosted endpoints (gitignored, perms 700)
bench/            package internals
bench.py          CLI entry
```

## Configs

The split is by axis-of-change. You rarely change which files to test, but
you constantly compare different models — so an N×M comparison needs only
N+M files, not N*M.

### Hosted models — API keys

Don't put real keys in committed config files. The recommended workflow:

```bash
mkdir -p .secrets && chmod 700 .secrets
echo 'sk-...' > .secrets/openai.key
chmod 600 .secrets/openai.key
```

Then reference it from a model config:

```toml
# configs/models/gpt-5.5.toml
name              = "gpt-5.5"
base_url          = "https://api.openai.com"
api_key_file      = ".secrets/openai.key"   # path resolved from repo root
temperature       = 1.0
max_tokens        = 8000
reasoning_effort  = "none"
use_max_completion_tokens = true
```

`.secrets/` and any `*.key` file are already in `.gitignore`. Verify with
`git check-ignore -v .secrets/openai.key` — you should see a match.

Alternatives: `api_key_env = "OPENAI_API_KEY"` (read from environment), or
`api_key = "..."` (literal — only for non-secret tokens like LM Studio's
`"not-needed"` placeholder).

Full hosted-model details and known per-API quirks:
[`configs/CONFIG_README.md → Hosted models`](configs/CONFIG_README.md#hosted-models--api-keys-and-security).

> Field-by-field reference for every TOML key, plus recipes for adding a new
> corpus or model, lives in [`configs/CONFIG_README.md`](configs/CONFIG_README.md).

### Corpora — `configs/corpora/<name>.toml`

```toml
[files]
directory = "fixtures"   # required — relative to repo root, or absolute
glob      = "*.js"       # required
limit     = 1            # optional cap on matched files (sorted lexically)

[sample]
k              = 16      # number of functions to test
seed           = 42
min_code_lines = 0       # optional: skip prose-dominated targets

[scoring]
count_comments = true    # comments/docstrings earn credit (blank lines never do)
```

Shipped:
- `http_server` — single ~50KB Python file, fits any context, fast iteration
- `jquery` — ~280KB / ~80K-token JS, closest to the video's setup (needs ≥100K loaded context)

If `glob` matches multiple files, they're concatenated with comment-marker
headers (`# ====== path ======` / `// ====== path ======`) so the model sees
file boundaries. Cross-file name collisions are deduplicated (first occurrence
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

Shipped (see `configs/models/` for the full set):
- `qwen3-4b-2507` — small, honors `/no_think`, low `max_tokens` is fine
- `qwen36-35b` — reasoning-on-by-default; ignores `/no_think` and
  `reasoning_effort`, but `prefill_no_think = true` skips CoT reliably, so
  `max_tokens=1500` is plenty
- `gemma-4-31b-4bit` / `-bf16` — non-reasoning; needs `stop` sequences (parrots
  the prompt back) and `relax_indent = true` (normalizes leading whitespace)
- `qwen36-27b-mlx-4bit` / `-8bit` — same weights, same runtime, quant is the
  only difference: the controlled comparison for "does quantization hurt
  recall". Their LM Studio ids (`qwen3.6-27b` / `qwen3.6-27b-mlx`) misleadingly
  imply GGUF-vs-MLX, so both configs set an explicit `label` for charts.
- `gpt-5.5`, `claude-sonnet-4-6` — hosted; keys read from `.secrets/`

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
python3 bench.py run --corpus jquery --model qwen3-4b-2507
python3 bench.py run --corpus jquery --model qwen36-35b

# Override anything from the CLI
python3 bench.py run --corpus jquery --model qwen36-35b -k 8 --max-tokens 8000

# Test only specific functions (skips sampling)
python3 bench.py run --corpus http_server --model qwen36-35b \
    --function is_cgi --function translate_path

# Use a raw model identifier (no config file needed)
python3 bench.py run --corpus http_server --model "qwen/qwen3-4b-2507"

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
# -> analysis/charts/index.html + analysis/charts/<corpus>/<chart>.html
# (see analysis/VIZ_README.md for what each chart shows)
```

Supported source languages: `.js`, `.mjs`, `.cjs` (esprima), `.py` (`ast`).

## Reading the output

Per-function diff uses colors matching the video:

- **gray**       — matched line (expected + produced at correct position)
- **orange**     — expected but missing from the output
- **yellow**     — hallucinated / mangled line
- **blue/cyan**  — extra correct lines past the primary 20 (bonus)
- **dim**        — correct, but not eligible for credit (see below)

Pass threshold per function: **≥ 40% of the scored lines matched**. On a
20-line all-code window that's exactly the video's 8-of-20.

### What counts toward a score

A 20-line window is rarely 20 lines of code. In the shipped corpora:

| corpus | code | blank | comment | docstring |
|---|---:|---:|---:|---:|
| `http_server` | 40% | 17% | 5% | 38% |
| `jquery`      | 63% | 19% | 18% | — |

So the policy matters:

- **Blank lines never earn credit.** Reproducing whitespace demonstrates no
  recall, and blanks were ~18% of every window. They're excluded from both the
  numerator and the denominator. They still take part in the alignment, so a
  model that puts them in the right places stays in positional sync — only the
  accounting changes.
- **Comments and docstrings count by default.** Verbatim prose can't be
  inferred from surrounding code, so retrieving it is genuine recall. But it
  isn't *code* recall, so every result reports the split:
  ```
  === guess_type  [PASS]  matched=12/16 (75%)  hallucinated=0  bonus=0 ===
    composition: code 6/8 · prose 6/8 · 4 blank skipped
  ```
  Pass `--no-comments` to score code only. `http_server.log_message` has just
  **one** code line in its window — under the default policy it can pass on
  docstring recall alone, which the `code=0/1` column makes obvious.
- **Prose-dominated targets are flagged**, and `--min-code-lines N` (or
  `[sample] min_code_lines` in a corpus config) drops them from the sample
  entirely. Default is 0 — no filtering — so existing results stay reproducible.

Re-score any past run under a different policy without re-querying:

```
python3 bench.py rescore results/http_server__gpt-5.5.json --corpus http_server --no-comments
```

### Incomplete runs

A run that fail-fasts records `"complete": false` plus the query counts. Charts
label it **⚠ INCOMPLETE** and outline the bar in red, and `run-missing.py`
re-runs it instead of treating the file's existence as success. The leaderboard
plots **percentages**, not raw line counts, so a run with a smaller denominator
isn't misread as a worse model.

### Crash safety

The results dump is rewritten after **every** query, atomically, so a crash,
an OOM kill or a Ctrl-C costs you only the query in flight rather than the
whole run — which on an 80K-token corpus can be half an hour of inference. A
dump written mid-run carries `"in_progress": true`; `run-missing.py` re-runs
it and the charts flag it, so a partial run is never mistaken for a result.

### Run provenance

Each dump records the full request shape (model, temperature, token budget,
reasoning knobs, stop sequences), the sampling parameters (`k`, seed, filters),
the scoring policy, and a hash of the exact corpus text. Server-side settings
the API can't report — KV-cache quantization, loaded context length, quant
build — should be recorded by hand:

```
python3 bench.py run --corpus jquery --model qwen36-35b \
    --notes "LM Studio 0.3.x, Q8 KV cache, 131072 ctx, Unsloth Q4_K_XL"
```

### How a target is identified in the prompt

The prompt quotes the target's **own signature text**, copied verbatim from the
corpus:

```
Task: reproduce verbatim the first 20 lines of the body of the function named `val` ...

It is the function introduced by exactly this text:

    	val: function( value ) {

Output the 20 lines that come immediately after it.
```

Earlier versions instead told the model to look for `function <name>(`. That
text does not exist for property- or assignment-style definitions — 5 of the 16
sampled jQuery targets (`PSEUDO`, `init`, `then`, `val`, `parseHTML`) are
written as `val: function( value ) {` or `jQuery.parseHTML = function(...)`, so
the model was pointed at a string the file never contains. Across the stored
runs those five averaged **59%** against **79%** for the rest — a gap present in
every model tested, and not explained by their depth in the file. The benchmark
was measuring a defect in its own question.

**Unanswerable targets are excluded.** If a signature is not unique in the file,
no prompt could single that definition out, so the target is dropped from
sampling and the exclusion is printed:

```
excluded 4 unanswerable target(s) — duplicate name AND identical signature: ID, get, setup, sortOrder
```

jQuery declares 11 eligible names more than once; quoting the signature
disambiguates 10 of them, because the signatures differ even where the names do
not (`PSEUDO: function( match )` vs `PSEUDO: function( pseudo, argument )`).
Only the genuinely identical ones are dropped.

### Indentation: "hallucination" vs. re-indentation

A line the model reproduced correctly but indented differently is **not** a
hallucination. It is reported separately:

```
=== val  [PASS]  matched=19/20  hallucinated=0  reindented=1  bonus=0 ===
  -- reproduced, but re-indented (not hallucinations) --
                  var hooks, ret, valueIsFunction,
```

Scoring is unchanged: strict matching means *verbatim*, so a re-indented line
still counts as a miss and pass/fail verdicts stay comparable with earlier
runs. What changed is the label — such a line no longer inflates the
`hallucinated` count, and it is no longer penalized twice (once as a missing
expected line, once as a hallucinated emitted line).

If you care about content rather than exact whitespace, score by content:

```
python3 bench.py run --corpus jquery --model <model> --relax-indent
python3 bench.py rescore results/jquery__<model>.json --corpus jquery --relax-indent
```

or set `relax_indent = true` in the model config. The summary tells you when
this would make a difference. Models that normalize indentation (Gemma 4, for
one) can otherwise look far worse than they are.

## Server setup notes

For fair comparison matching the video:

- **llama.cpp**: `--ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0`,
  prompt caching on (default in recent builds).
- **LM Studio**: set context length to cover the file, enable "KV cache quantization"
  → Q8. Prefix cache is automatic.
- **Ollama**: set `num_ctx` via Modelfile or per-request; no KV quant yet, so
  comparison isn't apples-to-apples.

Keep temperature at 0. Default `max_tokens=6000` to leave room for reasoning models.

### Colour output

Colour is on for terminals and off when output is piped or redirected, so
anything you copy out of a log is plain text. Override either way:

```
python3 bench.py --no-color run --corpus http_server --model <model>   # never colour
python3 bench.py --color rescore results/<run>.json --corpus jquery    # force colour
```

`NO_COLOR=1` and `FORCE_COLOR=1` are honoured too ([no-color.org](https://no-color.org)).
On Windows, ANSI processing is enabled for the console when possible; if it
cannot be enabled, colour is disabled rather than printing escape codes as
literal text (`←[32m✓←[0m`).

### Windows / encoding

All files are read and written as UTF-8 explicitly, and `stdout`/`stderr` are
reconfigured to UTF-8 at startup. Without that, Python falls back to the locale
encoding — cp1252 on most Windows installs — which can't represent the `←` in
the chart navigation, the `≥`/`⚠`/`✓` in the console report, or the CJK
characters inside the bundled `plotly.min.js`. Symptoms this prevents:

```
UnicodeEncodeError: 'charmap' codec can't encode character '←'
```

Piping output to a file (`python bench.py run … > run.log`) is covered too —
that's the case where Windows drops to the locale encoding even though an
interactive console would have coped.

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
3. **Reasoning models** (qwen3.5, qwen3.6, …) mostly ignore `/no_think` and
   `reasoning_effort`, but **`prefill_no_think = true` does work** — it seeds an
   empty `<think></think>` assistant turn so the model continues past it. That's
   why the shipped qwen configs run at `max_tokens=1500` rather than 6000. See
   the full matrix in [`configs/CONFIG_README.md`](configs/CONFIG_README.md).
   If a model honors none of the three, give it budget for the chain-of-thought
   *plus* the answer — bump to 8000+ if responses come back empty.

### Hosted endpoints with rate limits

The client automatically retries `HTTP 429` (rate-limit) responses up to 3
times. It honors a server-supplied `Retry-After` header (in seconds), and
falls back to exponential backoff anchored at 60 seconds when the header is
missing — so a `bench.py run` against a Tier-1 OpenAI / Anthropic Build-tier
account no longer aborts after the first ratelimit. You'll see lines like:

```
⏸ HTTP 429 — sleeping 6.0s before retry 1/3
```

stream past while a long benchmark paces itself. After 3 retries the
underlying `RuntimeError` surfaces, so a structurally-too-large request
(e.g. a single Anthropic call exceeding the per-minute input-tokens limit)
still fails fast rather than looping forever.

Other 4xx/5xx errors (auth, bad request, server crash) are **not** retried —
they raise immediately, same as before the patch.

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
- `client_smoke_test.py` — client-only smoke test (mocked HTTP) for the 429-retry path
