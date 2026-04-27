# Configs

Two kinds of TOML files live here, split along stability axis:

```
configs/
  corpora/   what files to test, sample size — one per corpus
  models/    model identifier and per-model knobs — one per model
```

A run combines exactly one corpus with exactly one model:

```
python3 bench.py run --corpus <corpus-name> --model <model-name>
```

Both args resolve by **filename stem** (no `.toml`). E.g. `--corpus jquery`
loads `configs/corpora/jquery.toml`. Either also accepts an explicit path.

The output filename is `results/<corpus-stem>__<model-stem>.json`.

---

## Corpus configs — `configs/corpora/<name>.toml`

```toml
[files]
directory = "fixtures"   # required
glob      = "*.js"       # required
limit     = 1            # optional

[sample]
k    = 16                # optional, default 16
seed = 42                # optional, default 42
```

### `[files]`

| field | required | meaning |
|---|---|---|
| `directory` | yes | path to look in. Relative paths resolve from the **current working directory** (project root if you run `bench.py` from there). Absolute paths work too. |
| `glob` | yes | Python `pathlib` glob pattern. `*.js` is non-recursive; use `**/*.js` for recursive. |
| `limit` | no | cap on how many matched files to include, after sorting matched paths lexically. Useful when a glob would otherwise pull in too many files. |

If `glob` matches **more than one file**, the files are concatenated (sorted
lexically) into a single combined corpus. A header is inserted between them
(`# === path ===` for Python, `// === path ===` for JS) so the model sees file
boundaries. Cross-file function-name collisions are de-duplicated — first
occurrence wins, the rest are silently skipped. The prompt qualifies the
target by file path so the model knows which one to reproduce.

All matched files must be **the same language** — the loader picks an extractor
based on the first file's extension and refuses to mix `.py` with `.js`.

Supported extensions: `.js`, `.mjs`, `.cjs` (esprima), `.py` (`ast`).

### `[sample]`

| field | default | meaning |
|---|---|---|
| `k` | 16 | how many functions to test per run. If the corpus has fewer than `k` functions with ≥ 20 body lines, all of them are tested. Selection is stratified by file position so you cover the whole file, not just the start. |
| `seed` | 42 | RNG seed for the stratified sampler. Same seed + same corpus = same target functions across runs. Keep this fixed when comparing models so each model sees the same questions. |

### Adding a new corpus

```
cp configs/corpora/jquery.toml configs/corpora/three.toml
# edit configs/corpora/three.toml: change directory/glob/limit
python3 bench.py extract --corpus three           # see what would be tested
python3 bench.py extract --corpus three --all     # see every extractable function
```

---

## Model configs — `configs/models/<name>.toml`

```toml
name              = "qwen3.6-35b-a3b"      # required
base_url          = "http://localhost:1234"
api_key           = "not-needed"
temperature       = 0.0
max_tokens        = 1500
timeout           = 600.0
suppress_thinking = true
reasoning_effort  = "none"                  # optional
prefill_no_think  = false                   # optional
stop              = ["\n---", "\nTask:"]    # optional
api_key_file      = ".secrets/openai.key"   # optional, hosted models
api_key_env       = "OPENAI_API_KEY"        # optional, hosted models
use_max_completion_tokens = true            # optional, OpenAI GPT-5+
```

| field | required | default | meaning |
|---|---|---|---|
| `name` | yes | — | the model identifier the **server** knows it by (what `lms ls` shows or what `/v1/models` returns). Doesn't have to match the file's stem. |
| `base_url` | no | `http://localhost:1234` | OpenAI-compatible endpoint root. Common ports: llama.cpp `8080`, LM Studio `1234`, Ollama `11434`. Hosted: `https://api.openai.com`. |
| `api_key` | no | `not-needed` | bearer token (literal value). Use **only** for non-secret tokens like local-server placeholders. For real keys see *Hosted models* below. |
| `api_key_file` | no | — | path to a file containing the key. Resolved relative to repo root if not absolute. **Use this for hosted-API keys** — see *Hosted models* below. Takes precedence over `api_key_env` and `api_key`. |
| `api_key_env` | no | — | environment variable name to read. Resolved at config-load time; errors if unset. Takes precedence over `api_key`. |
| `temperature` | no | `0.0` | keep at 0 for the benchmark. Some hosted reasoning models (OpenAI o-series, GPT-5.x) require `1.0` and reject other values. |
| `max_tokens` | no | `6000` | completion-token budget. If you can disable reasoning (see below), 1500 is plenty. If you can't, you need enough headroom for the entire CoT plus the ~20-line answer — see the matrix. |
| `use_max_completion_tokens` | no | `false` | when `true`, sends `max_completion_tokens` in the request body instead of `max_tokens`. Required for OpenAI GPT-5 family — they reject the older parameter name. |
| `timeout` | no | `600.0` | HTTP request timeout in seconds. Bump for slow CPU-only setups. |
| `suppress_thinking` | no | `true` | appends `/no_think` to the user message. The CLI flag `--think` flips this off. See the matrix below — only some models honor it. |
| `reasoning_effort` | no | `null` | sends `reasoning_effort: <value>` in the request body. Values vary by model: Qwen accepts `"none"` (or anything; usually ignored). OpenAI GPT-5 accepts `"none"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` but rejects `"minimal"`. OpenAI o-series accepts `"low"` / `"medium"` / `"high"` only. |
| `prefill_no_think` | no | `false` | adds an assistant message containing `<think>\n</think>\n\n` after the user prompt. The model continues from after `</think>`, skipping CoT. |
| `stop` | no | `null` | list of stop sequences sent to the server. Useful for models that **parrot the prompt back** after answering (Gemma 4 does this on every query). Pick strings that appear in your prompt boilerplate but not in real code — `"\n---"`, `"\nTask:"`, `"\nRules:"` are good defaults. |

### Disabling chain-of-thought — the actual matrix

Reasoning-on-by-default models burn through `max_tokens` on CoT before
producing any output. Three escape hatches exist; **which one works depends
on the model.** From measurement against this LM Studio build:

| Model | `suppress_thinking` (`/no_think`) | `reasoning_effort = "none"` | `prefill_no_think = true` |
|---|:---:|:---:|:---:|
| Qwen 3 4B            | ✅ honors      | ✅ honors  | ⚠ confuses model |
| Qwen 3.5 9B          | ❌ ignored     | ✅ honors  | ✅ honors |
| Qwen 3.6 35B (A3B)   | ❌ ignored     | ❌ ignored | ✅ honors |

Practical guide:
- **Qwen 3 (non-reasoning)**: `suppress_thinking = true` is enough.
- **Qwen 3.5 family**: `reasoning_effort = "none"`. `prefill_no_think` works too.
- **Qwen 3.6 family**: `prefill_no_think = true`. The other two don't help.
- **OpenAI o-series**: use `reasoning_effort = "low"` or `"minimal"` (the API
  rejects `"none"`); CoT can't be fully disabled but you can shrink it.
- **Unknown model**: probe by hand. The fields are independent and combining
  techniques is harmless on models that ignore the unrecognized ones.

When CoT is truly off, `max_tokens=1500` is plenty even for big models.

### Adding a new model

```
cp configs/models/qwen36-35b.toml configs/models/llama-3.3-70b.toml
# edit configs/models/llama-3.3-70b.toml:
#   name = "<id from `lms ls` or /v1/models>"
#   tune max_tokens / reasoning_effort / prefill_no_think for the model
python3 bench.py run --corpus http_server --model llama-3.3-70b
```

If the model is non-reasoning, drop the reasoning-disable fields and use
`max_tokens = 1500`.

### Skipping the config entirely

`--model FOO` works even without a file. If no `configs/models/FOO.toml`
exists, FOO is treated as a raw model identifier with the defaults above —
useful for one-off runs but not great for repeated comparisons (you'd be
re-typing the per-model knobs).

### Hosted models — API keys and security

For hosted endpoints (OpenAI, Anthropic, Together, Groq, …) you need to
authenticate with a real bearer token. **Don't put it in the model config
file directly** — `configs/` is committed; that would leak the key. Use one
of two indirections:

#### Option 1 (recommended): a key file under `.secrets/`

```
mkdir -p .secrets && chmod 700 .secrets
echo 'sk-proj-...' > .secrets/openai.key
chmod 600 .secrets/openai.key
```

Then in the model config:

```toml
api_key_file = ".secrets/openai.key"
```

The path is resolved relative to the repo root if not absolute. The file's
contents are read and stripped (trailing whitespace removed) once at
config-load time. The `.secrets/` folder, plus any `*.key` file, is already
in `.gitignore` so the key can't be accidentally committed.

#### Option 2: an environment variable

```
export OPENAI_API_KEY=sk-proj-...
```

Then in the model config:

```toml
api_key_env = "OPENAI_API_KEY"
```

Useful when keys are managed by your shell, direnv, a secret manager, or CI.

#### Verifying it's gitignored

```
git check-ignore -v .secrets/openai.key
# .gitignore:17:.secrets/   .secrets/openai.key
```

If you don't see a hit, your `.gitignore` is wrong — fix it before committing.

#### Hosted-model gotchas (observed)

| API | Quirk | Config workaround |
|---|---|---|
| OpenAI GPT-5 family | Rejects `max_tokens`; needs `max_completion_tokens` | `use_max_completion_tokens = true` |
| OpenAI GPT-5.5 | `reasoning_effort = "minimal"` rejected; `"none"` works | `reasoning_effort = "none"` |
| OpenAI o-series | Often forces `temperature = 1.0`; rejects `0.0` | `temperature = 1.0` |

### Example hosted-model config

```toml
# configs/models/gpt-5.5.toml
name              = "gpt-5.5"
base_url          = "https://api.openai.com"
api_key_file      = ".secrets/openai.key"
temperature       = 1.0
max_tokens        = 8000
reasoning_effort  = "none"
suppress_thinking = false
use_max_completion_tokens = true
```

---

## How configs combine at run time

There is **no inheritance file or shared parent**. The corpus and model are
loaded independently and stitched together in the runner.

**Layering order** (later wins):

1. Loader-level defaults (the table above)
2. Fields set in the model config file
3. CLI overrides — `--base-url`, `--max-tokens`, `--temperature`, `--timeout`,
   `--api-key`
4. Sample overrides (`-k`, `--seed`) layer over the corpus config the same way

`--file` and `--corpus` are mutually exclusive on the source side: use one or
the other, not both. For the model you can mix config + overrides freely:
`--model qwen36-35b --max-tokens 12000` reads the config and bumps just that
one knob for this run.

**Filename composition**: `results/<corpus-stem>__<model-stem>.json`. Stems
come from the **filename** (not the model's `name` field), so they're clean —
`qwen36-35b.toml` produces `…__qwen36-35b.json` regardless of what the model's
real identifier looks like.

---

## Tips

- **Comparing models**: run the same corpus with each. Same `seed` means same
  16 functions across all runs, so the diff is purely the model.
- **Stratified sampling matters**: if a corpus has 100+ functions and `k=16`,
  the 16 are spread across the file by start line. Don't manually pick the
  first 16 — the video's whole point is that recall degrades with depth.
- **Keep `suppress_thinking = true`** for recall benchmarks. CoT just costs
  tokens; it doesn't help the model reproduce code it has in front of it.
- **`base_url` per model** lets you point different models at different servers
  (e.g. local for small models, hosted for big ones).
