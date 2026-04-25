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
max_tokens        = 6000
timeout           = 600.0
suppress_thinking = true
```

| field | required | default | meaning |
|---|---|---|---|
| `name` | yes | — | the model identifier the **server** knows it by (what `lms ls` shows or what `/v1/models` returns). Doesn't have to match the file's stem. |
| `base_url` | no | `http://localhost:1234` | OpenAI-compatible endpoint root. Common ports: llama.cpp `8080`, LM Studio `1234`, Ollama `11434`. |
| `api_key` | no | `not-needed` | bearer token. Local servers ignore it; hosted APIs need a real value. |
| `temperature` | no | `0.0` | keep at 0 for the benchmark — recall isn't a creative task and determinism makes results comparable. |
| `max_tokens` | no | `6000` | completion-token budget. **Reasoning models** (qwen3.5+, o-series) need a lot — the chain-of-thought eats tokens before the answer. Rule of thumb: 1500 for non-reasoning, 6000 for "small" reasoning, 8000–12000 for big ones. If responses come back empty, this is almost always why. |
| `timeout` | no | `600.0` | HTTP request timeout in seconds. Bump for slow CPU-only setups. |
| `suppress_thinking` | no | `true` | appends `/no_think` to the user message. Qwen3 4B (non-reasoning) honors this; Qwen3.5+ ignores it but the marker is harmless. The CLI flag `--think` flips this off if you want to compare CoT vs no-CoT recall. |

### Adding a new model

```
cp configs/models/qwen36-35b.toml configs/models/llama-3.3-70b.toml
# edit configs/models/llama-3.3-70b.toml:
#   name = "<id from `lms ls` or /v1/models>"
#   tune max_tokens, suppress_thinking for that model's behavior
python3 bench.py run --corpus http_server --model llama-3.3-70b
```

If the model isn't reasoning-heavy, you can drop `max_tokens` to 1500.

### Skipping the config entirely

`--model FOO` works even without a file. If no `configs/models/FOO.toml`
exists, FOO is treated as a raw model identifier with the defaults above —
useful for one-off runs but not great for repeated comparisons (you'd be
re-typing the per-model knobs).

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
