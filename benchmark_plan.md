# Positional Recall Benchmark — Analysis & Implementation Plan

## What the benchmark measures

It's a **positional recall benchmark** on long context. Rather than the classic "needle in a haystack" (find a random string inserted into filler text), it stuffs a real 108K-token source file into the model and asks it to reproduce a *specific* chunk by *identity*, not by position description.

The probe is carefully worded: *"reproduce the first 20 lines of function X **following** its opening brace."* The word "following" is what makes it positional — the model has to anchor to `function X {` in its context and emit what comes next, in order, verbatim. It's a memory-retrieval task disguised as a coding task.

The twist versus standard NIAH: the haystack is real code full of repeated boilerplate (1,300 functions, lots of near-duplicate return statements), so the model can't distinguish targets by surface novelty. That makes it a much harder recall test and it exposes architectural weaknesses — e.g. Gemma's 1K sliding-window attention collapses on long spans, while Qwen's gated DeltaNet holds up.

## How it works

1. **Corpus**: one large real-world file (~108K tokens, 8K+ lines, 1,300 function defs). Beautified, not minified, so line structure is meaningful.
2. **Targets**: 16 hand-picked functions spread across the file.
3. **Prompt**: `[entire file] + [query: "return the first 20 lines following the opening brace of function <name>"]`.
4. **Runs**: 16 independent queries per model. The file prefix is cached in the KV cache so only the tail query re-processes each time — that's both a speed optimization and what makes the runs comparable (identical prefix state).
5. **Fairness controls**: same runtime (llama.cpp), same 8-bit KV cache quantization, same sampling. He also swaps quant builds (Unsloth Q4_K_XL vs. LM Studio Q4_K_M) because builds matter.
6. **Scoring per function**:
   - Gray = line matches ground truth at the right position
   - Orange = expected line missing
   - Yellow = hallucinated / mangled line
   - Blue = extra correct lines beyond 20 (bonus)
   - Pass threshold: ≥ 8 of the 20 lines correct
7. **Aggregate**: pass/fail count out of 16, plus totals for matched / missing / hallucinated / extra lines.

## Implementation plan

### 1. Pick a corpus
Any sizable source file works — JS, Python, a concatenated repo — as long as it's big enough to push past short-context attention (~50K+ tokens) and has many similarly-shaped functions so surface-level retrieval doesn't help.

### 2. Extract functions + ground truth
Parse with a real parser, not regex:
- JS → `@babel/parser` or `acorn` + `estree-walker`
- Python → `ast`
- Multi-language → tree-sitter

For each function node, record `name`, `start_line_after_opening_brace`, and the next 20 source lines verbatim. Filter to functions with ≥ 20 body lines. Sample 16 spanning the beginning, middle, and end of the file (position matters — that's half the point).

### 3. Inference harness
Use `llama.cpp`'s HTTP server (`/completion` or `/v1/chat/completions`). Key flags:
- `--ctx-size` large enough for file + query + response
- `--cache-type-k q8_0 --cache-type-v q8_0` (matches the video's 8-bit KV cache)
- `--cache-reuse` / prompt caching on — so the file prefix isn't re-ingested for each of the 16 queries
- Temperature 0 for reproducibility

Prompt template:
```
<file contents>

Reproduce verbatim the first 20 lines following the opening brace
of the function named `<NAME>`. Output only those lines, no commentary.
```

### 4. Response parsing
Strip markdown code fences, trim. Split into lines.

### 5. Scorer
Do a line-level alignment between predicted and expected lines — `difflib.SequenceMatcher` on stripped lines works, or a proper LCS. Classify each expected line as matched/missing and each predicted line as matched/hallucinated. Count extra-but-correct lines beyond the 20 (the "blue" category) by extending the expected window and re-aligning. Record pass if matched ≥ 8.

### 6. Reporting
Per-function color-coded diff view (ANSI in terminal or HTML), plus an aggregate table: `{model, build, passes/16, matched, missing, hallucinated, extra}`.

### 7. Matrix runs
Script the sweep across (model × build × quant). Each full run is 16 queries; with KV-cache reuse the marginal cost is small.

## Likely gotchas

- **Whitespace normalization.** Models often normalize indentation or quote style. Decide up front whether to strip/normalize before comparing, or score verbatim. The video appears to score close-to-verbatim.
- **Prompt-cache invalidation.** If anything before the query changes between runs (e.g. you include the function name in a system prompt slot that's before the file), cache reuse breaks and latency explodes. Keep the file first, query last.
- **KV-cache quantization is a variable.** Q8 KV changes recall quality. Hold it constant across models, and ideally run the whole suite at full-precision KV once as a control.
- **Function-selection bias.** If all 16 picks are near the file end, you're testing recency, not position. Stratify by position bucket.
- **"Following the opening brace"** — in arrow functions or one-liners this is ambiguous. Restrict to classic `function foo(...) { \n ... }` forms.

## Source video context

Transcribed video describes the author comparing three local LLMs on this benchmark:

- **Gemma 3 27B** (sliding-window attention, 1K look-back) — 6/16 pass (Unsloth Q4_K_XL), 2/16 pass (LM Studio Q4_K_M). Silently dropped content inside long return statements, which broke downstream reverse-engineering tasks.
- **Qwen 3.5 35B** (gated DeltaNet) — 11/16 pass. 245 matched lines, 98 extra correct, 50 truncated.
- **Qwen 3.6 35B A3B** — near-perfect. 283/320 lines matched on best build; only two hallucinated lines across the whole suite.

All runs: llama.cpp runtime, 8-bit KV cache, same sampling, ~108K-token input.
