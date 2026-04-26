# Analysis & visualizations

```
analysis/
  visualize.py       Plotly dashboard builder
  charts/            generated HTML output (gitignored)
  VIZ_README.md      this file
```

## Run it

```
python3 analysis/visualize.py
```

Reads every `results/*.json`, groups runs by their corpus (the `files` field
in each dump), and writes one dashboard HTML per corpus to `analysis/charts/`.
Then opens with:

```
open analysis/charts/index.html
```

### Options

```
python3 analysis/visualize.py --results-dir results --output-dir analysis/charts
```

- `--results-dir` (default `results/`) — where to find result JSONs
- `--output-dir`  (default `analysis/charts/`) — where to write HTML

Re-run any time. It overwrites the HTML; nothing is incremental.

## What each chart shows

Every dashboard has three sections, each answering one question. Models get
a stable color across all charts so you can track them visually.

### 1. Leaderboard

Horizontal bars sorted best → worst by total primary lines matched.
Annotation on each bar: `<matched>/<total> lines · <P>/<N> pass · <H> halluc`.

This is the *who won* view. If you're glancing at one number, this is it.

### 2. Per-function score

One bar per (model × function), grouped by function and sorted left-to-right
from easiest (highest cross-model average) to hardest. A horizontal dashed
line marks the pass threshold (8/20).

This is the *where do models diverge* view. Functions where every model gets
20/20 are uninteresting; the spread on the right side of the chart is where
the benchmark is doing real work.

### 3. Recall vs. position in file

Each marker is a function placed at its **start line in the source**.
Connected with a line per model. Y-axis is `% primary lines matched`.

This is the chart that tests the video's core thesis: **does recall fall off
as functions appear deeper in context?**

How to read it:

| pattern | interpretation |
|---|---|
| flat line near 100% | model is comfortable with the entire context length |
| flat line, **but lower** (e.g. 60%) | quality issue, not depth — model recalls some but not all, equally bad everywhere |
| downward slope from left to right | classic sliding-window collapse — model is forgetting earlier context |
| bouncy / no trend | individual functions are easier or harder; depth isn't the bottleneck |

**Important caveat.** This chart only stresses positional recall when the
corpus is big enough to push the model past its "easy" zone. For a 14K-token
file (`http_server`), a model with 128K context is barely warmed up — both
qwen3.5 and qwen3.6 will look flat. To actually exercise long-context decay
you need ~80K+ tokens in the prompt: run the `jquery` corpus.

## Grouping rule

Runs are grouped by **corpus**, not model. The grouping key is the set of
file basenames in the dump's `files` field. So:

- All runs against `fixtures/http_server.py` → `analysis/charts/http_server.html`
- All runs against `fixtures/jquery.js` → `analysis/charts/jquery.html`
- A run against `fixtures/foo.py + fixtures/bar.py` → its own group

Within a dashboard, each run shows up as its own colored line / bar even if
it's the same model — re-running is a separate trace, distinguished in the
legend by the result-file stem (`http_server__qwen36-35b`).

## Data sources

The script reads:

- `results/*.json` — all the dumps `bench.py run` produced
- `fixtures/<file>` — re-extracted at viz time so the position chart can map
  function names → start lines (this is robust against repo renames)

It does *not* require the original `--corpus` config to render — only the
dump JSON and the fixture file. If the fixture moved, the position chart
silently drops the affected functions.

## Adding a new chart

The shape is in `analysis/visualize.py`:

1. Write a function `def my_chart(runs, colors)` that returns a `plotly.graph_objects.Figure`.
2. Append `("Section title", "caption shown under the chart", my_chart(runs, colors))` to the `sections` list inside `write_dashboard`.

The dashboard automatically renders the section, adds it to the top nav, and
shares the model-color palette with the existing charts.
