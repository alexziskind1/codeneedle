#!/usr/bin/env python3
"""Generate Plotly comparison dashboards from results/*.json.

One dashboard per corpus (grouped by the `files` field in the dump):

    1. Leaderboard          — lines matched per model, sorted best → worst
    2. Per-function bars    — each function's score across models
    3. Recall vs. position  — does recall fall off deeper in the file?

Output: analysis/charts/<corpus>.html + analysis/charts/index.html.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# This file lives in analysis/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import bench…` works regardless of cwd
PASS_THRESHOLD = 8    # matches bench/scorer.py

# Stable color palette — assigned once per model so every chart uses the same color.
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#ff9da6", "#9d755d", "#bab0ac", "#b279a2", "#eeca3b",
]


@dataclass
class Run:
    path: Path
    model: str
    group_name: str
    data: dict


def _group_name(data: dict) -> str:
    files = data.get("files") or ([data["source"]] if data.get("source") else [])
    if not files:
        return "unknown"
    if len(files) == 1:
        return Path(files[0]).stem
    return "+".join(Path(f).stem for f in files[:3])


def load_runs(results_dir: Path) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        if not data.get("results"):
            continue
        group = _group_name(data)
        groups[group].append(Run(path=p, model=data.get("model", p.stem), group_name=group, data=data))
    return groups


def resolve_line_positions(runs: list[Run]) -> dict[str, int]:
    """Map function name → start_line. Re-extracts from source so the depth
    chart works for old dumps too. Looks up files by basename under fixtures/
    if the original absolute path no longer exists (e.g. after a repo rename).
    """
    from bench.extract import extract as bench_extract

    fixtures_dirs = [REPO_ROOT / "fixtures"]
    name_to_line: dict[str, int] = {}
    tried: set[Path] = set()
    for run in runs:
        for raw in run.data.get("files") or ([run.data.get("source")] if run.data.get("source") else []):
            if not raw:
                continue
            p = Path(raw)
            if not p.exists():
                for d in fixtures_dirs:
                    alt = d / p.name
                    if alt.exists():
                        p = alt
                        break
                else:
                    continue
            if p in tried:
                continue
            tried.add(p)
            try:
                for t in bench_extract(p):
                    name_to_line.setdefault(t.name, t.start_line)
            except Exception as e:
                print(f"  (couldn't re-extract {p.name} for depth chart: {e})", file=sys.stderr)
    return name_to_line


def assign_colors(runs: list[Run]) -> dict[str, str]:
    models = sorted({r.model for r in runs})
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


# --- charts ---------------------------------------------------------------


def leaderboard(runs: list[Run], colors: dict[str, str]):
    """Horizontal bar: total primary lines matched, sorted desc. One bar per model.

    Bar annotated with 'M/T lines, P/N passed'. This is the 'who won' view.
    """
    import plotly.graph_objects as go

    rows = []
    for r in runs:
        matched = sum(x.get("primary_matched", 0) for x in r.data["results"])
        total = sum(x.get("primary_total", 0) for x in r.data["results"])
        passed = sum(1 for x in r.data["results"] if x.get("passed"))
        queries = len(r.data["results"])
        halluc = sum(x.get("hallucinated", 0) for x in r.data["results"])
        rows.append((r.model, matched, total, passed, queries, halluc, r.path.stem))

    rows.sort(key=lambda row: row[1], reverse=True)

    labels = [r[6] for r in rows]  # filename stem — distinguishes repeat runs of same model
    matched = [r[1] for r in rows]
    totals = [r[2] for r in rows]
    annotations = [
        f"{r[1]}/{r[2]} lines · {r[3]}/{r[4]} pass · {r[5]} halluc"
        for r in rows
    ]
    bar_colors = [colors[r[0]] for r in rows]
    hover = [f"{r[0]}<br>run: {r[6]}" for r in rows]

    fig = go.Figure(go.Bar(
        x=matched, y=labels, orientation="h",
        text=annotations, textposition="outside",
        marker_color=bar_colors,
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title="Leaderboard · total primary lines matched (of possible)",
        xaxis=dict(title="lines matched", range=[0, (max(totals) if totals else 1) * 1.35]),
        yaxis=dict(autorange="reversed"),
        height=max(260, 60 * len(rows) + 120),
        margin=dict(l=220, r=40, t=60, b=60),
    )
    return fig


def per_function_bars(runs: list[Run], colors: dict[str, str]):
    """Grouped bars: one bar per (function × model), Y=lines matched (0..primary_total).

    Horizontal dashed line at the pass threshold so you can eyeball
    'which functions did which models pass?' without reading numbers.
    """
    import plotly.graph_objects as go

    all_fns: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            all_fns.add(x["function"])

    # Sort functions by cross-model mean matched (hard ones last).
    def mean_score(fn: str) -> float:
        xs = []
        for r in runs:
            x = next((y for y in r.data["results"] if y["function"] == fn), None)
            if x and x.get("primary_total"):
                xs.append(x["primary_matched"] / x["primary_total"])
        return sum(xs) / len(xs) if xs else 0.0

    fns = sorted(all_fns, key=mean_score, reverse=True)

    fig = go.Figure()
    for r in runs:
        y = []
        total_max = 20
        for fn in fns:
            x = next((z for z in r.data["results"] if z["function"] == fn), None)
            if x is None:
                y.append(None)
            else:
                y.append(x.get("primary_matched", 0))
                total_max = max(total_max, x.get("primary_total", 20))
        fig.add_bar(
            x=fns, y=y,
            name=f"{r.model} · {r.path.stem}",
            marker_color=colors[r.model],
            hovertemplate="%{x}<br>%{y} lines matched<extra>" + r.path.stem + "</extra>",
        )

    fig.add_hline(
        y=PASS_THRESHOLD, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-function score · bars above the dashed line = pass",
        xaxis=dict(title="function (sorted by average difficulty)", tickangle=-40),
        yaxis=dict(title="primary lines matched", range=[0, total_max + 2]),
        barmode="group",
        height=max(380, 32 * len(fns) + 240),
        margin=dict(l=60, r=40, t=60, b=160),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    return fig


def recall_vs_depth(runs: list[Run], colors: dict[str, str], positions: dict[str, int]):
    """Scatter + line: X=line number in source (depth), Y=% matched.

    Tests the benchmark's core thesis: do models lose recall as functions
    appear deeper in context? Each model gets one connected trace.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    any_data = False
    max_line = 0
    for r in runs:
        pts = []
        for x in r.data["results"]:
            fn = x["function"]
            if fn not in positions:
                continue
            total = x.get("primary_total") or 20
            pct = x.get("primary_matched", 0) / total * 100
            pts.append((positions[fn], pct, fn, x.get("primary_matched", 0), total))
        if not pts:
            continue
        any_data = True
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        max_line = max(max_line, max(xs))
        ys = [p[1] for p in pts]
        hover = [
            f"{p[2]}<br>line {p[0]:,}<br>{p[3]}/{p[4]} matched ({p[1]:.0f}%)"
            for p in pts
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=f"{r.model} · {r.path.stem}",
            line=dict(color=colors[r.model], width=2),
            marker=dict(size=10, color=colors[r.model]),
            hovertext=hover, hoverinfo="text",
        ))

    if not any_data:
        return None

    fig.add_hline(
        y=PASS_THRESHOLD / 20 * 100, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD}/20 = {PASS_THRESHOLD/20*100:.0f}%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Recall vs. position in file · left = near top of context, right = deep",
        xaxis=dict(title="function start line (deeper in file →)", range=[0, max_line * 1.05]),
        yaxis=dict(title="% primary lines matched", range=[-5, 108]),
        height=460,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    return fig


# --- HTML assembly --------------------------------------------------------


def write_dashboard(group: str, runs: list[Run], out_path: Path) -> None:
    colors = assign_colors(runs)
    positions = resolve_line_positions(runs)

    sections = [
        (
            "Leaderboard",
            "Total primary lines matched across all tested functions. "
            "Sorted so the top bar is the best run. `halluc` = extra lines the model emitted that don't match the expected window.",
            leaderboard(runs, colors),
        ),
        (
            "Per-function score",
            "One bar per model for each function, sorted left-to-right from easiest to hardest. "
            "Bars above the dashed line passed (≥ 8 of 20 primary lines matched).",
            per_function_bars(runs, colors),
        ),
        (
            "Recall vs. position in file",
            "Each marker is a function placed at its line number in the source. "
            "If recall falls off as x increases, the model is losing context as depth grows — the video's core finding for sliding-window models.",
            recall_vs_depth(runs, colors, positions),
        ),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"<!doctype html><html><head><meta charset=utf-8><title>{group}</title>")
        f.write("""<style>
          body{font-family:system-ui,-apple-system,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1.25rem;color:#222;}
          h1{margin:.25rem 0;}
          p.meta{color:#666;margin-top:0;margin-bottom:2rem;}
          section{margin:0 0 2.5rem 0;}
          h2{margin:0 0 .25rem 0;font-size:1.25rem;}
          p.caption{color:#555;margin:0 0 .5rem 0;font-size:.95rem;}
          nav{margin-bottom:1.5rem;font-size:.9rem;}
          nav a{color:#4c78a8;text-decoration:none;margin-right:1rem;}
        </style>""")
        f.write(f"</head><body>")
        f.write(f"<h1>{group}</h1>")
        models = sorted({r.model for r in runs})
        queries = sum(len(r.data["results"]) for r in runs)
        f.write(f"<p class=meta>{len(runs)} run(s) · {queries} queries · "
                f"models: {', '.join(models)}</p>")
        f.write('<nav>')
        for title, _cap, fig in sections:
            if fig is None:
                continue
            anchor = title.lower().replace(" ", "-")
            f.write(f'<a href="#{anchor}">{title}</a>')
        f.write('</nav>')

        first = True
        for title, caption, fig in sections:
            if fig is None:
                continue
            anchor = title.lower().replace(" ", "-")
            f.write(f'<section id="{anchor}">')
            f.write(f"<h2>{title}</h2>")
            f.write(f'<p class=caption>{caption}</p>')
            f.write(fig.to_html(include_plotlyjs="cdn" if first else False, full_html=False))
            f.write("</section>")
            first = False
        f.write("</body></html>")


def write_index(groups: dict[str, list[Run]], out_dir: Path) -> Path:
    idx = out_dir / "index.html"
    with idx.open("w") as f:
        f.write("<!doctype html><html><head><meta charset=utf-8><title>codeneedle dashboards</title>")
        f.write("<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;}"
                "li{margin:.4rem 0;}small{color:#888;}a{color:#4c78a8;text-decoration:none;}"
                "a:hover{text-decoration:underline;}</style></head><body>")
        f.write("<h1>codeneedle · benchmark dashboards</h1><ul>")
        for name in sorted(groups):
            runs = groups[name]
            models = sorted({r.model for r in runs})
            f.write(f'<li><a href="{name}.html">{name}</a> '
                    f'<small>— {len(runs)} run(s), models: {", ".join(models)}</small></li>')
        f.write("</ul></body></html>")
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: analysis/charts/")
    args = ap.parse_args(argv)

    out_dir = args.output_dir or (REPO_ROOT / "analysis" / "charts")
    groups = load_runs(args.results_dir)
    if not groups:
        print(f"no usable result JSON files in {args.results_dir}")
        return 1

    total_runs = sum(len(r) for r in groups.values())
    print(f"Loaded {total_runs} run(s) in {len(groups)} group(s)")
    for name, runs in sorted(groups.items()):
        out = out_dir / f"{name}.html"
        write_dashboard(name, runs, out)
        print(f"  {name}: {len(runs)} run(s) -> {out}")

    idx = write_index(groups, out_dir)
    print(f"\nopen {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
