"""ANSI-colored rendering of scored results."""
from __future__ import annotations

import sys

from .scorer import FunctionScore, LineTag


# Colors chosen to mirror the video's legend:
#   gray   — correct (matched)
#   orange — expected but missing
#   yellow — hallucinated / mangled
#   blue   — extra correct lines past the primary 20
COLOR = {
    LineTag.MATCHED: "\x1b[37m",        # white/gray
    LineTag.MISSING: "\x1b[38;5;208m",  # 256-color orange
    LineTag.HALLUCINATED: "\x1b[33m",   # yellow
    LineTag.BONUS: "\x1b[36m",          # cyan (blue-ish)
}
RESET = "\x1b[0m"
BOLD = "\x1b[1m"


def _colorize(enabled: bool, color: str, text: str) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def render_function(score: FunctionScore, color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()

    status = "PASS" if score.passed else "FAIL"
    status_color = "\x1b[32m" if score.passed else "\x1b[31m"
    header = (
        f"\n=== {score.name}  "
        f"[{_colorize(color, status_color, status)}]  "
        f"matched={score.primary_matched}/{score.primary_total}  "
        f"hallucinated={score.hallucinated}  "
        f"bonus={score.bonus_matched} ==="
    )
    out = [header, "  -- model output --"]
    for r in score.predicted_tagged:
        out.append("  " + _colorize(color, COLOR[r.tag], r.text))

    missing = [r for r in score.expected_tagged if r.tag == LineTag.MISSING]
    if missing:
        out.append("  -- missing expected lines --")
        for r in missing:
            out.append("  " + _colorize(color, COLOR[r.tag], r.text))
    return "\n".join(out)


def render_summary(scores: list[FunctionScore], color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()
    passed = sum(1 for s in scores if s.passed)
    total_matched = sum(s.primary_matched for s in scores)
    total_possible = sum(s.primary_total for s in scores)
    total_halluc = sum(s.hallucinated for s in scores)
    total_bonus = sum(s.bonus_matched for s in scores)

    lines = [
        "",
        _colorize(color, BOLD, "=== SUMMARY ==="),
        f"  Pass:                  {passed}/{len(scores)}",
        f"  Primary lines matched: {total_matched}/{total_possible}",
        f"  Hallucinated lines:    {total_halluc}",
        f"  Bonus (extra correct): {total_bonus}",
    ]

    # Per-function one-liner
    lines.append("")
    lines.append("  per-function:")
    for s in scores:
        mark = _colorize(color, "\x1b[32m", "✓") if s.passed else _colorize(color, "\x1b[31m", "✗")
        lines.append(
            f"    {mark} {s.name:<40} "
            f"matched={s.primary_matched:>2}/{s.primary_total}  "
            f"halluc={s.hallucinated:>2}  bonus={s.bonus_matched:>2}"
        )
    return "\n".join(lines)
