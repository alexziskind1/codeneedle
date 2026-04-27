"""Align model output against ground-truth lines and classify each line."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


PASS_THRESHOLD = 8  # video's threshold: ≥8 of 20 expected lines matched = pass


class LineTag(str, Enum):
    MATCHED = "matched"            # gray — expected (primary) line reproduced
    MISSING = "missing"            # orange — expected (primary) line not produced
    HALLUCINATED = "hallucinated"  # yellow — produced but not in expected window
    BONUS = "bonus"                # blue — produced, correct, past the primary 20


@dataclass
class LineResult:
    tag: LineTag
    text: str


@dataclass
class FunctionScore:
    name: str
    primary_matched: int
    primary_total: int
    hallucinated: int
    bonus_matched: int
    passed: bool
    expected_tagged: list[LineResult]    # expected primary side (matched/missing)
    predicted_tagged: list[LineResult]   # model output side (matched/halluc/bonus)
    error: str | None = None             # request errored or returned no usable content; renderers should show ERROR instead of FAIL so it isn't confused with a real recall miss


def score(
    name: str,
    primary: list[str],
    bonus: list[str],
    predicted_text: str,
    relax_indent: bool = False,
) -> FunctionScore:
    """Score a single function's predicted output against expected lines.

    `relax_indent=True` normalizes both sides with `.strip()` instead of
    `.rstrip()` only — i.e. leading whitespace is ignored when matching. Use
    this for models like Gemma that emit semantically-correct code but
    normalize indentation, where strict verbatim matching would unfairly
    penalize content the model actually got right. Default is strict.
    """
    predicted = _clean_output(predicted_text)
    norm = _norm_relaxed if relax_indent else _norm

    exp_primary = [norm(l) for l in primary]
    exp_bonus = [norm(l) for l in bonus]
    exp_full = exp_primary + exp_bonus
    pred = [norm(l) for l in predicted]

    # trim trailing blank lines on prediction (common model artifact)
    while pred and pred[-1] == "":
        pred.pop()

    sm = SequenceMatcher(a=exp_full, b=pred, autojunk=False)

    matched_exp = [False] * len(exp_full)
    # -1 = hallucinated, 0 = primary match, 1 = bonus match
    pred_kind = [-1] * len(pred)

    for block in sm.get_matching_blocks():
        if block.size == 0:
            continue
        for i in range(block.size):
            ei = block.a + i
            pi = block.b + i
            matched_exp[ei] = True
            pred_kind[pi] = 0 if ei < len(exp_primary) else 1

    primary_matched = sum(1 for i in range(len(exp_primary)) if matched_exp[i])
    bonus_matched = sum(
        1 for i in range(len(exp_primary), len(exp_full)) if matched_exp[i]
    )
    hallucinated = sum(1 for k in pred_kind if k == -1)

    # Blank lines shouldn't count as hallucinations (models often insert them).
    hallucinated -= sum(
        1 for i, k in enumerate(pred_kind) if k == -1 and pred[i].strip() == ""
    )

    # Display the ORIGINAL lines (with their actual indentation), not the
    # normalized form used for matching. Otherwise indent-relaxed scoring
    # would render every line lstripped, hiding the model's real output.
    expected_display = [l.rstrip() for l in primary]
    pred_display = [l.rstrip() for l in _clean_output(predicted_text)]
    while pred_display and pred_display[-1] == "":
        pred_display.pop()
    if len(pred_display) != len(pred):
        # Defensive: alignment of pred_display to pred should match because
        # both started from the same _clean_output and stripped trailing blanks.
        pred_display = pred_display[: len(pred)] + [""] * max(0, len(pred) - len(pred_display))

    expected_tagged = [
        LineResult(
            LineTag.MATCHED if matched_exp[i] else LineTag.MISSING,
            expected_display[i],
        )
        for i in range(len(exp_primary))
    ]
    kind_to_tag = {
        0: LineTag.MATCHED,
        1: LineTag.BONUS,
        -1: LineTag.HALLUCINATED,
    }
    predicted_tagged = [
        LineResult(kind_to_tag[pred_kind[i]], pred_display[i]) for i in range(len(pred))
    ]

    return FunctionScore(
        name=name,
        primary_matched=primary_matched,
        primary_total=len(exp_primary),
        hallucinated=hallucinated,
        bonus_matched=bonus_matched,
        passed=primary_matched >= PASS_THRESHOLD,
        expected_tagged=expected_tagged,
        predicted_tagged=predicted_tagged,
    )


def _norm(s: str) -> str:
    # Preserve leading indentation; strip trailing whitespace (models are inconsistent there).
    return s.rstrip()


def _norm_relaxed(s: str) -> str:
    # Used when scoring indent-blind. Strips both leading and trailing whitespace.
    # Internal whitespace is preserved so things like `a    b` stay distinct from `a b`.
    return s.strip()


def _clean_output(text: str) -> list[str]:
    """Strip markdown fences and surrounding blank lines. Tolerant of prefix commentary."""
    lines = text.splitlines()

    # If the model wrapped output in a fenced code block, extract the fence contents.
    fence_idxs = [i for i, l in enumerate(lines) if l.lstrip().startswith("```")]
    if len(fence_idxs) >= 2:
        lines = lines[fence_idxs[0] + 1 : fence_idxs[-1]]
    else:
        # Drop any stray fence markers
        lines = [l for l in lines if not l.lstrip().startswith("```")]

    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines
