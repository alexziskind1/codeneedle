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


def score(
    name: str,
    primary: list[str],
    bonus: list[str],
    predicted_text: str,
) -> FunctionScore:
    predicted = _clean_output(predicted_text)

    exp_primary = [_norm(l) for l in primary]
    exp_bonus = [_norm(l) for l in bonus]
    exp_full = exp_primary + exp_bonus
    pred = [_norm(l) for l in predicted]

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

    expected_tagged = [
        LineResult(
            LineTag.MATCHED if matched_exp[i] else LineTag.MISSING,
            exp_primary[i],
        )
        for i in range(len(exp_primary))
    ]
    kind_to_tag = {
        0: LineTag.MATCHED,
        1: LineTag.BONUS,
        -1: LineTag.HALLUCINATED,
    }
    predicted_tagged = [
        LineResult(kind_to_tag[pred_kind[i]], pred[i]) for i in range(len(pred))
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
