"""The source-example record and its invariants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_SCHEMA_VERSION = "1.0"

SLICES = (
    "rba_single",
    "asx_single",
    "afr_single",
    "cross",
    "sentiment",
    "insufficient",
    "robust",
)

DIFFICULTIES = ("easy", "medium", "hard")
DATASETS = ("RBA", "ASX", "AFR")


@dataclass
class Example:
    """One evidence-to-answer training example.

    ``required_facts`` is **1:1 with gradeable components** — never finer. Measured across
    all fifteen public questions, ``required_facts`` matches ``grading.components`` exactly
    every time. Splitting a compound fact into its parts would both teach the wrong output
    shape and make the local evaluator flatter the adapter by reporting 2-of-3 where the
    real grader reports zero.

    Every string in ``required_facts`` must appear verbatim in ``expected_answer``; the
    preparation script enforces this.
    """

    example_id: str
    question_family: str
    question: str
    difficulty: str
    datasets: list[str]
    slice: str
    requested_components: list[str]
    verified_evidence: dict[str, Any]
    limitations: list[dict[str, Any]]
    expected_answer: str
    required_facts: list[str]
    points: list[float]
    split_key: str
    tolerance: str = "exact"
    evidence_style: str = "json"
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    def to_line(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, default=str)


def build(
    *,
    family: str,
    slice_: str,
    question: str,
    facts: list[str],
    evidence: dict,
    components: list[str],
    datasets: list[str],
    difficulty: str,
    split_key: str,
    limitations: list[dict] | None = None,
    tolerance: str = "exact",
    evidence_style: str = "json",
    notes: str = "",
    joiner: str = " ",
    max_score: float = 10.0,
) -> Example:
    """Assemble an example so that ``expected_answer`` contains every fact verbatim.

    Points are split evenly across components with the rounding remainder on the last one,
    exactly as the public bank does (``3.33 / 3.33 / 3.34``).
    """
    answer = joiner.join(facts)
    n = len(facts)
    each = round(max_score / n, 2)
    points = [each] * n
    points[-1] = round(max_score - each * (n - 1), 2)
    return Example(
        example_id="",  # assigned by the generator
        question_family=family,
        question=question,
        difficulty=difficulty,
        datasets=datasets,
        slice=slice_,
        requested_components=components,
        verified_evidence=evidence,
        limitations=list(limitations or []),
        expected_answer=answer,
        required_facts=list(facts),
        points=points,
        split_key=split_key,
        tolerance=tolerance,
        evidence_style=evidence_style,
        notes=notes,
    )
