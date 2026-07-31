"""The component judge -- one expected fact at a time, YES or NO.

FROZEN in the base commit (SESSION_KICKOFF.md §4, F4). Imported by:

  * src/eval/run_offline_eval.py  (session C)
  * training/eval_compare.py      (session D)

Both needed the same per-component judge. Two implementations would mean two
different numbers for the same thing, which is why this is frozen rather than
written twice.

This replicates the organizer's protocol rather than approximating it.
handout/03_scoring_and_examples.md specifies it precisely: the judge is
Qwen3.6-35B-A3B-FP8 via the private agent-brain service, it receives the
question, the answer, and **each expected fact one at a time**, and replies YES
or NO. A holistic rubric judge would correlate with the leaderboard; this one
predicts it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .. import config

JUDGE_SYSTEM = (
    "You are a strict grader. You are given a question, a candidate answer, and ONE "
    "expected fact. Decide whether the candidate answer states that expected fact. "
    "Equivalent date formats, harmless numeric formatting differences, and sentiment "
    "synonyms that preserve the meaning are acceptable. A hedged or approximated number "
    "is NOT acceptable. Reply with exactly one word: YES or NO."
)

JUDGE_USER = (
    "Question: {question}\n\n"
    "Candidate answer: {answer}\n\n"
    "Expected fact: {expected_fact}\n\n"
    "Does the candidate answer state this expected fact? Reply YES or NO."
)


@dataclass
class Verdict:
    expected_fact: str
    points: float
    passed: bool
    raw: str


def _post(payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{config.LITELLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {config.LITELLM_KEY}"}
                if config.LITELLM_KEY
                else {}
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def judge_fact(
    question: str,
    answer: str,
    expected_fact: str,
    *,
    points: float = 0.0,
    timeout: float = 30.0,
) -> Verdict:
    """Judge one expected fact. temperature=0, thinking off, pinned alias.

    chat_template_kwargs is required, not optional: without it a Qwen turn costs
    15.0s and 800 tokens instead of ~0.9s and 43. Over ~120 calibration calls
    that is the difference between minutes and half an hour.

    A judge failure returns NO rather than raising -- a crashed judge must not
    take down an eval run, and scoring a fact we could not verify as present
    would inflate the number.
    """
    payload = {
        "model": config.EVAL_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    question=question, answer=answer, expected_fact=expected_fact
                ),
            },
        ],
        "max_tokens": 4,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        data = _post(payload, timeout)
        raw = data["choices"][0]["message"]["content"].strip()
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        return Verdict(expected_fact, points, False, f"JUDGE_ERROR: {exc}")

    return Verdict(expected_fact, points, raw.upper().startswith("YES"), raw)


def score_question(
    question: str,
    answer: str,
    components: list[dict],
    *,
    timeout: float = 30.0,
) -> tuple[float, float, list[Verdict]]:
    """Score one question's components. Returns (earned, max, verdicts).

    Points-weighted, not fact-counted (kickoff F12 / review E.3). Granularity
    varies sharply across the 15 -- from a single 10-point compound (MHQ001) to
    five components worth 1-3 each (MHQ080) -- so a plain fraction-of-facts
    metric misreads it and flatters the model exactly where the points are.

    `components` are the `grading.components` entries from
    public_questions.jsonl: {"expected_fact": str, "points": float}.
    """
    verdicts = [
        judge_fact(
            question,
            answer,
            c["expected_fact"],
            points=float(c.get("points", 0.0)),
            timeout=timeout,
        )
        for c in components
    ]
    earned = sum(v.points for v in verdicts if v.passed)
    total = sum(v.points for v in verdicts)
    return earned, total, verdicts


def penalised(earned: float, elapsed_s: float) -> float:
    """Apply the organizer's response-time rule to an earned score.

    Challenge_Brief.md § Response-Time Rules: <=60s full credit, <=300s minus
    20%, >300s zero. This is the headline number -- an unpenalised score is not
    the score you get.
    """
    if elapsed_s <= config.PENALTY_THRESHOLD_S:
        return earned
    if elapsed_s <= 300.0:
        return earned * 0.8
    return 0.0
