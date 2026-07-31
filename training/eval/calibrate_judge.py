"""Judge calibration -- does the component judge accept paraphrase?

Phase 0 artifact (SESSION_KICKOFF.md §4, F6 / Q1). Session D READS the output,
does not run this.

The question this answers: the component judge asks Qwen YES/NO per
expected_fact. Does it accept the same facts reordered and reformatted, or only
something close to the reference's own wording?

  paraphrase arm passes -> D's assistant target is NATURAL SYNTHESIS
  paraphrase arm fails   -> D's target is NEAR-TEMPLATE REPRODUCTION

That is a materially different training set, and deciding it after generating
5,000 records means regenerating 5,000 records.

Three arms over the 15 public questions, via the frozen src/eval/component_judge:

  control     reference_answer verbatim              -> expect ALL YES
  negative    one number perturbed beyond tolerance  -> expect that one NO
  paraphrase  same facts, reordered, dates reformatted, different wording
                                                     -> expect ALL YES

Run:  .venv/bin/python -m training.eval.calibrate_judge
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval import component_judge  # noqa: E402

QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"
OUT_MD = REPO_ROOT / "training" / "eval" / "judge_calibration.md"
OUT_JSON = REPO_ROOT / "training" / "eval" / "judge_calibration.json"

MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def reformat_dates(text: str) -> str:
    """"5 Jun 2019" -> "2019-06-05", and "2019-06-05" -> "5 Jun 2019".

    Challenge_Brief.md says "equivalent date formats ... are accepted", so this
    is testing a documented tolerance, not hoping for luck.
    """

    def to_iso(m: re.Match) -> str:
        day, mon, year = m.group(1), m.group(2)[:3].lower(), m.group(3)
        return f"{year}-{MONTHS.get(mon, '01')}-{int(day):02d}"

    text = re.sub(r"\b(\d{1,2}) ([A-Za-z]{3,9}) (\d{4})\b", to_iso, text)

    inv = {v: k.capitalize() for k, v in MONTHS.items()}

    def to_long(m: re.Match) -> str:
        year, mon, day = m.group(1), m.group(2), m.group(3)
        return f"{int(day)} {inv.get(mon, 'Jan')} {year}"

    return re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", to_long, text)


def paraphrase(text: str) -> str:
    """Same facts, different surface. Must NEVER alter a number.

    Only the tolerances Challenge_Brief.md documents as accepted are exercised:
    equivalent date formats, clause reordering, and different wording.

    A previous version of this also inserted thousands separators into bare
    4-digit integers, which silently turned the YEAR 2015 into "2,015" — a fact
    corruption, not a paraphrase. The judge correctly rejected it and the arm
    reported false intolerance. Do not reintroduce numeric reformatting here
    without excluding years and date components.

    Two further corruptions were found and removed the same way:

      * Lowercasing the leading character to fit a "Based on the dataset, ..."
        prefix turned the TICKER `AMP.AX` into `aMP.AX`. That cost MHQ049 its
        entire 10-point component. Never case-fold; prefix with a colon instead.
      * Reordering at comma/semicolon boundaries split compound sentences
        mid-fact. Per FINETUNE_PLAN section 4.5 a compound fact spread across
        two sentences legitimately scores zero, so that arm was measuring the
        compound-fact rule, not paraphrase tolerance. Reorder whole SENTENCES
        only.
    """
    out = reformat_dates(text)
    # Reorder whole sentences, so fact order differs while every compound fact
    # stays intact inside its own sentence.
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", out) if s.strip()]
    if len(sentences) > 1:
        out = " ".join(sentences[1:] + sentences[:1])
    # No case folding -- tickers and proper nouns are facts.
    return f"Based on the dataset: {out}"


def perturb(text: str, expected_fact: str) -> tuple[str, bool]:
    """Break one number that actually appears in the TARGET component.

    Perturbing the first number in `reference_answer` is not a valid negative
    control: that number may belong to a different component (or to none), so
    the targeted component can legitimately still pass. Pick a number from the
    component under test and change it beyond any stated tolerance.
    """
    nums = re.findall(r"\d+\.?\d*", expected_fact)
    # Prefer a distinctive number: skip short values likely to be a year or a
    # small ordinal shared across components.
    target = next((n for n in nums if len(n.replace(".", "")) >= 3), nums[0] if nums else None)
    if target is None or target not in text:
        return text, False

    if "." in target:
        broken = f"{float(target) * 1.85 + 3:.2f}"
    else:
        broken = str(int(target) + 47)
    return text.replace(target, broken, 1), True


def main() -> int:
    questions = [json.loads(l) for l in QUESTIONS.open() if l.strip()]
    print(f"calibrating over {len(questions)} public questions\n")

    rows, calls = [], 0
    for q in questions:
        comps = q["grading"]["components"]
        ref = q["reference_answer"]
        para = paraphrase(ref)
        # Negative arm targets component 0 specifically.
        neg, ok = perturb(ref, comps[0]["expected_fact"])

        arms = {"control": ref, "paraphrase": para}
        if ok:
            arms["negative"] = neg

        res = {}
        for arm, answer in arms.items():
            earned, total, verdicts = component_judge.score_question(
                q["prompt"], answer, comps
            )
            calls += len(verdicts)
            res[arm] = {
                "recall": (earned / total) if total else 0.0,
                "passed": sum(1 for v in verdicts if v.passed),
                "n": len(verdicts),
                # For the negative arm this is the number that matters: did the
                # component we deliberately broke actually fail?
                "target_component_passed": verdicts[0].passed if verdicts else None,
            }

        rows.append(
            {
                "id": q["id"],
                "difficulty": q["difficulty"],
                "n_components": len(comps),
                "reference_answer": ref,
                "paraphrase": para,
                "negative": neg if ok else None,
                "arms": res,
            }
        )
        c = res["control"]
        p = res["paraphrase"]
        n = res.get("negative", {})
        print(
            f"{q['id']:8} comps={len(comps)}  "
            f"control {c['passed']}/{c['n']}  "
            f"paraphrase {p['passed']}/{p['n']}  "
            f"negative {n.get('passed','-')}/{n.get('n','-')}"
        )

    def mean(arm: str, key: str = "recall") -> float:
        vals = [r["arms"][arm][key] for r in rows if arm in r["arms"]]
        return sum(vals) / len(vals) if vals else 0.0

    control, para_m, neg_m = mean("control"), mean("paraphrase"), mean("negative")
    para_full = sum(
        1 for r in rows if r["arms"]["paraphrase"]["passed"] == r["arms"]["paraphrase"]["n"]
    )

    # The decision rule, fixed in advance so the outcome is not read into.
    if para_m >= 0.90:
        verdict = "NATURAL_SYNTHESIS"
        guidance = (
            "The judge accepts paraphrase. Session D's `assistant` target is **natural "
            "synthesis** — fluent, complete answers. Sentence order and date format are "
            "free.\n\n"
            "**One caveat this arm did NOT test.** The paraphrase arm reorders whole "
            "*sentences* and never splits a sentence, so it says nothing about compound "
            "facts. The FINETUNE_PLAN §4.5 rule still stands independently: each compound "
            "`required_fact` must stay in ONE sentence carrying every sub-clause. Four "
            "questions (MHQ001, MHQ040, MHQ049, MHQ076) are a single 10-point "
            "all-or-nothing component bundling 3–4 numbers — 26.7% of public points — and "
            "splitting one across two sentences still scores zero. Natural synthesis is "
            "safe; natural *decomposition* of a compound fact is not."
        )
    elif para_m >= 0.70:
        verdict = "REFERENCE_SHAPE_PREFERRED"
        guidance = (
            "The judge is partly tolerant. Session D should generate in **reference shape "
            "by default** — one sentence per gradeable component, facts in the question's "
            "own order — since that shape passes under both outcomes."
        )
    else:
        verdict = "TEMPLATE_REPRODUCTION"
        guidance = (
            "The judge is intolerant of paraphrase. Session D's target is **near-template "
            "reproduction of `reference_answer`'s shape**. This is a materially different "
            "training set — do not generate natural synthesis."
        )

    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "judge_model": component_judge.config.EVAL_JUDGE_MODEL,
                "judge_calls": calls,
                "verdict": verdict,
                "mean_points_weighted_recall": {
                    "control": control,
                    "paraphrase": para_m,
                    "negative": neg_m,
                },
                "rows": rows,
            },
            indent=2,
        )
    )

    md = [
        "# Judge calibration",
        "",
        "**Phase 0 artifact.** Session D reads this before designing the data mix; D does",
        "not produce it. Regenerate with `.venv/bin/python -m training.eval.calibrate_judge`.",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Judge:** `{component_judge.config.EVAL_JUDGE_MODEL}` via LiteLLM, "
        "`temperature=0`, `enable_thinking=false`, one expected fact per call",
        f"- **Judge calls:** {calls}",
        "",
        f"## Verdict: `{verdict}`",
        "",
        guidance,
        "",
        "## Mean points-weighted component recall",
        "",
        "| Arm | Input | Expected | Measured |",
        "|---|---|---|---:|",
        f"| control | `reference_answer` verbatim | ~1.00 | **{control:.3f}** |",
        f"| paraphrase | facts reordered, dates reformatted, reworded | ~1.00 if tolerant | **{para_m:.3f}** |",
        f"| negative | first number perturbed beyond tolerance | low | **{neg_m:.3f}** |",
        "",
        f"Paraphrase arm scored **full marks on {para_full} of {len(rows)}** questions.",
        "",
        "## Reading the arms",
        "",
        "- **control** below ~1.00 means the judge rejects its own reference answer — a",
        "  judge-prompt problem, not a model problem. Fix `component_judge.JUDGE_SYSTEM`",
        "  before trusting anything else here.",
        "- **negative** near control means the judge is not discriminating, so component",
        "  recall is not measuring correctness. That invalidates checkpoint selection.",
        "- **paraphrase** is the one that decides D's training target.",
        "",
        "## Per question",
        "",
        "| ID | Diff | Comps | Control | Paraphrase | Negative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        a = r["arms"]
        neg = a.get("negative")
        md.append(
            f"| {r['id']} | {r['difficulty']} | {r['n_components']} | "
            f"{a['control']['passed']}/{a['control']['n']} | "
            f"{a['paraphrase']['passed']}/{a['paraphrase']['n']} | "
            + (f"{neg['passed']}/{neg['n']} |" if neg else "n/a |")
        )
    md += ["", "Full inputs and per-arm detail: `judge_calibration.json`.", ""]
    OUT_MD.write_text("\n".join(md))

    print(f"\nverdict: {verdict}")
    print(f"control {control:.3f} | paraphrase {para_m:.3f} | negative {neg_m:.3f}")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)} and {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
