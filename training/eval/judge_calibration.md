# Judge calibration

**Phase 0 artifact.** Session D reads this before designing the data mix; D does
not produce it. Regenerate with `.venv/bin/python -m training.eval.calibrate_judge`.

- **Generated:** 2026-07-31T02:15:52.041710+00:00
- **Judge:** `agent-brain` via LiteLLM, `temperature=0`, `enable_thinking=false`, one expected fact per call
- **Judge calls:** 108

## Verdict: `NATURAL_SYNTHESIS`

The judge accepts paraphrase. Session D's `assistant` target is **natural synthesis** —
fluent, complete answers. Sentence order and date format are free.

**One caveat this arm did NOT test.** The paraphrase arm reorders whole *sentences* and never
splits one, so it says nothing about compound facts. The `FINETUNE_PLAN.md` §4.5 rule stands
independently: each compound `required_fact` must stay in **one** sentence carrying every
sub-clause. Four questions (MHQ001, MHQ040, MHQ049, MHQ076) are a single 10-point
all-or-nothing component bundling 3–4 numbers — 26.7% of public points — and splitting one
across two sentences still scores zero. **Natural synthesis is safe; natural *decomposition*
of a compound fact is not.**

## Mean points-weighted component recall

| Arm | Input | Expected | Measured |
|---|---|---|---:|
| control | `reference_answer` verbatim | ~1.00 | **1.000** |
| paraphrase | sentences reordered, dates reformatted, reworded | ~1.00 if tolerant | **1.000** |
| negative | one number in component 0 perturbed beyond tolerance | low | **0.482** |

Paraphrase arm scored **full marks on 15 of 15** questions. In the negative arm the
deliberately-broken component was correctly judged NO in **14 of 14** questions where a
perturbation was possible (MHQ090's refusal has no perturbable number), so the judge
discriminates rather than rubber-stamping.

### Three earlier runs disagreed — read this before trusting a re-run

This measurement reported `TEMPLATE_REPRODUCTION` (paraphrase 0.623) and then
`REFERENCE_SHAPE_PREFERRED` (0.823) before landing here. **All three earlier verdicts were
bugs in the calibration harness, not judge behaviour:**

1. Thousands separators were applied to bare 4-digit integers, turning the *year* `2015` into
   `2,015` — a fact corruption. The judge was right to reject it.
2. The leading character was lower-cased to fit a prose prefix, turning the ticker `AMP.AX`
   into `aMP.AX`. That alone zeroed MHQ049's single 10-point component.
3. Clauses were reordered at commas, splitting compound sentences mid-fact — which measures
   the §4.5 compound rule, not paraphrase tolerance.

The lesson generalises to D's generator: **a paraphrase that touches a number, a ticker, or
the interior of a compound fact is not a paraphrase, it is a wrong answer.** If a re-run
disagrees with this file, suspect the transform before the judge.

## Reading the arms

- **control** below ~1.00 means the judge rejects its own reference answer — a
  judge-prompt problem, not a model problem. Fix `component_judge.JUDGE_SYSTEM`
  before trusting anything else here.
- **negative** near control means the judge is not discriminating, so component
  recall is not measuring correctness. That invalidates checkpoint selection.
- **paraphrase** is the one that decides D's training target.

## Per question

| ID | Diff | Comps | Control | Paraphrase | Negative |
|---|---|---:|---:|---:|---:|
| MHQ001 | easy | 1 | 1/1 | 1/1 | 0/1 |
| MHQ035 | medium | 2 | 2/2 | 2/2 | 1/2 |
| MHQ040 | easy | 1 | 1/1 | 1/1 | 0/1 |
| MHQ045 | medium | 2 | 2/2 | 2/2 | 1/2 |
| MHQ049 | medium | 1 | 1/1 | 1/1 | 0/1 |
| MHQ055 | hard | 3 | 3/3 | 3/3 | 2/3 |
| MHQ058 | easy | 3 | 3/3 | 3/3 | 2/3 |
| MHQ061 | medium | 2 | 2/2 | 2/2 | 1/2 |
| MHQ067 | hard | 3 | 3/3 | 3/3 | 2/3 |
| MHQ072 | medium | 3 | 3/3 | 3/3 | 2/3 |
| MHQ074 | hard | 4 | 4/4 | 4/4 | 3/4 |
| MHQ076 | easy | 1 | 1/1 | 1/1 | 0/1 |
| MHQ080 | medium | 5 | 5/5 | 5/5 | 4/5 |
| MHQ084 | medium | 3 | 3/3 | 3/3 | 2/3 |
| MHQ090 | hard | 3 | 3/3 | 3/3 | n/a |

Full inputs and per-arm detail: `judge_calibration.json`.
