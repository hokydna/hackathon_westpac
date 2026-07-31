# Judge calibration

**Phase 0 artifact.** Session D reads this before designing the data mix; D does
not produce it. Regenerate with `.venv/bin/python -m training.eval.calibrate_judge`.

- **Generated:** 2026-07-31T02:14:07.154088+00:00
- **Judge:** `agent-brain` via LiteLLM, `temperature=0`, `enable_thinking=false`, one expected fact per call
- **Judge calls:** 108

## Verdict: `REFERENCE_SHAPE_PREFERRED`

The judge is partly tolerant. Session D should generate in **reference shape by default** — one sentence per gradeable component, facts in the question's own order — since that shape passes under both outcomes.

## Mean points-weighted component recall

| Arm | Input | Expected | Measured |
|---|---|---|---:|
| control | `reference_answer` verbatim | ~1.00 | **1.000** |
| paraphrase | facts reordered, dates reformatted, reworded | ~1.00 if tolerant | **0.823** |
| negative | first number perturbed beyond tolerance | low | **0.482** |

Paraphrase arm scored **full marks on 10 of 15** questions.

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
| MHQ035 | medium | 2 | 2/2 | 1/2 | 1/2 |
| MHQ040 | easy | 1 | 1/1 | 1/1 | 0/1 |
| MHQ045 | medium | 2 | 2/2 | 1/2 | 1/2 |
| MHQ049 | medium | 1 | 1/1 | 0/1 | 0/1 |
| MHQ055 | hard | 3 | 3/3 | 3/3 | 2/3 |
| MHQ058 | easy | 3 | 3/3 | 3/3 | 2/3 |
| MHQ061 | medium | 2 | 2/2 | 2/2 | 1/2 |
| MHQ067 | hard | 3 | 3/3 | 2/3 | 2/3 |
| MHQ072 | medium | 3 | 3/3 | 3/3 | 2/3 |
| MHQ074 | hard | 4 | 4/4 | 3/4 | 3/4 |
| MHQ076 | easy | 1 | 1/1 | 1/1 | 0/1 |
| MHQ080 | medium | 5 | 5/5 | 5/5 | 4/5 |
| MHQ084 | medium | 3 | 3/3 | 3/3 | 2/3 |
| MHQ090 | hard | 3 | 3/3 | 3/3 | n/a |

Full inputs and per-arm detail: `judge_calibration.json`.
