# Nemotron Fine-Tuning — Execution Plan

**Date:** 2026-07-31
**Status:** design approved, not yet implemented
**Owner:** fine-tuning workstream (one agent can execute this end to end)
**Budget:** 3 hours wall clock, hard stop
**Companion doc:** `src/PLAN.md` (agent harness). This file covers only the model. The two
workstreams share exactly one contract — §4 — and are otherwise independent.

---

## 1. Mission

Produce a fine-tuned `Llama-3.1-Nemotron-Nano-8B-v1` LoRA adapter, **served and wired into
the agent**, plus the evidence needed for the 30% fine-tuned-model-quality category.

**In scope:** training data generation, LoRA training, checkpoint selection, serving the
adapter, base-vs-fine-tuned comparison, `training/` artifacts.

**Explicitly out of scope:** Nemotron never plans, never selects tools, never emits tool
calls. Qwen (`agent-brain`) owns all of that and is frozen. Nemotron's only inputs are a
question plus already-verified tool results; its only output is the final answer string.
Violating this loses architecture credit — see
`Participant_Package/handout/03_scoring_and_examples.md` § "Bad: Nemotron used as the
planner and tool caller".

### 1.1 What actually earns the 30%

Read the rubric before optimising anything
(`Participant_Package/Challenge_Brief.md` § "Fine-Tuned Model Quality"). It scores six
things, and only two are "how good is the adapter":

1. Relevance, quality, **documented preparation** of the fine-tuning data
2. Training method, config, hyperparameters, checkpoints, **model-selection rationale**
3. **Quantitative** base-vs-fine-tuned comparison on held-out examples
4. Robustness, consistency, avoidance of unsupported claims
5. Evidence the fine-tuned model is genuinely used by the submitted solution
6. Reproducibility, with no hidden evaluation data

**Therefore the objective is not "best adapter in 3 hours". It is: shortest path to a
genuinely-served adapter, then every remaining minute on evidence.** A mediocre adapter
that is live and documented outscores an excellent one that is not.

`DOMAIN_PREDICT_MODE=mock` at submission forfeits this entire 30% *and* drags the
architecture 30% down with it. Getting to `=llm` is the single highest-value outcome in
this plan.

---

## 2. Environment — verified 2026-07-31

Everything in this section was measured on the live cluster during planning. Do not
re-derive; use as fixtures. Items marked *(per `src/PLAN.md`)* were measured by an earlier
session and are trusted but not re-confirmed here.

### 2.1 Cluster

```
node0 = 10.0.1.10   hostname aitopatom-6977   ← YOU ARE HERE
  1x NVIDIA GB10, 121 GB unified RAM (73 GB used, ~48 GB free)
  docker: litellm      :4000   aliases -> agent-brain, domain-ft
          vllm-brain   :8000   Qwen/Qwen3.6-35B-A3B-Instruct-FP8, 50 GB GPU, max_model_len 4096
  no torch on the host python
  images pulled: nvcr.io/nvidia/nemo:25.09 (33.5 GB), vllm/vllm-openai:latest (24.7 GB)
  disk: 3.3 TB free
  huggingface.co reachable (HTTP 200 in 0.28 s)

node1 = 10.0.1.11
  1x NVIDIA GB10, 128 GB unified
  vllm :8001  Llama-3.1-Nemotron-Nano-8B-v1  (BASE), max_model_len 4096
  model weights already on disk at /models/Llama-3.1-Nemotron-Nano-8B-v1  <-- no download needed
  ssh from node0: Permission denied (publickey,password)   <-- BLOCKER, see §3
```

### 2.2 The `domain-ft` alias is broken, and the fix is free

```
$ curl -s localhost:4000/v1/chat/completions -d '{"model":"domain-ft",...}'
{"error":{"message":"...The model `nemotron-8b-finance` does not exist.. Received Model Group=domain-ft"}}
```

LiteLLM routes `domain-ft` to a model named **`nemotron-8b-finance`**; node1 currently
advertises `Llama-3.1-Nemotron-Nano-8B-v1`. So when serving the adapter, pass
`--served-model-name nemotron-8b-finance` and the alias starts working with **no organizer
involvement and no LiteLLM config change**. This is a gift — do not miss it.

### 2.3 Environment file

`~/team.env` does **not** exist at the documented path. The real file is:

```
~/Cognitivo_Training/AI_Training_and_Hackathon/Sample_Activity_5/p2_agent/team.env
  TEAM_ID=team-01
  NODE0_IP=10.0.1.10
  NODE1_IP=10.0.1.11
  LITELLM_BASE_URL=http://localhost:4000/v1
  BRAIN_MODEL=agent-brain
  DOMAIN_FT_MODEL=domain-ft
```

`MODELS_DIR` is unset. Never hard-code endpoints or keys in source; read from env.

### 2.4 Datasets — paths confirmed

Root: `AI_Industry_Training_Hackathon/data set/`

| Dataset | Path | Shape | Gotchas |
|---|---|---|---|
| RBA | `RBA Rates/RBA-rates.jsonl` | 175 rows, 2010-02-03 → 2026-06-17 | **UTF-8 BOM — open with `encoding='utf-8-sig'`.** Dates are `"3 Feb 2010"`, not ISO. All values are **strings**, including signed changes (`"+0.25"`, `"0.00"`) |
| ASX | `ASX/<TICKER>-ASX-2015-2021.jsonl` | 18 files × 1,774 rows = 31,932 | Tickers carry `.AX`. Tabcorp is `TAH.AX` and is excluded in 5 of 15 public questions |
| AFR | `AFR/AFR_<YYYYMMDD>-<YYYYMMDD>.jsonl` | 85 files, 219,538 articles, 780 MB *(per `src/PLAN.md`)* | `PUBLICATIONDATE` is a `YYYYMMDD` **string**. Slice `[:4]`/`[:6]`; do not date-parse |

Field schemas:

```
RBA  {"Effective Date":"3 Feb 2010","Change % points":"0.00","Cash rate target%":"3.75"}
ASX  {"ticker":"AGL.AX","date":"2015-01-02","open":..,"high":..,"low":..,"close":..,"volume":..}
AFR  {"HEADLINE":..,"SUBHEAD":..,"INTRO":..,"TEXT":..,"NEWSPAPER":..,"PUBLICATIONDATE":"20150102"}
```

**AFR search rule — reproducibility-critical, non-negotiable per `Setup_Instructions.md`:**
case-insensitive, `\bword\b` anchored, matched across `HEADLINE + SUBHEAD + INTRO + TEXT`
**combined**, counted **once per record**. A different field set silently yields different
counts that will not match reference answers.

*(per `src/PLAN.md`)* An inverted index tokenised with `[a-z0-9]+` is **exactly equivalent**
to that rule — verified: `unemployment` 5,997 / `qbe` 1,546 / `nab` 7,372, identical to a
full regex scan. Do **not** include `'` in the token class (`[a-z0-9']+` gives 6,903 for
`nab` — wrong, because `\b` treats an apostrophe as a boundary). Index costs ~32 s to build,
then answers in <1 ms; a full regex scan is ~6.5 s per pattern and `re` does not release the
GIL, so threads do not help.

### 2.5 Reference answers — generator correctness fixtures

| ID | Expected | Exercises |
|---|---|---|
| MHQ001 | 41 of 175 records changed the rate: 20 increases, 21 decreases | RBA `count_changes` / sign split |
| MHQ040 | 18 ticker files, 1,774 rows each, 2 Jan 2015 → 30 Dec 2021 | ASX `describe` |
| MHQ061 | peak year 2020 with 1,452 records; peak month May 2020 with 218 | AFR index ≡ regex |
| MHQ049 | AMP.AX highest average daily volume, 11,635,671.71 shares/day | ASX `avg_volume` + `exclude_tickers` |
| MHQ045 | BHP.AX best 2018 at +22.17%; AMP.AX worst at −50.04% | ASX `rank_annual_returns` + `exclude_tickers` |

The last two were added because `exclude_tickers` is a first-class argument on every ASX
metric (§6.2) and nothing else in the fixture set tested it. MHQ049 is also one of the four
all-or-nothing compound components in §4.5.

### 2.6 The Nemotron tokenizer is reachable over HTTP today — use it

node1's vLLM server exposes `POST /tokenize`. **This is available right now, needs no ssh,
no torch and no local weights**, so it is not blocked by B1:

```bash
curl -s http://10.0.1.11:8001/tokenize -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.1-Nemotron-Nano-8B-v1","prompt":"..."}'
# -> {"count":77,"max_model_len":4096,"tokens":[...]}
```

Verified 2026-07-31. This is what makes the §4.4 sequence budget measurable instead of
guessed, and it is the only tokenizer access available before node1 is unblocked. **It
disappears the moment training takes that GPU** — so run the §6.6 budget measurement early,
in the same window as the base-model eval.

---

## 3. Gaps and blockers

| # | Gap | Impact | Action |
|---|---|---|---|
| **B1** | **ssh to node1 denied** | Blocks the whole critical path | Chase from minute zero. Fallback in §9 |
| G1 | `~/Cognitivo_Training/finagent-finetune/` does not exist — no `01_prepare_data.py`, no `07_train_8b_quicktest.sh`, no `03_train_1node.sh` | Every command in the training guide references scripts that are not on this box | We write our own. §6 |
| G2 | `llm-eval-nim-demo/` contains **only** `01-NIM-Evaluation.ipynb` — no `03-Customizer.ipynb`, no `guide/`, no `automodel_recipes/`, no `compare_infer.py`, no `spark_finetune.py`. Not a git repo | `HANDOVER.md` and the 30 Jul spec both lean on these | Treat as unavailable. Do not plan around them |
| G3 | `nemotron-customize` skill needs a Nemotron repo checkout with `src/nemotron/steps/` — absent | Skill unusable | Ignore |
| G4 | No base-model weights and no torch on node0 | node0 fallback needs container work | §9 |

---

## 4. Contracts

### 4.1 Synthesis prompt — shared with `src/agent/synth.py`

**This is the one thing the harness workstream also depends on. Freeze it before generating
data, and make the harness read the identical strings.** If training format and inference
format drift, the adapter degrades to noise.

```python
SYNTH_SYSTEM = (
    "You are a financial data analyst. You are given a question and verified results "
    "from deterministic data tools. Write ONE concise answer that states every fact the "
    "question asks for, using the exact values from the tool results. When the question "
    "asks for several numbers that belong together, state all of them in a single "
    "sentence. Do not hedge. Do not add facts that are not in the tool results. If the "
    "tool results are insufficient to answer, say so plainly and state what is missing."
)

SYNTH_USER = "Question: {question}\n\nVerified tool results:\n{tool_results}"
```

The single-sentence clause is not stylistic — see §4.5. Measured cost: `SYNTH_SYSTEM` goes
from 77 to ~95 tokens against the real Nemotron tokenizer (§2.6). That is affordable and it
is the cheapest available defence on 26.7% of the public points.

### 4.2 Training record

JSONL, one object per line:

```json
{
  "messages": [
    {"role": "system",    "content": "<SYNTH_SYSTEM>"},
    {"role": "user",      "content": "<SYNTH_USER filled>"},
    {"role": "assistant", "content": "<the answer>"}
  ],
  "meta": {
    "id": "gen_rba_000123",
    "slice": "rba_single",
    "datasets": ["RBA"],
    "required_facts": ["41 of the 175 decision records changed the rate: 20 increases and 21 decreases."],
    "points":         [10.0],
    "tolerance": "exact"
  }
}
```

`meta` is **not** trained on. It exists so the evaluator in §7 can score component recall.

**`required_facts` granularity is a contract, not a convention.** Measured over all fifteen
public questions: `required_facts` is **exactly 1:1 with `grading.components`** in every
single one. One string per gradeable component — never finer.

An earlier draft of this section split MHQ001 into three strings
(`"…changed the rate"`, `"20 increases"`, `"21 decreases"`). That is wrong twice over:

1. It teaches the generator to emit three separate facts where the grader wants one bundled
   sentence, which scores **zero** on MHQ001 (§4.5).
2. It makes the §7 component-recall metric finer-grained than the real judge, so it would
   report 2-of-3 as 67% where the grader reports 0%. **The headline metric would flatter the
   adapter and hide the exact failure that costs the most points.**

Carry `points` alongside so the evaluator can weight recall the way the grader does, rather
than treating a 10-point compound and a 2-point sentiment clause as equal.

### 4.5 Compound components — the constraint that decides real points

Four of the fifteen public questions are a **single 10-point all-or-nothing component** whose
`expected_fact` bundles three or four numbers. **26.7% of public points sit behind four
YES/NO gates where three of four numbers right scores zero:**

| ID | The one component |
|---|---|
| MHQ001 | `41 of the 175 decision records changed the rate: 20 increases and 21 decreases.` |
| MHQ040 | `There are 18 ticker files, each containing 1,774 rows, covering 2 Jan 2015 through 30 Dec 2021.` |
| MHQ049 | `AMP.AX has the highest average daily volume at 11,635,671.71 shares per trading day.` |
| MHQ076 | `There are 369 AFR records matching whole-word QBE in 2021, and QBE.AX had the best non-Tabcorp 2021 return at +35.57%.` |

**Every compound `required_fact` in the training data must therefore be one sentence carrying
every sub-clause, in the reference's own shape.** Splitting a compound fact across two
sentences is how a perfectly-computed answer scores zero. This binds §4.1 (the prompt), §4.2
(the record), §6 (generation) and §7 (scoring) simultaneously — it is the single most
load-bearing constraint in this plan.

### 4.3 Loss masking — do not skip

Train on **assistant tokens only**. Use TRL's completion-only collator (or
`assistant_only_loss=True`). Without masking, the model learns to generate tool results as
well as answers, which is the opposite of what we want.

### 4.4 Sequence budget

`max_seq_len = 512`. Build every record so the **assistant answer is never truncated** —
cap `tool_results` at ~1,200 characters and truncate from the tool-results side only. A
truncated answer teaches the model to emit nothing.

The harness must clamp tool results to the same cap at inference time (`src/PLAN.md` §6
currently says 2,000 chars — **reconcile these two numbers before training**).

---

## 5. Strategy and rationale

Two calls depart from the handout. Both are deliberate; record the reasoning in
`training/MODEL_SUMMARY.md` because the rubric scores rationale.

### 5.1 Use HF `peft` + `trl`, not the NeMo container

The handout's own step 4 exports an `hf_adapter` directory and notes "vLLM loads the
adapter at runtime. No weight merge required." NeMo is being used to produce a **standard
PEFT adapter** — which is exactly what ~60 lines of `SFTTrainer` produces directly.

Writing a first-ever NeMo recipe on aarch64 GB10, with no working smoke-test precedent and
with the `finagent-finetune/` scripts absent (G1), is the largest schedule risk in a 3-hour
budget, and it buys nothing vLLM needs. The handout's *hyperparameters* still bind; only
the harness changes. `nemo:25.09` is the documented fallback (§9), not the primary.

### 5.2 Bank the base-model evaluation before touching node1

`10.0.1.11:8001` serves base Nemotron **right now**. The moment training starts, that GPU
is ours and the control arm disappears. Base numbers are free and available today — collect
them in the first hour or lose the comparison that is worth 30%.

### 5.3 Nemotron is a discipline model, not a knowledge model

It never retrieves or calculates. What it must learn is behavioural:

1. Restate **every** requested component exactly — the judge checks facts one at a time
2. Zero hedging — "approximately 41" is explicitly scored wrong
3. Add nothing unsupported — no invented drivers or forecasts
4. State insufficiency when coverage is missing (MHQ090's correct answer is a justified
   refusal: RBA covers the 2022–23 hikes, but ASX and AFR both end in 2021)
5. The sentiment path — AFR text + applicable RBA rate → sentiment + likely direction, and
   explicitly **no** fabricated numeric forecast (`Setup_Instructions.md` L95)

This is learnable from a few thousand templated examples, which is why a short run can
genuinely beat base. Base Nemotron's documented failure mode is thinking out loud and
hedging — precisely what this trains out.

### 5.4 Step count is measured, not assumed

The handout's ~90 s/step (100 steps ≈ 2–3 h) is far slower than an 8B LoRA at seq 512
should run on a GB10. The smoke test measures the real rate; `MAX_STEPS` is then set to
whatever finishes by T+2:00. At 5 s/step we get hundreds of steps and a real ablation; at
90 s/step we take the step-20 checkpoint and ship. Either way we know at minute 50, not
minute 150.

---

## 6. Data generation

Target **3,000–5,000 examples**. Volume is not the constraint; coverage and phrasing
variety are.

### 6.1 Mix

| Slice | Share | Teaches |
|---|---:|---|
| `*_single` — RBA / ASX / AFR single-dataset synthesis | 45% | state every component exactly |
| `cross` — two or three datasets combined | 25% | multi-source composition |
| `sentiment` — AFR article text + applicable RBA rate | 12% | sentiment + direction, **no** numeric forecast |
| `insufficient` — question outside coverage | 10% | justified refusal (MHQ090 shape) |
| `robust` — tool error, empty result, more facts supplied than asked | 8% | don't hedge, don't pad |

### 6.2 Metric coverage

Derived from the real 15-question bank. Items marked ★ are required by the bank but absent
from the execution guide's metric list — easy to miss.

- **RBA** — `count`, `count_changes`, `count_increases`, `count_decreases`, `extremes`,
  `max_hold_streak`, `lookup_rate` (**as-of**: rate in effect on-or-before a date, never
  nearest-match, which can return a future decision), `cycle_summary` ★ (cumulative change
  over a date range)
- **ASX** — `annual_return`, `rank_annual_returns`, `full_sample_return`, `volatility`,
  `correlation`, `max_drawdown` (+ peak/trough dates), `avg_volume`, `describe` ★
  (file/row counts + date range, for MHQ040), `window_return` ★ (N-day return from an RBA
  decision date, incl. equal-weighted baskets). `exclude_tickers` is a first-class argument
  on **every** ASX metric
- **AFR** — `count`, `count_by_month`, `share`, `retrieve_by_headline` ★ (exact headline +
  publication date)
- **coverage** ★ — compares date ranges across datasets; feeds the `insufficient` slice

The combinatorics are generous: 18 tickers × 7 years for annual returns alone, plus
arbitrary RBA date ranges for `cycle_summary`. Reaching 5,000 examples is easy.

### 6.3 Three design rules that matter more than volume

1. **3–5 phrasing variants per metric type, randomised.** A single fixed template teaches
   the template, not the discipline, and collapses on unseen question shapes.
2. **Include cases where tool results contain more facts than the question asks for**, with
   answers stating only what was asked. The brief is explicit: "The grader requires only
   information requested by the prompt." Over-answering buries components.
3. **Vary numeric formatting in the tool results** (commas, trailing zeros, ISO vs
   `3 Feb 2010` dates) while keeping answers exact. Teaches robustness to the harness's
   formatting.

### 6.4 Splits

`train` / `val` / `heldout` = 80 / 10 / 10, **split by metric-and-entity key, not by row**,
so no ticker-year or RBA-range appears in two splits. Cross-split leakage inflates the
comparison and is exactly what the rubric means by "must not contain hidden evaluation
data". Assert zero overlap and log the assertion.

### 6.5 Correctness gate — blocking

The generator must reproduce all three fixtures in §2.5 before any training starts. If they
do not match, the training data is wrong and nothing downstream matters. Fail loudly.

---

## 7. Evaluation design

**Do not use BLEU/ROUGE.** They measure n-gram overlap, not whether a required fact is
present, and the only local precedent is inherited from a legal-title-generation task. The
companion review (`Cognitivo_Labs/.../reviews/2026-07-31-evaluation-strategy-review.md` §5)
makes this case at length.

Instead, replicate the organizer's judge mechanically:

| Metric | Definition |
|---|---|
| **Component recall** | over all `meta.required_facts` in heldout: fraction present in the generated answer. **The headline number.** |
| **All-components rate** | fraction of examples where *every* required fact is present |
| **Hedge rate** | fraction of answers matching `\b(approximately|roughly|about|around|~)\b` before a number — explicitly scored wrong |
| **Unsupported-numeric rate** | fraction of answers containing a number absent from the tool results — the hallucination proxy |
| **Mean answer length** | tokens; base is expected to ramble, FT to be terse |

Fact matching: dates and counts exact (accept equivalent date formats and comma/trailing-zero
variants); returns/drawdowns/volatility ±0.02 pp; correlations ±0.001; quoted closes ±0.0001;
average volume ±1 share. These mirror the tolerance tiers in the public question bank.

**Both arms use identical prompts and identical decoding** (`temperature=0`,
`max_tokens=256`). The only variable is the adapter. Anything else is not a controlled
comparison and the judges will say so.

---

## 8. Timeline

`T+0:00` is when execution starts. Times are budgets, not estimates — hit the gate or take
the fallback.

| Window | Work | Gate to pass |
|---|---|---|
| 0:00–0:15 | **Parallel.** (a) resolve B1 node1 access; confirm `/models` path, container, torch/peft availability, internet on node1. (b) start the generator on node0 | node1 reachable **or** fallback chosen |
| 0:15–0:50 | Generator → `train/val/heldout.jsonl` | **§6.5 fixtures reproduce** |
| 0:50–1:05 | **Bank base-model eval** against live `10.0.1.11:8001` ‖ smoke test on node1 (10 steps) | base metrics written to disk; **step time measured** |
| 1:05–1:10 | Set `MAX_STEPS` so training ends by 2:00; scale warmup (§8.1) | — |
| 1:10–2:00 | Train. Checkpoint every ~10% of the run | ≥1 checkpoint on disk |
| 2:00–2:20 | Serve best checkpoint on node1:8001 as `nemotron-8b-finance`; verify `domain-ft` returns 200 | alias healthy |
| 2:20–2:40 | FT eval, same heldout, same decoding → comparison table | — |
| 2:40–3:00 | `training/` artifacts; `submission.json` model block; `DOMAIN_PREDICT_MODE=llm` | — |

**Hard rules**

- Whatever checkpoint exists at **T+2:00** is the one that ships. No exceptions.
- Base eval is banked **before** node1's GPU is disturbed. Non-negotiable ordering.
- If the smoke test has not passed by **T+1:10**, switch to the §9 fallback immediately
  rather than debugging.

### 8.1 Hyperparameters

Handout baseline, with one correction.

```
base_model      = /models/Llama-3.1-Nemotron-Nano-8B-v1
lora_r          = 32
lora_alpha      = 64
lora_dropout    = 0.05
target_modules  = all *_proj
learning_rate   = 5e-5        # NOT 1e-4 — handout confirms a loss spike at warmup step 50
max_seq_len     = 512
per_device_bs   = 2
grad_accum      = 4           # effective batch 8
lr_scheduler    = cosine
warmup_ratio    = 0.10        # <-- THE CORRECTION, see below
max_steps       = <measured>  # §5.4
save_steps      = max(5, max_steps // 10)
bf16            = True
seed            = 42
```

**The warmup correction.** The handout's `WARMUP_STEPS=50` assumes a 100-step run. If we
run 60 steps with a fixed 50-step warmup, the LR never reaches target and we have trained on
essentially nothing. Use a **ratio**, not a constant. This is the most likely silent failure
in the whole plan.

**The learning-rate trap.** NVIDIA's own deck
(`Evaluation-and-Light-Customization-NVIDIA.pdf` p.23) advises PEFT LR "often higher than
SFT, e.g. 1e-4 to 1e-3". The handout warns in bold that **1e-4 causes a loss spike at warmup
step 50 on this exact model and hardware**. The handout wins. Do not follow the deck here.

### 8.2 Checkpoint selection

Select on **val component recall**, not val loss. Loss rewards mimicking phrasing; recall
rewards stating facts, which is what is scored. Record the rationale — the rubric asks for
it explicitly. Log every checkpoint's numbers so the choice is auditable.

The handout reports the step-20 checkpoint already beating base (val loss 0.098) and
performing comparably to the final one, so evaluate early checkpoints rather than assuming
more steps are better.

---

## 9. Fallback decision tree

```
B1 node1 access unresolved by T+0:20
  -> train on node0 inside vllm/vllm-openai (already pulled, ~48 GB RAM free)
     reduce: per_device_bs 1, max_seq_len 384, grad_accum 8
     DO NOT disturb the vllm-brain container — the harness workstream depends on it
     node1:8001 stays up, which keeps the base control arm alive for free

TRL smoke test fails by T+1:10
  -> switch to nvcr.io/nvidia/nemo:25.09 (pulled; 25.04 is known to crash on GB10)
     accept a shorter run; the artifact still has to be a PEFT-format adapter

Training crashes with no checkpoint on disk
  -> restart with max_steps = 20 and save_steps = 5, nothing else changed
     20 steps is the handout's own "already meaningfully better than base" mark

Cannot serve on node1:8001 by T+2:20
  -> serve on node0:8002 and update the LiteLLM domain-ft target
     if LiteLLM cannot be changed, point DOMAIN_FT_MODEL directly at the vLLM endpoint
     never ship with DOMAIN_PREDICT_MODE=mock
```

**Cut order if the schedule slips:** cross-dataset slice down to 10% → the ablation →
hedge/unsupported-numeric metrics.
**Never cut:** the served adapter, the base-vs-FT table, the §6.5 correctness gate.

---

## 10. Serving

```bash
# on node1, after stopping the base server
vllm serve /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --served-model-name nemotron-8b-finance \
  --enable-lora \
  --lora-modules domain-ft=/path/to/adapter \
  --max-model-len 4096 \
  --port 8001
```

`--served-model-name nemotron-8b-finance` is what repairs the `domain-ft` alias (§2.2).
Verify before declaring done:

```bash
curl -s localhost:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"domain-ft","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":5}'
```

Must return 200 with content, not the `nemotron-8b-finance does not exist` 404.

Then set `DOMAIN_PREDICT_MODE=llm` and confirm the agent actually routes through it. The
harness must also survive **three concurrent `/query` requests** — vLLM handles concurrency,
but confirm the adapter is loaded once and shared, not per-request.

---

## 11. Deliverables

Rubric-mandated home is `training/` (`Challenge_Brief.md` § Required Deliverables). This
plan lives in `src/` by request; the artifacts below still go to `training/`.

- [ ] `training/prepare_data.py` — the generator, with the §6.5 fixture assertions inline
- [ ] `training/train_lora.py` — TRL/PEFT training script
- [ ] `training/eval_compare.py` — the §7 evaluator, both arms
- [ ] `training/config.yaml` — every hyperparameter, including the measured `max_steps`
- [ ] `training/data/{train,val,heldout}.jsonl` — or a documented regeneration command if
      too large to commit
- [ ] `training/metrics/base.json`, `training/metrics/finetuned.json`
- [ ] `training/COMPARISON.md` — the base-vs-FT table, with 2–3 side-by-side answer samples
- [ ] `training/MODEL_SUMMARY.md` — data prep method, config, **checkpoint-selection
      rationale**, the two §5 deviations and why, known limitations
- [ ] `logs/train.log`, `logs/eval.log` — non-sensitive
- [ ] `submission.json` → `model.model_name = "nemotron-8b-finance"`, `model.endpoint`
      pointing at node1:8001 (**not** `localhost`)
- [ ] `DOMAIN_PREDICT_MODE=llm` confirmed in the running agent

**Security:** no credentials, keys, or hidden evaluation data in any committed file. Scan
before the final commit.

---

## 12. Open questions

1. **B1** — who owns node1 access, and how fast can it land? Everything queues behind this.
2. Does node1 have internet for `pip install peft trl datasets accelerate`? If not, install
   on node0 and mount, or use `nemo:25.09` which already bundles them.
3. Reconcile the tool-result character cap: this plan says ~1,200, `src/PLAN.md` §6 says
   2,000. **Training and inference must agree** — pick one before generating data.
4. Should the sentiment path share one adapter with synthesis (assumed yes, 12% of the mix)
   or be a second adapter? One adapter is assumed for time; note the assumption in
   `MODEL_SUMMARY.md`.
5. `src/PLAN.md` §11 says the harness may still be mid-build. If `src/agent/synth.py` does
   not exist at T+2:20, the FT eval still runs standalone against the served endpoint — this
   workstream is not blocked by the harness.
