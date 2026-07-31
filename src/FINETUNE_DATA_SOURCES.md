# Fine-Tuning Data — What to Build Now vs. What Waits on Qwen

**Date:** 2026-07-31
**Scope:** the *data* for the Nemotron LoRA adapter only. Training config, serving and the
timeline live in [FINETUNE_PLAN.md](FINETUNE_PLAN.md); this file answers one question —
*for each training record, where does each field come from, and can I produce it right now?*

> **[SESSION_KICKOFF.md](SESSION_KICKOFF.md) is the source of truth** (§2 defects F1–F8, §9
> resolutions F9–F16). This file is subordinate. Two corrections applied: the record in §1 now
> carries **`points`**, without which the headline metric cannot be points-weighted; and **Q1
> judge calibration is Phase 0's, not session D's** — you read its result, you do not run it.

---

## 0. The one-paragraph answer

**~85% of the training set is deterministic and can be generated today with nothing but
Python and the three corpora.** Every numeric slice — RBA counts and cycles, ASX returns,
drawdowns, volumes, correlations, AFR whole-word counts, coverage refusals — has a computable
gold answer, so the `assistant` target is a template over numbers you calculate yourself.

**Three things genuinely require Qwen (`agent-brain`) output, and only one of them blocks
generation:**

| # | Needs Qwen | Blocks? | Why |
|---|---|---|---|
| **Q1** | **Judge calibration** — does the component judge accept paraphrase, or only near-verbatim reference shape? **Run in Phase 0, before session D forks (F6) — D reads the result, D does not run it.** | **YES — blocks answer-style choice for the whole set** | Decides whether the `assistant` target is natural synthesis or template reproduction. Changing it later means regenerating everything. ~120 calls, minutes. |
| **Q2** | **Sentiment + direction labels** for the `sentiment` slice | Blocks that slice only (12%) | No ground truth exists in the corpus. Sentiment is a judgement; you need a teacher or hand labels. |
| **Q3** | **Real tool-call traces** — the actual shape/ordering/noise of `tool_results` at inference | Blocks fidelity, not generation | Train on a `tool_results` format that differs from what the live harness feeds Nemotron and the adapter degrades. Start with synthetic, resample once traces exist. |

Everything else — all of §2 below — starts now. **Do not idle waiting on Q1/Q2.**

**Endpoint status, probed 2026-07-31:** LiteLLM `:4000` → 200, `vllm-brain` `:8000` → 200,
node1 base Nemotron `10.0.1.11:8001` → 200. Qwen is **available now**, so Q1 and Q2 are work
items to schedule, not external blockers. The only hard blocker remains **B1** (ssh to node1),
which affects *training*, not *data*.

---

## 1. What one training record is

Frozen contract, from `FINETUNE_PLAN.md` §4.2. Four fields, four different provenances:

```json
{
  "messages": [
    {"role": "system",    "content": "<SYNTH_SYSTEM>"},
    {"role": "user",      "content": "Question: {question}\n\nVerified tool results:\n{tool_results}"},
    {"role": "assistant", "content": "<the answer — THE ONLY TRAINED TOKENS>"}
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

| Field | Provenance | Available now? |
|---|---|---|
| `system` | Verbatim from `src/prompts.py` (`SYNTH_SYSTEM`) | Yes — **frozen in the base commit, never edit** (F1) |
| `user` → `{question}` | Your generator's phrasing templates | Yes |
| `user` → `{tool_results}` | Deterministic computation over the corpora, serialised in the tool layer's format, **capped at 1,200 chars** (`config.TOOL_RESULT_CHAR_CAP`, F2) | Yes (synthetic) / refine with Q3 |
| `assistant` | Template over the computed numbers | Yes — **style depends on Q1** |
| `meta.required_facts` | The same computed numbers, one string per gradeable component — **1:1 with `grading.components`, never finer** | Yes |
| `meta.points` | Points that component is worth, parallel array to `required_facts` | Yes |

**`points` is mandatory, not optional.** `FINETUNE_PLAN.md` §4.2 carries it and an earlier
draft of this file omitted it. Component recall is **points-weighted** —
`sum(points where YES) / sum(max_points)` — because granularity varies from one 10-point
compound (MHQ001) to five components worth 1–3 each (MHQ080). Without `points` the headline
metric silently flatters the adapter exactly where the points actually are.

Two rules that decide whether any of this works:

- **Loss masking — assistant tokens only.** TRL completion-only collator or
  `assistant_only_loss=True`. Without it the model learns to emit tool results too.
- **`max_seq_len = 512`, `tool_results` capped at 1,200 chars, truncating from the
  tool-results side only.** A truncated assistant answer teaches the model to emit nothing.
  Same cap at inference (`config.TOOL_RESULT_CHAR_CAP = 1200`, F2).

---

## 2. Tier A — build now, zero dependencies

These slices need only the corpora. Every gold answer is a calculation you own end to end.

### 2.1 Raw material actually on disk

| Corpus | Path | Shape | Gotchas |
|---|---|---|---|
| RBA | `data set/RBA Rates/RBA-rates.jsonl` | 175 rows, 2010-02-03 → 2026-06-17 | **UTF-8 BOM → `encoding='utf-8-sig'`.** Dates are `"3 Feb 2010"`. All values are **strings**, incl. signed changes (`"+0.25"`, `"0.00"`) |
| ASX | `data set/ASX/<Name>-ASX-2015-2021.jsonl` | 18 files × 1,774 rows = 31,932 | Tickers carry `.AX`. Tabcorp = `TAH.AX`, excluded in 5 of 15 public questions |
| AFR | `data set/AFR/AFR_<YYYYMMDD>-<YYYYMMDD>.jsonl` | 85 files, 219,538 articles, 780 MB | `PUBLICATIONDATE` is a `YYYYMMDD` **string** — slice `[:4]`/`[:6]`, never date-parse |

**AFR search rule — reproducibility-critical.** Case-insensitive, `\bword\b` anchored, matched
across `HEADLINE + SUBHEAD + INTRO + TEXT` **combined**, counted **once per record**. An
inverted index tokenised `[a-z0-9]+` is exactly equivalent (verified: `unemployment` 5,997 /
`qbe` 1,546 / `nab` 7,372). Do **not** include the apostrophe in the token class — `[a-z0-9']+`
gives 6,903 for `nab`, which is wrong.

### 2.2 The slices and where each gold answer comes from

| Slice | Share | Gold answer derived from | Combinatoric supply |
|---|---:|---|---|
| `rba_single` | ~15% | Count / sum / as-of lookup over 175 rows | 41 change dates × arbitrary date ranges → thousands |
| `asx_single` | ~20% | First-to-last returns, running-peak drawdown, volume means, stdev, Pearson correlation | 18 tickers × 7 years = 126 annual returns, plus pairwise correlations (153 pairs) and full-sample metrics |
| `afr_single` | ~10% | Inverted-index counts, by-year / by-month aggregation, share-of-total, exact headline retrieval | Any vocabulary term × 7 years × 12 months |
| `cross` | 25% | Compose two or three of the above (e.g. RBA cut date → N-day basket return) | 41 RBA dates × window lengths × ticker subsets |
| `insufficient` | 10% | Date-range comparison across corpora — pure metadata | RBA runs to 2026; ASX and AFR stop at 2021. Every post-2021 join is a legitimate refusal |
| `robust` | 8% | Deliberately malformed / over-supplied tool results, answer states only what was asked | Perturb any Tier-A record |

That is **88% of the mix, all Tier A.** Target 3,000–5,000 examples; supply is not the
constraint, phrasing variety is.

### 2.3 Metric coverage the generator must implement

Derived from the real 15-question bank. ★ = required by the bank but **absent from the
handout's metric list** — easy to miss.

- **RBA** — `count`, `count_changes`, `count_increases`, `count_decreases`, `extremes`,
  `max_hold_streak`, `lookup_rate` (**as-of**: rate in effect on-or-before a date — nearest-match
  can return a *future* decision and is wrong), `cycle_summary` ★ (cumulative change over a range,
  MHQ035/MHQ084 shape)
- **ASX** — `annual_return`, `rank_annual_returns`, `full_sample_return`, `volatility`,
  `correlation`, `max_drawdown` (+ peak/trough dates), `avg_volume`, `describe` ★ (file/row
  counts + date range, MHQ040), `window_return` ★ (N-day return from an RBA decision date, incl.
  equal-weighted baskets, MHQ072/MHQ074). `exclude_tickers` is a first-class argument on
  **every** ASX metric
- **AFR** — `count`, `count_by_month`, `share`, `retrieve_by_headline` ★ (exact headline +
  publication date — the entry point for all three sentiment questions)
- **coverage** ★ — cross-corpus date-range comparison; feeds `insufficient`

### 2.4 Correctness gate — blocking, run before any training

The generator must reproduce these three exactly. If it doesn't, the training data is wrong and
nothing downstream matters. **Fail loudly.**

| ID | Expected |
|---|---|
| MHQ001 | 41 of 175 records changed the rate: 20 increases, 21 decreases |
| MHQ040 | 18 ticker files, 1,774 rows each, 2 Jan 2015 → 30 Dec 2021 |
| MHQ061 | peak year 2020 with 1,452 records; peak month May 2020 with 218 |

### 2.5 Three design rules that matter more than volume

1. **3–5 phrasing variants per metric type, randomised.** One fixed template teaches the
   template, not the discipline, and collapses on unseen question shapes.
2. **Include records where `tool_results` contain more facts than the question asks for**, with
   answers stating only what was asked. The brief: *"The grader requires only information
   requested by the prompt."* Over-answering buries components.
3. **Vary numeric formatting inside `tool_results`** (commas, trailing zeros, ISO vs
   `3 Feb 2010`) while keeping answers exact. Teaches robustness to harness formatting.

### 2.6 Splits

`train` / `val` / `heldout` = 80 / 10 / 10, **split by metric-and-entity key, not by row**, so no
ticker-year or RBA date range appears in two splits. Assert zero overlap and log the assertion —
cross-split leakage is precisely what the rubric means by *"must not contain hidden evaluation
data."*

---

## 3. Tier B — what actually waits on Qwen

### Q1 — Judge calibration *(blocks answer style for the entire set — **owned by Phase 0**)*

> **Not session D's task.** This runs in the Phase 0 window before any session forks, because
> its result decides the answer style of every record D generates and D cannot afford to wait
> on it or to guess. D's first act is to **read** `training/eval/judge_calibration.md`; if it
> is absent, D stops and escalates rather than generating against a guess (kickoff F6).

**The question:** the component judge asks Qwen a YES/NO per `expected_fact`. Does it accept a
*paraphrase* with reordered facts and reformatted dates, or does it only accept something close
to the reference's own wording?

**Why it blocks:** it changes what the `assistant` target should be.

- Paraphrase arm passes → target is **natural synthesis**. Train for fluent, complete answers.
- Paraphrase arm fails → target is **near-template reproduction of `reference_answer`'s shape**.
  A materially different training set.

Deciding this *after* generating 5,000 records means regenerating 5,000 records.

**How to run it** — three arms over the 15 public questions, via the frozen
`src/eval/component_judge.py` against `agent-brain`:

| Arm | Input | Expected |
|---|---|---|
| Control | `reference_answer` verbatim | all YES |
| Negative | one number perturbed beyond tolerance | that component NO |
| Paraphrase | same facts, reordered, dates reformatted (`5 Jun 2019` → `2019-06-05`), different wording | all YES |

Write the result to `training/eval/judge_calibration.md`. Cost: ~120 calls, minutes.

**Cheap insurance if you want to start before the answer lands:** generate the `assistant`
target in **reference shape by default** — one sentence per gradeable component, facts in the
question's own order. That shape passes under *both* outcomes. Natural synthesis only passes
under one.

### Q2 — Sentiment and direction labels *(blocks the 12% sentiment slice only)*

Three of the fifteen public questions (MHQ058, MHQ067, MHQ080) are article-grounded sentiment.
Their components decompose like this:

```
MHQ058  C01  2 pts  "The RBA cash-rate target in force was 0.10%."          <- Tier A, deterministic
        C02  4 pts  "The article's sentiment is positive."                   <- judgement, needs a label
        C03  4 pts  "The likely direction for ASX travel shares is upward."  <- judgement, needs a label
```

**Note the split: the deterministic rate lookup is already 20% of that question's points and is
pure Tier A.** Build the rate-lookup half now; only the two judgement components wait.

**There is no ground truth for sentiment in the corpus.** Options, best first:

1. **Qwen as teacher.** Prompt `agent-brain` with `HEADLINE + INTRO + TEXT` + the as-of RBA rate,
   force the three-sentence output shape, sample a few hundred articles. Cheap, available now,
   consistent. Its labels are a *teacher signal*, not truth — say so in `MODEL_SUMMARY.md`.
2. **Hand-label 30–50** across clearly-positive / clearly-negative / genuinely-mixed articles.
   Slow, but higher quality and it gives you a check set for arm 1.
3. **Bootstrap from the three reference answers.** They pin the exact output shape and vocabulary
   (`positive` / `mixed with a negative bias`; `upward` / `mixed-to-down, with rate-sensitive
   shares under pressure`). Too few to train on, essential as style anchors.

Do 1 + 3, and 2 if minutes allow.

**The behaviour that must be trained in, and is easy to lose:** sentiment + direction, and
**explicitly no fabricated numeric forecast**. MHQ067's reference hedges the *direction*
(`mixed-to-down`) while never inventing a number. Every teacher-labelled record must obey this or
the adapter learns to make up percentages.

**Teacher prompt to fire at Qwen (`chat_template_kwargs={"enable_thinking": false}`):**

```
Article headline: {headline}
Published: {date}
Text: {intro_and_text_first_1200_chars}
RBA cash-rate target in force on that date: {rate}%

In exactly three sentences and nothing else:
1. State the RBA cash-rate target in force.
2. Classify the article's sentiment (positive / negative / mixed, optionally with a bias).
3. State the likely direction for the relevant ASX shares.
Do NOT forecast any numeric value. Do NOT add a fourth sentence.
```

### Q3 — Real tool-call traces *(affects fidelity, not whether you can generate)*

`{tool_results}` at inference is whatever the harness produces after executing **Qwen's chosen
tool calls**. Three properties you cannot guess reliably:

1. **Serialisation format** — the exact strings session A's tools return. Owned by
   `src/tools/registry.py`.
2. **Which tools Qwen picks, and how many** — a question you'd answer with one call, Qwen may
   split across three, concatenating three result blocks.
3. **Authentic noise** — empty result sets, a wrong-argument call, a retry, results in an
   unexpected order. Your `robust` slice is guessing at these; real traces show the true
   distribution.

**Format drift is the single failure mode that turns the adapter into noise** — train on one
shape, serve another.

**How to not be blocked:** generate Tier A now with your best-guess serialisation, but
**centralise it in one function** (`format_tool_results(...)`) so re-serialising the whole set is
a one-line change. When session A's registry lands and a handful of real Qwen traces exist,
regenerate. Cost of that regeneration ≈ minutes; cost of not centralising ≈ the whole set.

### Not Tier B: question paraphrasing

Tempting to have Qwen write phrasing variants (§2.5 rule 1). **Don't wait on it.** 3–5
hand-written templates per metric, randomised, are enough and cost you nothing in dependencies.
Use Qwen for this only if it is idle at the end.

---

## 4. What NOT to put in the training data

| Don't | Why |
|---|---|
| **The 15 public questions as training rows** | They are your correctness fixtures and your style anchors. Training on them destroys the only honest signal you have and edges toward what the rubric calls hidden evaluation data. Use them to *validate* the generator, never as rows. |
| **Raw AFR article rows in `tool_results`** | Both models cap at `max_model_len 4096`, `max_seq_len` is 512. Pass computed results, not corpora. |
| **Anything selected by BLEU/ROUGE** | They measure n-gram overlap, not whether a required fact is present. The primary metric is **component recall** over `meta.required_facts`. |
| **Records whose assistant answer is truncated** | Teaches the model to emit nothing. Truncate from the tool-results side only. |
| **Hedged answers in the gold targets** | *"approximately 41"* is explicitly scored wrong. Zero hedging, everywhere. |
| **Any ticker-year or RBA range in two splits** | Inflates the base-vs-FT comparison and the judges will find it. |

---

## 5. The constraint that decides real points

**Four of the fifteen public questions are a single 10-point all-or-nothing component** whose
`expected_fact` bundles three or four numbers — **26.7% of public points behind four YES/NO
gates**, where getting three of four numbers right scores **zero**:

| ID | Single component, all-or-nothing |
|---|---|
| MHQ001 | `41 of the 175 decision records changed the rate: 20 increases and 21 decreases.` |
| MHQ040 | `There are 18 ticker files, each containing 1,774 rows, covering 2 Jan 2015 through 30 Dec 2021.` |
| MHQ049 | `AMP.AX has the highest average daily volume at 11,635,671.71 shares per trading day.` |
| MHQ076 | `There are 369 AFR records matching whole-word QBE in 2021, and QBE.AX had the best non-Tabcorp 2021 return at +35.57%.` |

**Therefore: every compound `required_fact` in the training data must be a single sentence
carrying every sub-clause, in the reference's own shape.** Splitting one compound fact across
two sentences is how you score zero on a question you computed perfectly.

The mirror of this is **MHQ090** — a justified refusal worth 10 points across three components.
`"No"` alone earns 3.33; the evidence-boundary reasoning (*RBA covers the 2022–23 hikes, AFR and
ASX both end in 2021*) is worth more than the verdict. The `insufficient` slice must teach the
reasoning, not the verdict.

---

## 6. Execution order

```
PHASE 0, before session D forks (NOT D's work):
  [Q1] Judge calibration  ──────────────►  training/eval/judge_calibration.md
       ~120 Qwen calls, minutes                 decides answer style

D's T+0:00 — first act, in parallel:
  [read] training/eval/judge_calibration.md   <-- if absent, STOP and escalate
  [A]    Tier A generator: corpora loaders, metrics, templates
         gate on MHQ001 / MHQ040 / MHQ061 before emitting a single row

THEN:
  fix the assistant-answer style per Q1, emit train/val/heldout (80/10/10 by entity key)
  assert zero cross-split overlap, log it
  bank the BASE-MODEL eval on the heldout split before training touches node1's GPU

IN PARALLEL WITH TRAINING PREP:
  [Q2] Qwen teacher-labels a few hundred AFR articles  ──►  sentiment slice (12%)
       three-sentence shape, no numeric forecast

WHEN SESSION A's REGISTRY LANDS (or a few real traces exist):
  [Q3] re-run format_tool_results(...) over the whole set. One function, one change.
```

**Sequencing rule:** never let Q2 or Q3 stop Tier A. Tier A is 88% of the rows and
zero-dependency; it should be generating within the first thirty minutes. Q1 is the one
exception — it gates *answer style*, which is why it was moved ahead of the fork rather than
run alongside generation.

**Cut order if the schedule slips:** cross-dataset slice 25% → 10%, then Q3 re-serialisation,
then the `robust` slice.
**Never cut:** the §2.4 correctness gate, the entity-key split assertion, or the compound-fact
sentence rule in §5.
