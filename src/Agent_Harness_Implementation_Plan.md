# Agent Harness — Implementation Plan

**Date:** 2026-07-31 (reconciled to `SESSION_KICKOFF.md` on 2026-07-31)
**Status:** design approved, not yet implemented
**Constraints driving every decision below:** <12 hours to submission, 2–3 people building in parallel.

> **`src/SESSION_KICKOFF.md` is the source of truth.** This file is the harness design
> reference and is subordinate to it. Where they disagree, the kickoff wins. Corrections
> already applied here: the tool-result cap is **1,200** not 2,000 (kickoff F2); the
> difficulty split is **4 easy / 7 medium / 4 hard** (F7); the latency budget is decomposed
> (F10); §11's warmup is a **ratio** (F13); merges are sequential (F14); there are **four**
> sessions, not three (§12). Repo facts in §12.2 describe the old organizer checkout — see
> kickoff §2 "Repo facts" for the live ones.

This is the design artefact for `src/`. It exists so several people can start
simultaneously without blocking each other. No implementation code lives in this file.

---

## 1. Why this shape

The required architecture is fixed by the organizers:

```
question
  → Qwen (agent-brain, LiteLLM) plans + emits tool calls   [supplied, frozen]
  → our runtime validates + executes the tool calls
      ├─ deterministic data tools: rba / asx / afr / coverage
      └─ domain_sentiment → Nemotron  [Role 2 — the ONE tool that calls Nemotron]
  → results loop back to Qwen until reasoning completes
  → our fine-tuned Nemotron synthesizes the final answer   [Role 1 — NOT a tool]
  → POST /query returns {"answer", "steps", "tool_trace"}
```

**Nemotron has two roles and the distinction is scored.** Role 1 (final synthesis) is
unconditional and never appears in `ALL_TOOLS` — `Challenge_Brief.md` § Required Model Roles
forbids using Nemotron as "the primary tool-calling model" and requires it to receive results
*after* the loop, and `handout/03` titles the inverse pattern "Bad: Nemotron used as the
planner and tool caller." Role 2 (`domain_sentiment`) **is** a tool Qwen selects, because
`Setup_Instructions.md` L95 requires article-grounded sentiment to route AFR text + the
applicable RBA rate through `DOMAIN_FT_MODEL`. Full rules: `SESSION_KICKOFF.md` §10.

Two facts measured on the live cluster force a specific implementation of that shape:

1. **Tool calls are not machine-parseable as shipped.** `vllm-brain` runs
   `--tool-call-parser hermes`, but Qwen3.6 emits XML. `message.tool_calls` is always
   `null` and the call text leaks into `message.content`. So LangChain's
   `create_agent()`, which routes on `AIMessage.tool_calls`, would never dispatch a tool.
   We parse the XML ourselves.
2. **`enable_thinking:false` is not in effect server-side.** Per-request
   `chat_template_kwargs={"enable_thinking": False}` takes a brain turn from
   **15.0s → 0.9s** (800 → 43 completion tokens). It is required on every brain call.

**Chosen approach — hybrid.** Define tools with LangChain `@tool` (free Pydantic
argument validation and auto-derived JSON schemas for the brain), but drive them from
our own explicit loop, so we keep direct control of the three things scoring actually
pressures: turn count, wall-clock deadline, and the 4096-token context ceiling.

Rejected: pure hand-rolled (hand-maintaining every tool JSON schema, losing the arg
coercion we need because XML yields `year='2018'` as a string) and wrapping
`ChatOpenAI` to fabricate `tool_calls` (debugging LangChain internals at hour 9 is the
wrong risk to take on a 12-hour clock).

---

## 2. Module tree

```
src/
├── Agent_Harness_Implementation_Plan.md   ← this file
├── SESSION_KICKOFF.md  ← the source of truth; read it first
├── config.py           every env var + budget constant, one place  [FROZEN]
├── prompts.py          SYNTH_* + SENTIMENT_* prompts, shared with training  [FROZEN]
├── contracts.py        ToolCall, AgentResult, ALL_TOOLS/BRAIN_SCHEMAS stubs  [FROZEN]
├── domain_client.py    the single DOMAIN_FT_MODEL caller, mock|llm  [FROZEN]
├── app.py              FastAPI: GET /health, POST /query — thin
├── agent/
│   ├── loop.py         orchestrator: turn cap, deadline, trace assembly
│   ├── brain.py        one Qwen turn: thinking-off, tool schemas, timeout
│   ├── parser.py       XML <tool_call> → [ToolCall]
│   ├── guard.py        allowlist + argument coercion/validation
│   ├── synth.py        Nemotron synthesis: mock | llm
│   ├── budget.py       token clamping + deadline + tool_result_message()
│   └── tracing.py      @traceable no-ops unless LANGSMITH_TRACING  [FROZEN]
├── tools/
│   ├── corpora.py      startup loaders: RBA rows, ASX series, AFR inverted index
│   ├── rba.py          deterministic RBA metrics
│   ├── asx.py          deterministic ASX metrics
│   ├── afr.py          index-backed count / count_by_month / share / retrieve
│   └── registry.py     @tool definitions → ALL_TOOLS + BRAIN_SCHEMAS
└── eval/
    ├── component_judge.py   one expected_fact → YES/NO vs agent-brain  [FROZEN]
    └── run_offline_eval.py  score all 15 public questions — the acceptance gate

tests/                  pytest, asyncio_mode=auto (langchain-basics pattern)
tests/fixtures/dataset/ tiny (<1MB) corpus, one row per §8 gotcha
requirements.txt        PINNED
pyproject.toml          requires-python, asyncio_mode, test markers
```

Six files are **frozen in the base commit** and imported but never edited — that is what
lets four sessions run without merge conflicts. `eval/` lives under `src/` (not at the repo
root as an earlier draft had it) so it is one importable package: `python -m
src.eval.run_offline_eval`.

Each of the four cluster-level hazards lands in exactly one file: XML parsing in
`parser.py`, thinking suppression in `brain.py`, context/deadline in `budget.py`, model
mode in `synth.py`. When something breaks at hour 9, you know which file to open.

`tools/` is the largest unit and has no LLM dependency at all — it is where the
hidden-question points come from, and it is fully testable against known reference
answers.

---

## 3. Frozen contracts

These four freeze before anyone starts, so parallel work never blocks:

```python
ToolCall    = {"name": str, "args": dict}            # parser → guard → registry
AgentResult = {"answer": str, "steps": int, "tool_trace": list[dict]}   # loop → app

registry.ALL_TOOLS: list[BaseTool]      # owner A → owner B
registry.BRAIN_SCHEMAS: list[dict]      # owner A → owner B, auto-derived from ALL_TOOLS
```

Every tool returns `str`. That is the LangChain `@tool` convention and it is what lands
in the brain's context, which keeps clamping in exactly one place.

`tool_trace` entries are `{"tool": str, "args": dict, "result": str}`, matching
`Participant_Package/answer_template.json`.

---

## 4. Work split

| Owner | Files | Blocked by |
|---|---|---|
| **A** | `src/tools/` — all five files | nothing, start now |
| **B** | `src/agent/` — all six (not `tracing.py`, which is frozen) | nothing; codes against §3 contracts, tests with a fake brain |
| **C** | `src/app.py`, `src/eval/` (**except** the frozen `component_judge.py`), submission hardening | nothing |
| **D** | `training/` entirely | node1 ssh (B1); nothing in `src/` |

**`config.py` is no longer owner C's.** It is frozen in the base commit and every session
imports it read-only — the same for `prompts.py`, `contracts.py`, `tracing.py` and
`eval/component_judge.py`. That is the §12.1 rule applied across both workstreams, and it is
what pulled three more files into the base commit than §12.1 originally listed. See
`SESSION_KICKOFF.md` §4.

Owner B never needs owner A's real tools to make progress: the loop is driven by a fake
brain returning canned XML and a stub registry.

An "owner" is a Claude session. See §12 for the reasoning and `SESSION_KICKOFF.md` §5–§6 for
the roster and the prompts.

---

## 5. Data flow

```
POST /query {"question": ...}
  → app.py starts the deadline clock
  → loop.answer(question):
        msgs = [system(tool docs + answer rules), user(question)]
        for turn in 1..MAX_TURNS:
            if deadline breached: break
            reply = brain.plan(msgs)              # thinking off
            calls = parser.parse(reply.content)
            if not calls: break                   # brain is done reasoning
            for call in guard.validate(calls):
                result = await registry[call.name].ainvoke(call.args)
                msgs.append(tool_message(budget.clamp(result)))
                trace.append({tool, args, result})
        return synth.write(question, trace)
  → AgentResult, shaped to validate.json
```

**Startup ordering is load-bearing.** Corpora load at import, before uvicorn binds the
port, so the port only opens once the AFR index is built. `/health` therefore stays a
pure process-liveness check and can never 503 mid-build. `GET /health` failing is a hard
gate that zeroes the entire 40% hidden-question category, so it must not depend on
corpora state or on either model being reachable.

---

## 6. Budgets

Measured on this cluster, not estimated:

| Stage | Measured | Per-call timeout |
|---|---:|---:|
| Brain turn, thinking off | ~1.0s | 5s |
| Brain turn, thinking **on** (the trap) | 15.0s, 20.7s under 3-way concurrency | — |
| RBA / ASX tool | <0.1s | — |
| AFR tool, indexed | <0.01s | — |
| Synthesis, base Nemotron-8B | 3.7s | 15s |
| **End-to-end, 2 turns** | **~6–10s** | **45s hard deadline** |

Scoring thresholds: ≤60s full credit, 60–300s −20%, >300s zero. ~6x headroom.

Three rules live in `budget.py`:

1. **Turn cap** — `MAX_TURNS = 3`, as a runaway backstop rather than the primary governor.
   The handout advises ≤3 tool calls and warns that looping more than 5 times will exceed
   60s. **Note that turns ≠ tool calls:** one brain turn can emit several `<tool_call>`
   blocks, so a 3-turn cap already permits 6+ calls. `steps` in the response and `MAX_TURNS`
   are therefore different quantities — document them as such (kickoff F9).
2. **Wall deadline** — `LOOP_DEADLINE_S = 40s`, bounding **when the loop must stop**, not
   the whole request. On breach, stop looping and synthesize from whatever trace exists
   rather than returning nothing.

   ```python
   PENALTY_THRESHOLD_S = 60.0   # >60s costs 20% of the question's points
   SYNTH_TIMEOUT_S     = 15.0
   SAFETY_MARGIN_S     =  5.0
   LOOP_DEADLINE_S     = PENALTY_THRESHOLD_S - SYNTH_TIMEOUT_S - SAFETY_MARGIN_S   # 40.0
   ```

   *40, not the 45 this plan originally specified.* 45s of looping plus a 15s synthesis
   timeout is 60s before FastAPI even serialises, landing on the wrong side of the penalty
   line. Assert the sum in `test_budget_invariants` so tuning one constant cannot silently
   break the total. Resolved in kickoff F10.
3. **Context clamp** — every tool result truncated to **1,200 characters**
   (`config.TOOL_RESULT_CHAR_CAP`), AFR article text to 1,200, and the whole message list
   held under 3,000 tokens (drop oldest tool results first, never the system prompt or the
   question). Both models are capped at `max_model_len 4096`, so this is a correctness rule,
   not a nicety. Never put raw rows in context.

   *1,200, not the 2,000 this plan originally specified.* `FINETUNE_PLAN.md` §4.4 caps
   training-time `tool_results` at 1,200 against `max_seq_len = 512`, and training and
   inference **must** clamp identically or the adapter is served a context shape it never
   saw. Resolved in kickoff F2.

Our own timeouts must sit well under the organizer LiteLLM config's
`request_timeout: 120` with `num_retries: 2`, whose worst case is 360s on a single call
— past the harness's own 300s timeout.

---

## 7. Error handling

One principle: **never return a non-answer.** Every failure path still produces a valid
`answer` string stating the limitation. Required by `Challenge_Brief` § Rules ("Return a
response for every question... State the limitation clearly in the `answer` field") and
by `validate.json` (`answer` is the only required field, `minLength: 1`).

| Failure | Behaviour |
|---|---|
| Disallowed tool name | Denied in `guard`; structured error into trace; brain replans; nothing executed |
| Malformed arguments | Coerce first (`"2018"` → `2018`), then Pydantic error down the same path |
| Empty result set | Valid "no results" string, never an exception — brain can try another query |
| Brain timeout / 5xx | Stop looping, synthesize from the partial trace |
| Synthesis failure | Deterministic template answer built from the trace, so a Nemotron outage degrades to partial credit rather than zero |
| Deadline breach | Same as above |
| `/health` | Touches neither models nor corpora; returns 200 while the process lives |

Concurrency safety is by construction, not by locking: corpora are read-only after
startup, and messages/trace are per-request locals. The harness sends up to three
concurrent `/query` requests and state must not bleed between them.

---

## 8. Verified constants

Measured this session. Implementers should **not** re-derive these — use them as test
fixtures.

**Datasets** — `data set/` at the repo root, resolved through `config.DATASET_DIR` (absolute,
so every worktree reads one copy). The directory is **untracked**, so worktrees do not contain
it; unit tests use `tests/fixtures/dataset/` instead (kickoff F11).

| Dataset | Shape |
|---|---|
| RBA | 175 rows, 3 Feb 2010 → 17 Jun 2026. `RBA-rates.jsonl`, **UTF-8 BOM** — open with `encoding='utf-8-sig'` |
| ASX | 18 tickers × 1,774 rows = 31,932 rows, 2015-01-02 → 2021-12-30 |
| AFR | 219,538 articles across 85 files, 780MB on disk / 771MB of text in RAM |

**Field gotchas**

- RBA dates are `"3 Feb 2010"`, not ISO. All RBA values are strings, including signed
  changes (`"+0.25"`, `"0.00"`).
- AFR `PUBLICATIONDATE` is a `YYYYMMDD` **string**, not ISO. Slice `[:4]` for year and
  `[:6]` for month; do not date-parse.
- ASX tickers carry the `.AX` suffix. Tabcorp is `TAH.AX` and is excluded in 5 of the 15
  public questions.

**AFR search rule — reproducibility-critical.** `Setup_Instructions.md` calls this
non-negotiable: case-insensitive, `\bword\b` anchored, matched across
`HEADLINE + SUBHEAD + INTRO + TEXT` combined, counted **once per record**. A different
field set silently produces different counts that will not match the reference answers.

Verified: tokenizing each record with `[a-z0-9]+` and using an inverted index is
**exactly equivalent** to that rule.

| Pattern | Full regex scan | Inverted index | |
|---|---:|---:|---|
| `\bunemployment\b` | 5,997 | 5,997 | match |
| `\bqbe\b` | 1,546 | 1,546 | match |
| `\bnab\b` | 7,372 | 7,372 | match |

The naive tokenizer `[a-z0-9']+` gives 6,903 for `nab` — wrong, because `\b` treats an
apostrophe as a boundary. Do not include `'` in the token class.

**AFR performance.** Full regex scan is ~6.5–6.9s per pattern and `re` does not release
the GIL, so neither caching nor threads fix it. The inverted index costs ~32s to build at
startup and then answers in **<1ms**. This is why the index is mandatory rather than an
optimization.

**Reference answers reproduced** (use as test fixtures)

- MHQ001 — 41 of 175 records changed the rate: 20 increases, 21 decreases
- MHQ040 — 18 ticker files, 1,774 rows each, 2 Jan 2015 → 30 Dec 2021
- MHQ061 — peak year 2020 with 1,452 records; peak month May 2020 with 218

**Cluster topology**

```
node0 = 10.0.1.10  (this box — brain/agent node)
  litellm :4000    aliases: agent-brain, domain-ft
  vllm    :8000    Qwen/Qwen3.6-35B-A3B-Instruct-FP8, max_model_len 4096,
                   --enable-auto-tool-choice --tool-call-parser hermes
node1 = 10.0.1.11  (fine-tuning/model node)
  vllm    :8001    Llama-3.1-Nemotron-Nano-8B-v1  (BASE), max_model_len 4096
```

`domain-ft` is currently **unhealthy**: LiteLLM routes it to model name
`nemotron-8b-finance`, but node1 advertises `Llama-3.1-Nemotron-Nano-8B-v1`. When our
adapter is served we must either serve it under `nemotron-8b-finance` or get the alias
changed. Node1's base model is meanwhile a free base-vs-fine-tuned control endpoint for
the 30% model-quality evidence — start collecting that side now.

---

## 9. Tool surface

Derived from the real 15-question bank (**4 easy / 7 medium / 4 hard**; 8 cross-dataset;
150 points total). Re-measured from `public_questions.jsonl` — the 5/7/3 in earlier drafts
was wrong, and eval slices must be built on the real distribution (kickoff F7).
Items marked ★ are required by the question bank but absent from the execution guide's
metric list — they are easy to miss.

**`rba`** — `count`, `count_changes`, `count_increases`, `count_decreases`, `extremes`,
`max_hold_streak`, `lookup_rate`, `cycle_summary` ★ (cumulative change over a date range,
for tightening/easing-cycle questions)

`lookup_rate` must be **as-of** semantics: the rate in effect on-or-before a date.
Nearest-match can return a future decision.

**`asx`** — `annual_return`, `rank_annual_returns`, `full_sample_return`, `volatility`,
`correlation`, `max_drawdown`, `avg_volume`, `describe` ★ (file/row counts and date
range, for MHQ040), `window_return` ★ (N-day return from an RBA decision date, incl.
equal-weighted baskets, for the cross-dataset questions)

`exclude_tickers` is a first-class argument on every ASX metric.

**`afr`** — `count`, `count_by_month`, `share`, `retrieve_by_headline` ★ (exact headline +
publication date, for MHQ058/067/080)

**Status: implemented and verified.** Every metric below now exists in `src/tools/` and reproduces its published reference answer — see `SESSION_KICKOFF.md` §12 for the measured constants and for four semantics the plans got wrong (ASX filenames, MHQ061's search term, the equal-weighted basket, and trading-day snapping).

**`coverage`** ★ — compares date ranges across datasets. MHQ090 is a hard question whose
correct answer is a **justified refusal**: RBA covers the 2022–23 hikes but AFR and ASX
both end in 2021, so the analysis is unsupported. The fine-tuned model must be trained to
state insufficiency rather than fabricate.

**`domain_sentiment` ★ — the one tool that calls Nemotron.** `Setup_Instructions.md` L95
requires article-grounded sentiment questions to route the retrieved AFR text *and* the
applicable RBA rate through `DOMAIN_FT_MODEL`, returning positive/negative/mixed plus a likely
market direction — and explicitly **not** a fabricated numeric forecast. Worth 30 of the 150
public points (MHQ058, MHQ067, MHQ080).

```python
domain_sentiment(headline: str, article_text: str, publication_date: str, rba_rate: str) -> str
```

It lives in **`src/tools/registry.py`** (owner A), not in `synth.py` — Qwen has to orchestrate
it (retrieve the article, look up the as-of rate, then classify), so it belongs in the tool
surface. It imports the frozen `src/domain_client.py` and `src/prompts.py`, never owner B's
`synth.py`. It returns a **classification clamped to 200 chars, not an answer**, it is denied
when the trace holds no retrieved article, and final synthesis still runs after it. Those four
constraints are what keep Role 2 from collapsing into the prohibited Role-1-as-a-tool pattern.
See `SESSION_KICKOFF.md` §10.

---

## 10. Build sequence

TDD throughout: write the failing test, confirm red, implement, confirm green, commit.
Ordered so the walking skeleton exercises the real loop shape as early as possible.

**Step 0 — the base commit (Phase 0, one session before anyone forks, ~40 min)**
Not owner C's — this is `SESSION_KICKOFF.md` §4 and it is **blocking**. `config.py` with every
env var and budget constant, `prompts.py`, `contracts.py`, `eval/component_judge.py`,
`agent/tracing.py`, `tests/conftest.py`, `tests/fixtures/dataset/`, `requirements.txt` +
`pyproject.toml`, the `role:"tool"` probe, and the judge calibration. Every one of these is a
file two or more sessions would otherwise both write.

**Step 1 — walking skeleton (owner C, ~30 min)**
`app.py` with `GET /health` returning 200 and `POST /query` returning a hardcoded answer
that validates against `validate.json`. Test: `/health` returns 200 with
`LITELLM_BASE_URL` pointing nowhere.

**Step 2 — the two blockers (owner B, ~45 min)**
`parser.py` against the real captured XML payload, then `brain.py` with
`chat_template_kwargs={"enable_thinking": False}` and a 5s timeout. These two are what
make everything downstream viable — do them before the loop.

**Step 3 — corpora + RBA (owner A, ~45 min)**
`corpora.py` RBA and ASX loaders, then `rba.py`. Test: MHQ001 reproduces 41/175/20/21.

**Step 4 — loop (owner B, ~1h)**
`guard.py`, `budget.py`, `loop.py`. Tests with a fake brain: turn cap honoured, deadline
breach synthesizes from partial trace, trace shape matches the template.

**Step 5 — AFR index (owner A, ~1h)**
Inverted index in `corpora.py` with the `[a-z0-9]+` tokenizer, then `afr.py`. Test:
MHQ061 reproduces 2020=1,452 and May 2020=218, and tokenizer counts equal regex counts
for `nab`/`qbe`/`unemployment`.

**Step 6 — ASX metrics (owner A, ~1h)**
`asx.py` incl. `describe`, `window_return`, `exclude_tickers`. Test: MHQ040 dimensions;
MHQ045 best/worst 2018 excluding Tabcorp.

**Step 7 — synthesis (owner B, ~45 min)**
`synth.py` with `mock` and `llm` modes and the deterministic template fallback, over the
frozen `domain_client`. Point `llm` mode at node1's base model first — that both validates the
wiring and collects the base-model side of the comparison evidence. **Synthesis is
unconditional and is never registered as a tool** (§1, kickoff §10): `loop.py` calls it after
the loop exits on every request, including deadline breach and zero successful tool calls. The
sentiment path is *not* here — it is owner A's `domain_sentiment` tool in step 6b.

**Step 6b — `domain_sentiment` tool (owner A, ~30 min)**
The Role-2 Nemotron tool, in `registry.py`, over the frozen `domain_client` and
`prompts.SENTIMENT_*`. Test in `mock` mode first so it needs no live adapter. Assert: output
clamped to 200 chars, no digit-bearing forecast in the output, and denial when the trace holds
no retrieved AFR article.

**Step 8 — registry wiring (owners A+B, ~30 min)**
`registry.py` exports `ALL_TOOLS` and `BRAIN_SCHEMAS`. First real end-to-end `/query`.

**Step 9 — eval gate (owner C, ~1h)**
`src/eval/run_offline_eval.py` scoring all 15 public questions via the frozen
`src/eval/component_judge.py`. Two passes, never interleaved: collect all 15 answers timed
under 3 concurrent workers, then judge untimed (running the judge inline contends for the
same vLLM brain and corrupts the timings). Headline number is the **penalised** score —
`earned_points × (1.0 if t≤60s else 0.8 if t≤300s else 0.0)`. This is the acceptance gate; a
systematic miss is a bug in the synthesis prompt or tool coverage, never something to patch
in the harness, and never a question-ID-specific hardcoded answer. See kickoff §6.4.

**Step 10 — hardening (owner C, ~45 min)**
Three concurrent `/query` requests with distinct questions, asserting no crossed answers.
Latency measurement against the 60s threshold. Secret scan. `submission.json` with the
pinned commit SHA. Confirm `data set/` is untracked.

**Critical path check.** Per owner, sequentially: A ≈ 3h (steps 3, 5, 6, 8), B ≈ 2h 45m
(steps 2, 4, 7, 8), C ≈ 2h 35m (steps 0, 1, 9, 10). The three run concurrently, so `src/`
lands in roughly 3–4 hours including integration friction — leaving the rest of the
12-hour budget for the training run and iteration on the eval gate.

---

## 11. Open dependency, outside `src/`

**The fine-tuning scripts do not exist on this box.** `~/llm-eval-nim-demo` contains only
`01-NIM-Evaluation.ipynb` — no `automodel_recipes/`, no `compare_infer.py`,
no `spark_finetune.py`. And `~/Cognitivo_Training` contains exactly one file
(`AI_Training_and_Hackathon/Sample_Activity_5/p2_agent/team.env`) — no
`finagent-finetune/scripts/` as the handouts instruct. `MODELS_DIR` is also unset, and
`team.env` is not at `~/team.env`.

`nvcr.io/nvidia/nemo:25.09` is pulled, so the container is fine; the recipes are the gap.
A 100-step 8B run takes 2–3 hours, which makes this the critical path on a 12-hour clock.
Someone must chase the organizers for these **in parallel with** the `src/` build.

`src/` works either way — `DOMAIN_PREDICT_MODE` switches between `mock` and `llm` — but
`DOMAIN_PREDICT_MODE=mock` at submission forfeits the 30% model-quality category and
architecture credit with it.

When training does happen, use the handout's baseline, not the older spec's:
`LORA_RANK=32`, `LR=5e-5`, `MAX_SEQ_LEN=512`, `CHECKPOINT_EVERY=20`, and
**`warmup_ratio=0.10`** with `max_steps` **measured** by a smoke test rather than fixed at
100. The handout warns in bold that `LR=1e-4` causes a loss spike at warmup step 50, and
reports the step-20 checkpoint already beating base (val loss 0.098).

*The handout's `WARMUP_STEPS=50` assumes a 100-step run and must not be copied literally.*
On a 60-step run a fixed 50-step warmup means the LR never reaches target and the model
trains on essentially nothing — `FINETUNE_PLAN.md` §8.1 calls this the most likely silent
failure in the whole project. Use a ratio (kickoff F13). `FINETUNE_PLAN.md` is authoritative
for everything in this section.

---

## 12. Running parallel Claude sessions

> **Superseded in the operational details.** `SESSION_KICKOFF.md` §4–§6 is what sessions
> actually execute: **four** sessions (the training track is now a first-class stream, not a
> future arrival), a larger base-commit set, corrected repo facts, and copy-paste prompts.
> §12.1 below — the *rule* — is the part that still governs. Read this for the reasoning and
> the kickoff for the instructions.

### 12.1 The rule that makes it work

**Every file that two sessions would both need to edit must be complete in the base
commit, before any session forks.** Get that right and the three tracks in §4 touch
strictly disjoint files, so their branches merge without conflicts and nobody blocks
anybody.

Applied to this plan, three things must land in the base commit:

- `src/config.py` — complete. A and B both *read* budget constants and env vars from it;
  neither may edit it.
- The §3 contracts as importable stubs — `ToolCall`, `AgentResult`, and empty
  `ALL_TOOLS` / `BRAIN_SCHEMAS` exports.
- `tests/conftest.py` — shared fixture root, or all three sessions collide on it.

One file needs a deliberate owner call: `src/tools/registry.py` is listed in §10 step 8 as
"owners A+B". Give it to **A alone**. B imports it and never edits it.

### 12.2 Base commit — do this before forking

> **Dead text, kept for the reasoning only.** This subsection described the organizer's
> checkout: `origin` pointing at `cognitivo-aifactory/…`, `data set/` tracked, no
> `.gitignore`. **All three are resolved.** The live repo is
> `~/Hackathon_3107/hackathon_westpac`, `origin` is ours, `data set/` is untracked and
> ignored, and the `.gitignore` is in place. There is no `team` remote and none is needed.
> **Use `SESSION_KICKOFF.md` §2 "Repo facts" and §4 for what to actually run.**

The reasoning that still applies: because `data set/` is untracked, **worktrees do not
contain it.** `config.py` therefore resolves datasets through an absolute `DATASET_DIR` — one
variable, and every worktree reads the same single 785 MB copy. And because CI and a judge's
clone have neither, unit tests read `tests/fixtures/dataset/` instead, with only
`needs_dataset`-marked tests touching the real corpus (kickoff F11).

### 12.3 Fork the sessions

Each Claude session isolates itself. If the session has a native worktree tool
(`EnterWorktree` or similar) it should use that and let the harness own placement and
cleanup; otherwise it falls back to git:

```bash
git worktree add .worktrees/tools        feat/tools
git worktree add .worktrees/agent        feat/agent
git worktree add .worktrees/serving-eval feat/serving-eval
git worktree add .worktrees/training     feat/training
```

All four branches already exist off `main`, so omit `-b`. `.worktrees/` is git-ignored.

Never run two sessions in the same working tree. They will fight over the git index and
over test runs, and a failure in one becomes unattributable.

### 12.4 Session briefs

Give each session its own brief. Every brief points at a plan file and names the sections to
read — that is what keeps the sessions consistent without any cross-talk between them.

**The authoritative roster and the copy-paste prompts are `SESSION_KICKOFF.md` §5–§6**, which
covers four sessions and adds `config.py` / `prompts.py` to every "must not touch" column. The
table below is the three-session original, kept for its reasoning.

**All three briefs share this preamble:**

> Read `src/Agent_Harness_Implementation_Plan.md` §1–3 (approach and frozen contracts) and
> §8 (verified constants). §8 values are already measured — use them as test fixtures, do
> not re-derive them. Work TDD: failing test, confirm red, implement, confirm green,
> commit. Do not edit files outside your owned set; if you need a change there, note it and
> keep going.

| Session | Owns | Must not touch | Plan sections | Done when |
|---|---|---|---|---|
| **A — tools** | `src/tools/*` (all five, incl. `registry.py`), `tests/test_rba.py`, `test_asx.py`, `test_afr.py` | `src/agent/*`, `app.py`, `config.py` | §9, §10 steps 3, 5, 6, 8 | MHQ001, MHQ040, MHQ061 reproduce; `ALL_TOOLS` + `BRAIN_SCHEMAS` export |
| **B — agent** | `src/agent/*` (all six), `tests/test_parser.py`, `test_guard.py`, `test_loop.py` | `src/tools/*`, `app.py`, `config.py` | §5, §6, §7, §10 steps 2, 4, 7 | Real XML parses; brain turn ~1s; turn cap + deadline + partial-synth all tested against a fake brain |
| **C — serving & eval** | `app.py`, `eval/*`, `tests/test_app.py`, submission hardening | `src/agent/*`, `src/tools/*` | §5, §7, §10 steps 1, 9, 10 | `/health` 200 with LiteLLM unreachable; `/query` validates against `validate.json`; `eval/run_public.py` scores all 15 |

Session B's highest-value instruction, worth stating explicitly in its brief: **do
`parser.py` and `brain.py` before the loop.** Those two files are the difference between
~1s and ~15s per brain turn, and between tools dispatching at all versus never.

### 12.5 Integration

Merge order is free — the branches are file-disjoint by construction — but merge **one at a
time**, not as an octopus. An octopus merge aborts wholesale on any conflict and loses the
signal about *which* boundary leaked, which is the only diagnostic a conflict gives you here
(kickoff F14).

```bash
git switch main
for b in feat/tools feat/agent feat/serving-eval feat/training; do
  git merge --no-ff "$b" || { echo "BOUNDARY LEAKED: $b"; break; }
done
pytest                                   # full suite
python -m src.eval.run_offline_eval      # the acceptance gate
```

If a merge *does* conflict, the ownership boundary leaked. Fix the boundary, don't
hand-resolve and move on.

Then §10 step 8's first real end-to-end `/query`, and step 10's hardening. Both need all
three tracks landed, so they belong to whoever integrates rather than to a parallel
session.

### 12.6 The eval and training plans — both have landed

Written when both were pending. Both now exist, and both slotted in exactly as predicted:

- **Eval plan** → `Cognitivo_Labs/Cognitivo_Labs/docs/superpowers/reviews/2026-07-31-evaluation-strategy-review.md`.
  It specifies the component judge, calibration protocol, frozen-tool-results ablation,
  failure attribution and penalised scoring. Session C implements §10 step 9 from it directly
  and **is not waiting on anything.** Read it in place — never copy that tree into this repo,
  it contains a plaintext credential. No separate `feat/eval` session: the review's §E.5
  argued for one, the kickoff rejected it (F8), and the kickoff wins.
- **Training plan** → `src/FINETUNE_PLAN.md` plus `src/FINETUNE_DATA_SOURCES.md`, landing in
  `training/`, which no other session touches. It is session **D** on `feat/training`, and it
  forks **in the same batch as A/B/C, not after them** — it is the critical path (2–3 h of
  training behind a blocked ssh prerequisite), so it must start first, not last.

Both slotted in cleanly for the same reason A/B/C do: disjoint file ownership over a base
commit that already contains every shared file. That is the whole trick — and applying the
§12.1 rule to the training plan is what pulled `prompts.py` and `component_judge.py` into the
base commit (kickoff F1, F4).
