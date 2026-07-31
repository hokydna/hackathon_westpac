# Agent Harness — Implementation Plan

**Date:** 2026-07-31
**Status:** design approved, not yet implemented
**Constraints driving every decision below:** <12 hours to submission, 2–3 people building in parallel.

This is the single planning artefact for `src/`. It exists so three people can start
simultaneously without blocking each other. No implementation code lives in this file.

---

## 1. Why this shape

The required architecture is fixed by the organizers:

```
question
  → Qwen (agent-brain, LiteLLM) plans + emits tool calls   [supplied, frozen]
  → our runtime validates + executes the tool calls
  → results loop back to Qwen until reasoning completes
  → our fine-tuned Nemotron synthesizes the final answer
  → POST /query returns {"answer", "steps", "tool_trace"}
```

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
├── config.py           every env var + budget constant, one place
├── app.py              FastAPI: GET /health, POST /query — thin
├── agent/
│   ├── loop.py         orchestrator: turn cap, deadline, trace assembly
│   ├── brain.py        one Qwen turn: thinking-off, tool schemas, timeout
│   ├── parser.py       XML <tool_call> → [ToolCall]
│   ├── guard.py        allowlist + argument coercion/validation
│   ├── synth.py        Nemotron synthesis: mock | llm
│   └── budget.py       token clamping + deadline tracking
└── tools/
    ├── corpora.py      startup loaders: RBA rows, ASX series, AFR inverted index
    ├── rba.py          deterministic RBA metrics
    ├── asx.py          deterministic ASX metrics
    ├── afr.py          index-backed count / count_by_month / share / retrieve
    └── registry.py     @tool definitions → ALL_TOOLS + BRAIN_SCHEMAS

tests/                  pytest, asyncio_mode=auto (langchain-basics pattern)
eval/run_public.py      score all 15 public questions — the acceptance gate
```

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
| **A** | `tools/` — all five files | nothing, start now |
| **B** | `agent/` — all six files | nothing; codes against §3 contracts, tests with a fake brain |
| **C** | `config.py`, `app.py`, `eval/`, submission hardening | nothing |

Owner B never needs owner A's real tools to make progress: the loop is driven by a fake
brain returning canned XML and a stub registry.

An "owner" may be a person or a Claude session. See §12 for the protocol that lets three
Claude sessions run these three tracks concurrently without merge conflicts.

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

1. **Turn cap** — `MAX_TURNS = 3`. The handout advises ≤3 tool calls and warns that
   looping more than 5 times will exceed 60s.
2. **Wall deadline** — 45s from request start. On breach, stop looping and synthesize
   from whatever trace exists rather than returning nothing.
3. **Context clamp** — every tool result truncated to 2,000 characters, AFR article text
   to 1,500, and the whole message list held under 3,000 tokens (drop oldest tool results
   first, never the system prompt or the question). Both models are capped at
   `max_model_len 4096`, so this is a correctness rule, not a nicety. Never put raw rows
   in context.

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

**Datasets** (`AI_Industry_Training_Hackathon/data set/`)

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

Derived from the real 15-question bank (5 easy / 7 medium / 3 hard; 8 cross-dataset).
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

**`coverage`** ★ — compares date ranges across datasets. MHQ090 is a hard question whose
correct answer is a **justified refusal**: RBA covers the 2022–23 hikes but AFR and ASX
both end in 2021, so the analysis is unsupported. The fine-tuned model must be trained to
state insufficiency rather than fabricate.

**Second role for the fine-tuned model.** `Setup_Instructions.md` requires that
article-grounded sentiment questions route the retrieved AFR text *and* the applicable
RBA rate through `DOMAIN_FT_MODEL`, returning positive/negative/mixed plus a likely
market direction — and explicitly not a fabricated numeric forecast. This is distinct
from final synthesis and needs its own prompt path in `synth.py`.

---

## 10. Build sequence

TDD throughout: write the failing test, confirm red, implement, confirm green, commit.
Ordered so the walking skeleton exercises the real loop shape as early as possible.

**Step 0 — scaffold (owner C, ~20 min)**
`config.py` with every env var and budget constant; `tests/conftest.py`; `.gitignore`
excluding `data set/`. Freeze the §3 contracts as type stubs so A and B can import them.

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
`synth.py` with `mock` and `llm` modes, the deterministic template fallback, and the
sentiment path. Point `llm` mode at node1's base model first — that both validates the
wiring and collects the base-model side of the comparison evidence.

**Step 8 — registry wiring (owners A+B, ~30 min)**
`registry.py` exports `ALL_TOOLS` and `BRAIN_SCHEMAS`. First real end-to-end `/query`.

**Step 9 — eval gate (owner C, ~1h)**
`eval/run_public.py` scoring all 15 public questions with per-component graders. This is
the acceptance gate; a systematic miss is a bug in the synthesis prompt or tool coverage,
never something to patch in the harness.

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
`LORA_RANK=32`, `LR=5e-5`, `MAX_SEQ_LEN=512`, `MAX_STEPS=100`, `WARMUP_STEPS=50`,
`CHECKPOINT_EVERY=20`. The handout warns in bold that `LR=1e-4` causes a loss spike at
warmup step 50, and reports the step-20 checkpoint already beating base (val loss 0.098).

---

## 12. Running three Claude sessions in parallel

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

### 12.2 Base commit — do this before forking (single session, ~20 min)

Two repo facts make this a prerequisite rather than a nicety:

- `origin` points at the **organizer's** repo, `cognitivo-aifactory/AI_Industry_Training_Hackathon`.
  Never push there. Add our own remote and push to that.
- `data set/` is **tracked** — 106 files, 780MB. Each worktree gets its own copy of every
  tracked file, so leaving it tracked costs ~2.3GB across three worktrees, and it must not
  ship in a public submission repo anyway.

```bash
cd ~/Hackathon_3107/AI_Industry_Training_Hackathon

printf '%s\n' 'data set/' '.worktrees/' '__pycache__/' '*.pyc' '.venv/' '.env' > .gitignore
git rm -r --cached "data set"        # untrack; leaves the 780MB on disk
git remote add team git@github.com:<our-org>/<our-repo>.git

# ... add config.py, contract stubs, tests/conftest.py, this plan ...
git add -A && git commit -m "Base: contracts, config, gitignore, plan"
git push team main
```

Because `data set/` becomes untracked, worktrees won't contain it. `config.py` therefore
resolves datasets through a `DATASET_DIR` env var, defaulting to the main checkout's
absolute path — one variable, and every worktree reads the same single 780MB copy.

### 12.3 Fork the sessions

Each Claude session isolates itself. If the session has a native worktree tool
(`EnterWorktree` or similar) it should use that and let the harness own placement and
cleanup; otherwise it falls back to git:

```bash
git worktree add .worktrees/tools        -b feat/tools
git worktree add .worktrees/agent        -b feat/agent
git worktree add .worktrees/serving-eval -b feat/serving-eval
```

`.worktrees/` is in `.gitignore` from §12.2, which the worktree skill requires before
creating anything project-local.

Never run three sessions in the same working tree. They will fight over the git index and
over test runs, and a failure in one becomes unattributable.

### 12.4 Session briefs

Give each session its own brief. Every brief points at this file and names the sections to
read — that is what keeps three sessions consistent without any cross-talk between them.

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

Merge order is free — the branches are file-disjoint by construction. Suggested:

```bash
git checkout main
git merge feat/tools feat/agent feat/serving-eval   # expect zero conflicts
pytest                                              # full suite
python eval/run_public.py                           # the acceptance gate
```

If a merge *does* conflict, the ownership boundary leaked. Fix the boundary, don't
hand-resolve and move on.

Then §10 step 8's first real end-to-end `/query`, and step 10's hardening. Both need all
three tracks landed, so they belong to whoever integrates rather than to a parallel
session.

### 12.6 Slotting in the eval and training plans when they arrive

Those two plans are still pending, and this split is deliberately built so neither
disturbs sessions A or B:

- **Eval plan** → lands in `eval/`, which session C already owns. C's §10 step 9 becomes
  "implement per the eval plan." Until it arrives, C has steps 1 and 10 to work on, so it
  is not idle. If the eval plan turns out to be large, fork it to its own session on
  `feat/eval` — still file-disjoint from everything above.
- **Training plan** → lands in `training/`, which no session above touches at all. Fork it
  as a fourth session on `feat/training` whenever the plan is ready. It is blocked on §11,
  not on anything in `src/`.

Both slot in cleanly for the same reason A/B/C do: disjoint file ownership over a base
commit that already contains every shared file. That is the whole trick — apply the §12.1
rule to each new plan as it lands and the parallelism keeps holding.
