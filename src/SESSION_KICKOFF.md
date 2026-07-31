# Session Kickoff — how we start parallel implementation

**Date:** 2026-07-31 (reconciled 2026-07-31, repo migration + review reconciliation)
**Status:** ready to execute. Phase 0 is blocking; nothing forks until it lands.
**Repo:** `~/Hackathon_3107/hackathon_westpac` → `git@github.com:hokydna/hackathon_westpac.git`
**Covers:** `src/Agent_Harness_Implementation_Plan.md` (the harness) and `src/FINETUNE_PLAN.md`
(the model). Both are design-approved. This file is the bridge between "approved" and
"four Claude sessions are typing."

> ## This file is the source of truth
>
> Where this document and any other `src/*.md` disagree, **this document wins.** The other
> four are subordinate:
>
> | Doc | Standing |
> |---|---|
> | `Agent_Harness_Implementation_Plan.md` | Design reference for the harness. Its §6/§9/§11/§12 numbers are corrected here — read §2 and §9 below before trusting any constant in it. |
> | `Agent_Harness_Plan_Review.md` | Review of the above. Its §E is superseded; §A–§D findings are resolved in §9 below, adopted or rejected explicitly. |
> | `FINETUNE_PLAN.md` | Session D's plan of record, subordinate to §2 and §9 here. |
> | `FINETUNE_DATA_SOURCES.md` | Session D's data provenance reference. |
>
> Do not fix a contradiction by editing one of those four. Fix it here, then propagate.

Read this if you are about to start a session, or if you are the one who owns Phase 0.

---

## 1. What this file does

The harness plan already contains the parallelism protocol (§12). This file:

1. Records the **cross-plan defects found reviewing the two plans against each other and
   against the Participant Package** (§2). Six of them are shared-file collisions that
   would surface as merge conflicts or, worse, as a silently mistrained adapter.
2. Turns the harness plan's §12.1 rule into a **concrete Phase 0 checklist** (§4) that
   also covers the fine-tuning workstream, which §12 did not.
3. Gives **copy-paste kickoff prompts** for all four implementation sessions (§6).
4. Resolves every open finding from `Agent_Harness_Plan_Review.md` — adopted or rejected,
   with the reason (§9). Nothing from that review is left silently dropped.

The one rule everything below serves:

> **Every file two sessions would both need to edit must be complete in the base commit,
> before any session forks.**

Applied across *both* plans rather than just the harness, that rule pulls three more files
into the base commit than §12.1 lists — see §4.

---

## 2. Review findings — resolve these before forking

Both plans are sound and internally consistent. The problems are all at the seam between
them, plus a few things the Participant Package requires that neither plan carries.

| # | Finding | Where | Decision |
|---|---|---|---|
| **F1** | **The synthesis prompt is a shared file with no owner.** `FINETUNE_PLAN.md` §4.1 freezes `SYNTH_SYSTEM`/`SYNTH_USER` and says the harness must read the identical strings. The harness plan assigns `agent/synth.py` to session B, which never reads the fine-tuning plan. If B writes its own prompt, the adapter is trained on one format and served another — §4.1's own words: "the adapter degrades to noise." | both | Extract to `src/prompts.py` in the **base commit**. B and D both import it; neither edits it. |
| **F2** | **Tool-result character cap contradicts.** Harness §6 clamps to 2,000 chars; `FINETUNE_PLAN.md` §4.4 caps training-time `tool_results` at ~1,200 and flags the conflict as its own open question #3. Training and inference must agree. | both | **1,200.** `max_seq_len=512` is the binding constraint and a truncated assistant answer teaches the model to emit nothing. Lands as `TOOL_RESULT_CHAR_CAP = 1200` in `config.py`. |
| **F3** | **`FINETUNE_PLAN.md` references a file that does not exist.** It cites `src/PLAN.md` eight times as the companion doc. The file is `src/Agent_Harness_Implementation_Plan.md`. A fresh session will not find it. | `FINETUNE_PLAN.md` | Fix the references in Phase 0, or the D-session prompt supplies the real path (§6 does both). |
| **F4** | **The component judge is a second shared file.** Harness §10 step 9 gives `eval/run_public.py` to session C. `FINETUNE_PLAN.md` §7 + §11 give `training/eval_compare.py` to session D. Both need the same per-`expected_fact` YES/NO judge against `agent-brain`. Two implementations = two different numbers for the same thing. | both | `src/eval/component_judge.py` in the **base commit**. C and D both import it. |
| **F5** | **Four public questions are single 10-point all-or-nothing compound components** — MHQ001, MHQ040, MHQ049, MHQ076. 26.7% of public points sit behind four YES/NO gates where three of four numbers right scores **zero**. Neither plan mentions this. It is the strongest constraint on the FT data design and on the synthesis prompt. | neither | Written into the D and B prompts. FT data must teach: emit every sub-clause of a compound fact in one sentence, in the reference's own shape. |
| **F6** | **Judge calibration is unscheduled.** Whether the FT target is *natural synthesis* or *near-template reproduction of `reference_answer`* is currently a guess, and it changes the training set. ~120 judge calls, minutes to run. `FINETUNE_PLAN.md` §8 starts generating data at T+0:15 without it. | `FINETUNE_PLAN.md` | Runs in the base commit / Phase 0 window, before D generates data. Result goes in `training/eval/judge_calibration.md`. |
| **F7** | **Difficulty distribution is wrong in the harness plan.** §9 says 5 easy / 7 medium / 3 hard. Measured over `public_questions.jsonl`: **4 / 7 / 4**. Cross-dataset count (8) is right. | harness §9 | Cosmetic; correct it in Phase 0 so eval slices are built on the real distribution. |
| **F8** | **The eval plan the harness §12.6 waits for already exists** — it is `Cognitivo_Labs/.../reviews/2026-07-31-evaluation-strategy-review.md`, which specifies the component judge, calibration protocol, frozen-tool-results ablation, failure attribution, and penalised scoring. Session C does not need to wait for anything. | harness §12.6 | C's prompt (§6) points at it directly. No fifth session needed. |

### Still-live blockers, verified 2026-07-31

Re-probed this session; every one of these still holds:

- **B1 — `ssh 10.0.1.11` → `Permission denied (publickey,password)`.** The entire
  fine-tuning critical path queues behind it. Not a coding task; needs a human chasing
  organizers from minute zero.
- **`domain-ft` alias broken** — LiteLLM routes it to `nemotron-8b-finance`, node1
  advertises `Llama-3.1-Nemotron-Nano-8B-v1`. Free fix: serve our adapter with
  `--served-model-name nemotron-8b-finance`.
- **Base Nemotron is live on `10.0.1.11:8001` right now.** It is the control arm for 30%
  of the score and it disappears the moment training starts. Collect the base arm first.
- **No fine-tuning scripts on this box.** `~/Cognitivo_Training` has one `team.env`;
  `~/llm-eval-nim-demo` has one notebook. `MODELS_DIR` is unset. We write our own (both
  plans already assume this).

### Repo facts — changed 2026-07-31, supersedes every other doc

The submission repo **moved**. Anything in the other four docs about remotes, `origin`, a
`team` remote, or untracking `data set/` describes the old repo and is dead text.

| Fact | Now |
|---|---|
| Working repo | `~/Hackathon_3107/hackathon_westpac` — **this** is the repo root for every session |
| `origin` | `git@github.com:hokydna/hackathon_westpac.git` — **ours.** Push here. There is no `team` remote and none is needed |
| Organizer's repo | `cognitivo-aifactory/AI_Industry_Training_Hackathon`, kept as a separate local checkout at `~/Hackathon_3107/AI_Industry_Training_Hackathon`. It is **not** a remote of this repo, so there is nothing to push to by accident |
| `data set/` | Already untracked and ignored. 785 MB on disk, 0 bytes in git. **Never `git add -f` it** |
| `.gitignore` | Already in place, covers `data set/`, weights, raw logs, secrets, `.worktrees/` |
| Repo visibility | Must be **public** before submission — private and collaborator-only are rejected |
| `submission.json` | Still template values (`mock-team`, `example/mock-team-agent`, `0123…4567`). Owned by the integrator, filled last |

---

## 3. What starts when

The two plans have different clocks and they are not sequential.

```
T+0:00  Phase 0 (one session, ~40 min)  ── base commit, no forking
        ‖ in parallel, human:  chase B1 (ssh node1)
T+0:40  fork A, B, C, D  ── four sessions, four worktrees, disjoint files
        D's clock starts here: 3 hours, hard stop, per FINETUNE_PLAN §8
T+3:40  D's hard stop.  Whatever checkpoint exists is what ships.
T+4:00  integration (§8): merge, full pytest, eval gate, hardening
```

D is the critical path — 2–3 h of training plus a blocked prerequisite — so **D forks in
the same batch as A/B/C, not after them.** A/B/C total ~3 h of concurrent work and land
inside D's training window.

---

## 4. Phase 0 — the base commit (blocking, one session, ~40 min)

Do **not** fork until every box is ticked. Each of these is a file two or more sessions
would otherwise both write.

**Repo hygiene — already done, verify only**

- [x] `.gitignore` — `data set/`, weights, raw logs, secrets, `.worktrees/`
- [x] `data set/` untracked — 785 MB on disk, 0 in git
- [x] `origin` is our own repo; no `team` remote needed
- [ ] Confirm nothing from `Cognitivo_Labs/` is copied into this repo. `Notes.md` line 25
      holds a plaintext cluster password and the submission repo must be fully public.
- [ ] Repo set to **public** on GitHub (required; do this before the final gate, not at hour 11)

**Shared code — the whole point of Phase 0**

- [ ] `src/config.py` — every env var and budget constant, read from env, no hard-coded
      endpoints. Must include `TOOL_RESULT_CHAR_CAP = 1200` (F2), `MAX_TURNS = 3` (F9),
      the derived latency budget below (F10), `BRAIN_TIMEOUT_S = 5`,
      `DATASET_DIR` (absolute, so worktrees share one 785 MB copy), `DOMAIN_PREDICT_MODE`,
      and the `LANGSMITH_*` block from F12 — even though tracing ships off, the variables
      must exist here or adding them later violates the frozen-file rule.

      ```python
      PENALTY_THRESHOLD_S = 60.0   # scoring boundary: >60s costs 20%
      SYNTH_TIMEOUT_S     = 15.0
      SAFETY_MARGIN_S     =  5.0
      LOOP_DEADLINE_S     = PENALTY_THRESHOLD_S - SYNTH_TIMEOUT_S - SAFETY_MARGIN_S  # 40.0
      ```

      **`LOOP_DEADLINE_S` bounds when the loop must stop, not the whole request** (F10).
      Assert the sum in `tests/test_budget_invariants.py` so nobody tunes one constant and
      silently breaks the total.
- [ ] `requirements.txt`, pinned, plus `pyproject.toml` with `requires-python`,
      `asyncio_mode = "auto"` and the markers `unit, integration, contract, regression,
      smoke, e2e, perf, needs_dataset` (F11). The module tree in the harness plan has no
      dependency manifest at all, and "reproducibility" is a named rubric item.
- [ ] `tests/fixtures/dataset/` — a **tiny** (<1 MB) corpus, one row per §8-gotcha of the
      harness plan: a UTF-8 BOM, a `"+0.25"` string change, an apostrophe-adjacent `nab`,
      a `PUBLICATIONDATE` on a month boundary, a `TAH.AX` row (F11). Without this, A's
      tests only run on this box — worktrees have no `data set/`, so a judge cloning the
      public repo cannot run the suite at all.
- [ ] `src/agent/tracing.py` — `@traceable` wrappers that are **no-op passthroughs when
      `LANGSMITH_TRACING` is falsy** (F12). Lands here so B only decorates and never edits
      a shared file. Ships disabled; dev-only.
- [ ] `src/prompts.py` — **F1.** `SYNTH_SYSTEM` and `SYNTH_USER` verbatim from
      `FINETUNE_PLAN.md` §4.1, plus the sentiment-path prompt. Imported by
      `src/agent/synth.py` (B) and `training/prepare_data.py` (D). **Neither may edit it.**
- [ ] `src/contracts.py` — `ToolCall`, `AgentResult` as importable stubs
      (harness §3), and empty `ALL_TOOLS` / `BRAIN_SCHEMAS` exports so B can import
      before A has written anything.
- [ ] `src/eval/component_judge.py` — **F4.** One `expected_fact` per call against
      `agent-brain`, `chat_template_kwargs={"enable_thinking": false}`, returns YES/NO.
      Imported by C and D.
- [ ] `tests/conftest.py` — shared fixture root, `asyncio_mode=auto`.

**Plan corrections — done 2026-07-31, listed for audit**

- [x] `src/PLAN.md` → `src/Agent_Harness_Implementation_Plan.md` throughout
      `FINETUNE_PLAN.md` (F3)
- [x] Harness §6: 2,000 → 1,200 chars (F2)
- [x] Harness §9: 5/7/3 → 4/7/4 (F7) — re-measured from `public_questions.jsonl`:
      4 easy / 7 medium / 4 hard, 150 points total
- [x] Harness §11: `WARMUP_STEPS=50` → `warmup_ratio=0.10`, per `FINETUNE_PLAN.md` §8.1 (F13)
- [x] Sequential merges replace the octopus merge in harness §12.5 and §8 here (F14)

**The 2-minute probe that unblocks session B (F15)**

- [ ] Establish whether a `role: "tool"` message is accepted. The brain's XML arrives in
      `message.content`, so there is no `tool_calls` entry for a `tool_call_id` to reference.
      If LiteLLM rejects it, B's loop must append results as a `user` message shaped like
      `<tool_response>…</tool_response>` instead.

      ```bash
      curl -s "$LITELLM_BASE_URL/chat/completions" -H "Authorization: Bearer $LITELLM_KEY" \
        -H 'Content-Type: application/json' -d '{
        "model":"agent-brain","max_tokens":64,
        "chat_template_kwargs":{"enable_thinking":false},
        "messages":[
          {"role":"user","content":"count RBA changes"},
          {"role":"assistant","content":"<tool_call>{\"name\":\"rba\",\"arguments\":{\"metric\":\"count_changes\"}}</tool_call>"},
          {"role":"tool","content":"41 of 175 records changed the rate."}
        ]}' | head -40
      ```

      Record the outcome in this file's §9 F15 row before B forks. `budget.py` owns a single
      `tool_result_message()` factory either way, so the decision lives in one place.

**The one measurement that must happen before D generates data**

- [ ] **Judge calibration (F6).** Three arms over the 15 public questions —
      `reference_answer` verbatim (expect all YES), one number perturbed beyond tolerance
      (expect that component NO), and a paraphrase with reformatted dates and reordered
      facts (expect all YES). Write `training/eval/judge_calibration.md`. If the paraphrase
      arm largely **fails**, tell session D immediately: the FT target becomes template
      reproduction, not natural synthesis, and that is a materially different training set.

Then:

```bash
git switch main
git add -A && git commit -m "chore(repo): base commit — config, frozen prompts, contracts, component judge"
git push origin main          # origin IS our repo now; there is no `team` remote
```

**Two ownership calls, made here:**

1. `src/tools/registry.py` is listed in harness §10 step 8 as "owners A+B." It belongs to
   **A alone**. B imports it and never edits it.
2. **Judge calibration (F6/Q1) belongs to Phase 0, not to session D.**
   `FINETUNE_DATA_SOURCES.md` §6 schedules it as D's parallel work; that is wrong, because
   its result decides the answer style for every record D generates. It runs here, before
   forking, and D reads `training/eval/judge_calibration.md` as an input it never writes.

---

## 5. Session roster

Four sessions, four branches, file-disjoint by construction.

The four branches already exist off `main`. Each session claims one and works in its own
worktree — directory name matches the branch suffix, so `git worktree list` is readable:

```bash
git worktree add .worktrees/tools        feat/tools
git worktree add .worktrees/agent        feat/agent
git worktree add .worktrees/serving-eval feat/serving-eval
git worktree add .worktrees/training     feat/training
```

(`.worktrees/` is git-ignored. Drop the `-b` — the branches are already created; `-b` would
fail.)

If the session has a native worktree tool (`EnterWorktree`), use that instead and let the
harness own placement and cleanup. **Never run two sessions in the same working tree** —
they fight over the git index and failures become unattributable.

| Session | Branch | Owns | Must not touch | Done when |
|---|---|---|---|---|
| **A — tools** | `feat/tools` | `src/tools/*` (all five incl. `registry.py`), `tests/test_{rba,asx,afr}.py` | `src/agent/*`, `app.py`, `config.py`, `prompts.py` | MHQ001, MHQ040, MHQ061 reproduce exactly; `ALL_TOOLS` + `BRAIN_SCHEMAS` export |
| **B — agent** | `feat/agent` | `src/agent/*` (all six), `tests/test_{parser,guard,loop}.py` | `src/tools/*`, `app.py`, `config.py`, `prompts.py` | Real captured XML parses; brain turn ≈1 s; turn cap + deadline + partial-synthesis all tested against a fake brain |
| **C — serving & eval** | `feat/serving-eval` | `app.py`, `src/eval/*` (except `component_judge.py`), `tests/test_app.py`, submission hardening | `src/agent/*`, `src/tools/*`, `config.py`, `prompts.py` | `/health` 200 with LiteLLM unreachable; `/query` validates against `validate.json`; offline harness scores all 15 with penalised scoring + attribution |
| **D — fine-tuning** | `feat/training` | `training/*` entirely | everything under `src/` | Adapter served as `nemotron-8b-finance`, `domain-ft` returns 200, base-vs-FT table written, `DOMAIN_PREDICT_MODE=llm` |

Nobody blocks anybody: B codes against the frozen contracts with a fake brain and a stub
registry; D's evaluator runs standalone against the served endpoint even if `src/` is
mid-build.

---

## 6. Kickoff prompts

Paste one of these into a fresh session, in its own worktree. They are written to be
self-contained — a session with no prior context can execute from the prompt alone.

### 6.1 Shared preamble (prepended to all four)

```
You are one of four parallel sessions building a hackathon submission. Repo root:
/home/cognitivo_g01/Hackathon_3107/hackathon_westpac

Read first, in this order:
  1. src/SESSION_KICKOFF.md — it is the SOURCE OF TRUTH. §2 (cross-plan defects and repo
     facts), §5 (your ownership boundary), §9 (resolved review findings). Where it and any
     other src/*.md disagree, it wins; do not "fix" the other doc.
  2. the plan sections named in your brief below

Ground rules, all sessions:
- Work TDD: write the failing test, confirm it is red, implement, confirm green, commit.
  Small commits. Use the superpowers test-driven-development skill.
- Measured constants in the plans (dataset shapes, latencies, reference answers) are
  already verified on this cluster. USE THEM AS TEST FIXTURES. Do not re-derive them;
  re-measuring the AFR corpus costs minutes you do not have.
- Do not edit any file outside your owned set. If you need a change there, append a note
  to HANDOFF-<your-letter>.md at the repo root and keep going. Never "just quickly fix"
  another session's file — that is what breaks the merge.
- src/config.py and src/prompts.py are FROZEN. Import from them; never edit them.
  src/prompts.py holds the synthesis prompt that the fine-tuned adapter is trained
  against. Changing it silently invalidates the adapter.
- Push to `origin <your-branch>`. `origin` is OUR repo (hokydna/hackathon_westpac). Ignore
  any instruction in the older docs about a `team` remote or not pushing to origin — the
  repo moved and those lines are dead.
- Never commit `data set/` (785 MB, already ignored), model weights, or a .env. Never
  `git add -f`.
- No credentials, endpoints, or keys in source. Read from env.
- Conventional Commits, and name the plan section in the body: `Refs: kickoff §4` or
  `Refs: harness §10 step 3`. Judges score repository structure; an unstructured history
  reads as one dump.
- Report status when you hit your definition of done, or when you are blocked for more
  than 15 minutes.
```

### 6.2 Session A — tools

```
<shared preamble>

You own the deterministic tool layer: src/tools/{corpora,rba,asx,afr,registry}.py and
tests/test_rba.py, test_asx.py, test_afr.py. This is where the hidden-question points
come from and it has no LLM dependency at all — everything you write is testable against
known reference answers.

Read: src/Agent_Harness_Implementation_Plan.md §1-3 (approach + frozen contracts),
§8 (verified constants — these are your fixtures), §9 (tool surface), §10 steps 3, 5, 6, 8.

Build in this order:
  1. corpora.py — RBA + ASX loaders. RBA is UTF-8 BOM: open with encoding='utf-8-sig'.
     Dates are "3 Feb 2010", not ISO. All RBA values are strings including signed
     changes ("+0.25", "0.00").
  2. rba.py — test first: MHQ001 must reproduce 41 of 175 changed, 20 increases,
     21 decreases.
  3. AFR inverted index in corpora.py, then afr.py. Tokenize with [a-z0-9]+ — do NOT
     include the apostrophe; [a-z0-9']+ gives 6,903 for "nab" instead of the correct
     7,372, because \b treats an apostrophe as a boundary. Test: MHQ061 reproduces
     2020=1,452 and May-2020=218, and index counts equal regex counts for
     nab/qbe/unemployment.
  4. asx.py — including describe, window_return, and exclude_tickers as a first-class
     argument on every metric. Test: MHQ040 dimensions; MHQ045 best/worst 2018 excluding
     TAH.AX.
  5. registry.py — export ALL_TOOLS and BRAIN_SCHEMAS. You own this file alone.

Non-negotiable semantics (Setup_Instructions.md calls these reproducibility-critical —
a different field set silently produces counts that will not match the reference answers):
- AFR: case-insensitive, \bword\b anchored, matched across HEADLINE+SUBHEAD+INTRO+TEXT
  COMBINED, counted ONCE PER RECORD.
- RBA lookup_rate is AS-OF: the rate in effect on-or-before a date. Nearest-match can
  return a future decision and is wrong.
- Every tool returns a str (LangChain @tool convention) and never raises on an empty
  result set — return a valid "no results" string so the brain can try another query.
- Never put raw rows in a return value. Both models are capped at max_model_len 4096.

Add a `coverage` tool that compares date ranges across datasets. MHQ090's correct answer
is a justified refusal — RBA covers the 2022-23 hikes but AFR and ASX both end in 2021.

Two measurements only you can make, both go in your status report:
- Peak RSS after the AFR index build. Each node is a single GB10 with unified memory SHARED
  with vLLM, and vllm-brain lives on node0 next to you. An unmeasured multi-GB index next to
  a preallocated KV cache OOMs the brain and takes sessions B and C down with it. If it is
  tight, hold offsets into an mmap'd corpus instead of the text.
- Index build wall time. If it is anywhere near the measured ~32s, persist it to disk
  (pickle keyed on a hash of corpus mtimes+sizes) so a restart costs seconds. /health is a
  hard 40% gate and a restart currently means ~32s of failing /query calls.

Run unit tests against tests/fixtures/dataset/ (in the base commit), not the real 785 MB
corpus — your worktree does not contain `data set/`, it is untracked. Only tests marked
needs_dataset use the real thing via DATASET_DIR.

Done when: MHQ001, MHQ040 and MHQ061 reproduce exactly, all tools are unit-tested against
reference values, and ALL_TOOLS + BRAIN_SCHEMAS export cleanly.
```

### 6.3 Session B — agent runtime

```
<shared preamble>

You own the agent loop: src/agent/{loop,brain,parser,guard,synth,budget}.py and
tests/test_parser.py, test_guard.py, test_loop.py.

Read: src/Agent_Harness_Implementation_Plan.md §1-3, §5 (data flow), §6 (budgets),
§7 (error handling), §10 steps 2, 4, 7.

DO parser.py AND brain.py BEFORE THE LOOP. These two are the difference between tools
dispatching at all versus never, and between ~1s and ~15s per brain turn:

  1. parser.py — vllm-brain runs --tool-call-parser hermes but Qwen3.6 emits XML, so
     message.tool_calls is ALWAYS null and the call text leaks into message.content.
     LangChain's create_agent() routes on AIMessage.tool_calls and would therefore never
     dispatch a tool. Parse the XML yourself. Args arrive as strings (year='2018') —
     coerce at the guard boundary. Test against real captured XML:
       <tool_call><function=query_data><parameter=dataset>asx</parameter>...
  2. brain.py — pass chat_template_kwargs={"enable_thinking": False} on EVERY brain call.
     The server's --override-generation-config does not govern the chat template. Measured:
     15.0s / 800 tokens without it, 0.9s / 43 tokens with it. Assert it in a test. 5s
     per-call timeout.

Then guard.py (allowlist + Pydantic coercion), budget.py (turn cap 3, 45s wall deadline,
clamp every tool result to config.TOOL_RESULT_CHAR_CAP), and loop.py.

Then synth.py — mock and llm modes plus a deterministic template fallback built from the
trace, so a Nemotron outage degrades to partial credit rather than zero. Import the
prompts from src/prompts.py; do NOT write your own. Point llm mode at node1's base model
(http://10.0.1.11:8001/v1) first — that validates the wiring and collects the base side of
the model-quality comparison. synth.py also needs a second, distinct prompt path for
article-grounded sentiment: AFR text + the applicable RBA rate in, sentiment
(positive/negative/mixed) + likely market direction out, and explicitly NO fabricated
numeric forecast.

One principle governs every failure path: NEVER RETURN A NON-ANSWER. Brain timeout, tool
error, deadline breach, synthesis failure — all still produce a valid answer string that
states the limitation. validate.json requires only `answer`, minLength 1.

Answer-style constraint that decides real points: four of the fifteen public questions are
a SINGLE 10-point component whose expected_fact bundles three or four numbers. Getting
three of four right scores ZERO on that question. Whatever synth.py emits must state every
sub-clause of a compound fact in one sentence. No hedging — "approximately 41" is
explicitly scored wrong.

Test everything against a fake brain returning canned XML and a stub registry — you never
need session A's real tools to make progress.

Done when: real XML parses to a ToolCall; a brain turn measures ~1s; and the turn cap,
deadline-breach-synthesizes-from-partial-trace, and trace-shape behaviours are all tested.
```

### 6.4 Session C — serving, eval, submission

```
<shared preamble>

You own: src/app.py, src/eval/* (except component_judge.py, which is frozen in the base
commit), tests/test_app.py, and submission hardening.

Read: src/Agent_Harness_Implementation_Plan.md §5, §7, §10 steps 1, 9, 10 — and
Cognitivo_Labs/Cognitivo_Labs/docs/superpowers/reviews/2026-07-31-evaluation-strategy-review.md
in full. That review IS the eval plan; you are not waiting on anything. Read it in place —
never copy that tree into this repo, it contains a plaintext credential.

Build in this order:

  1. app.py walking skeleton. GET /health returns 200; POST /query returns a hardcoded
     answer that validates against Participant_Package/validate.json. Test that /health
     returns 200 with LITELLM_BASE_URL pointing nowhere. /health is a HARD GATE — if it
     fails the pre-eval check the team is skipped and the entire 40% hidden-question
     category is zeroed. It must never touch models or corpora. Load corpora at import,
     before uvicorn binds, so the port only opens once the AFR index is built.

  2. Offline eval harness. Two passes, never interleaved:
     - Pass 1: collect all 15 answers, TIMED, under 3 concurrent workers (that is the
       graded condition; single-request timings understate it by ~40%).
     - Pass 2: judge, UNTIMED, via the frozen src/eval/component_judge.py. Running the
       judge inline contends for the same single vLLM brain and corrupts the timings you
       are trying to collect.
     Headline number is the PENALISED score:
       question_score = earned_points * (1.0 if t<=60s else 0.8 if t<=300s else 0.0)
     Read grading.tolerance_note per question — three different tolerance tiers are in
     play across the 15. Report compound-component hit rate separately: MHQ001, MHQ040,
     MHQ049 and MHQ076 are single 10-point all-or-nothing components carrying 26.7% of
     public points.

  3. Failure attribution, logged on every run. Per question record tool calls + args, tool
     results, the exact synthesis input, the final answer, and per-component verdicts.
     Then: component NO and the fact is ABSENT from tool results -> routing/execution
     (sessions A/B). Component NO and the fact is PRESENT in tool results -> synthesis
     (session D). Without this a 0 tells nobody which workstream to fix.

  4. Hardening. Three concurrent /query requests with distinct questions, asserting no
     crossed answers. Latency measured against the 60s threshold, and specifically MEASURE
     THINKING-OFF BRAIN LATENCY UNDER 3-WAY CONCURRENCY — the harness plan §6 has 20.7s for
     thinking-ON under concurrency but no thinking-off figure, and 3 concurrent requests is
     the graded condition. That number decides whether the 60s threshold is actually safe.
     Assert the budget invariant too: LOOP_DEADLINE_S + SYNTH_TIMEOUT_S + SAFETY_MARGIN_S
     == PENALTY_THRESHOLD_S (60). Secret scan broader than `nvapi-` — generic
     password/credential patterns too. Confirm `data set/` is still untracked.
     submission.json with the pinned 40-char commit SHA, the machine's real IP (not
     localhost), and agent.timeout_seconds=300.

A systematic miss on the eval gate is a bug in the synthesis prompt or in tool coverage.
It is never something to patch in the harness, and never a question-ID-specific hardcoded
answer — the brief explicitly prohibits that.

Done when: /health returns 200 with both models unreachable; /query validates against
validate.json; the offline harness scores all 15 with penalised scoring and attribution;
three concurrent requests do not cross state.
```

### 6.5 Session D — fine-tuning

```
<shared preamble>

You own training/ entirely and you execute src/FINETUNE_PLAN.md end to end. That file is
subordinate to this one — where they disagree, this file wins (see §9).

Read: src/FINETUNE_PLAN.md in full (it is your plan, not a reference), plus
Cognitivo_Labs/Cognitivo_Labs/docs/superpowers/reviews/2026-07-31-evaluation-strategy-review.md
§3 and §5. Read that review in place — never copy that tree into this repo, it contains a
plaintext credential.

Your clock is 3 hours, hard stop. Whatever checkpoint exists at T+2:00 is what ships.

The objective is NOT "best adapter in 3 hours." The rubric scores six things and only two
are adapter quality; the rest are documented data prep, config and checkpoint-selection
rationale, a quantitative base-vs-FT comparison, and evidence the adapter is genuinely
used. So: shortest path to a genuinely-served adapter, then every remaining minute on
evidence. A mediocre adapter that is live and documented outscores an excellent one that
is not. Shipping with DOMAIN_PREDICT_MODE=mock forfeits the entire 30% model-quality
category and drags architecture credit down with it.

Ordering that is not negotiable:

  1. Bank the base-model evaluation against the live base endpoint at
     http://10.0.1.11:8001/v1 BEFORE training takes that GPU — the control arm disappears
     with it, and it is half of a 30%-weighted deliverable. It scores the heldout split, so
     it runs immediately after generation (FINETUNE_PLAN §8, ~T+0:50), not before it. Nothing
     may touch node1 until it is banked. Grab the §2.6 /tokenize sequence-budget measurement
     in the same window — same endpoint, same deadline.
     Read training/eval/judge_calibration.md from the base commit FIRST, before you design
     the data mix. If it is missing, stop and say so: generating against a guessed answer
     style means regenerating everything.
  2. In parallel, chase blocker B1: `ssh 10.0.1.11` currently returns
     "Permission denied (publickey,password)". Verified still broken today. Everything
     downstream queues behind it. If unresolved by T+0:20, take the FINETUNE_PLAN §9
     fallback and train on node0 inside the already-pulled vllm/vllm-openai container —
     and DO NOT disturb the vllm-brain container, sessions A/B/C depend on it.
  3. Generate data. Blocking gate before any training starts: the generator must reproduce
     MHQ001 (41/175, 20 up, 21 down), MHQ040 (18 files x 1,774 rows, 2 Jan 2015 –
     30 Dec 2021) and MHQ061 (2020=1,452; May 2020=218). If they do not match, the
     training data is wrong and nothing downstream matters. Fail loudly.
  4. Smoke test to MEASURE the real step rate, then set max_steps so training ends by
     T+2:00. Use warmup_RATIO 0.10, not the handout's fixed WARMUP_STEPS=50 — with a
     60-step run a fixed 50-step warmup means the LR never reaches target and you have
     trained on essentially nothing. This is the most likely silent failure in the plan.
  5. Select the checkpoint on VAL COMPONENT RECALL, not val loss. Log every checkpoint's
     numbers so the choice is auditable; the rubric asks for the rationale explicitly.
     The handout reports step-20 already beating base, so evaluate early checkpoints
     rather than assuming more steps are better.

Hard constraints:
- Import SYNTH_SYSTEM/SYNTH_USER from src/prompts.py verbatim. Training format and
  inference format drifting apart is the single failure that turns the adapter into noise.
- Cap tool_results in every training record at 1200 chars (config.TOOL_RESULT_CHAR_CAP),
  truncating from the tool-results side ONLY. A truncated assistant answer teaches the
  model to emit nothing.
- Train on assistant tokens only — TRL completion-only collator or assistant_only_loss.
  Without masking the model learns to generate tool results as well as answers.
- LR = 5e-5. NOT 1e-4 — the handout documents a loss spike at warmup step 50 on this exact
  model and hardware. NVIDIA's own deck recommends 1e-4 to 1e-3 for PEFT; the handout wins.
- Split train/val/heldout 80/10/10 BY METRIC-AND-ENTITY KEY, not by row, so no ticker-year
  or RBA date range appears in two splits. Assert zero overlap and log the assertion —
  cross-split leakage is exactly what the rubric means by "must not contain hidden
  evaluation data."
- Never use BLEU/ROUGE to select anything. The primary metric is component recall over
  meta.required_facts, judged by the frozen src/eval/component_judge.py. Same prompts,
  same decoding (temperature=0, max_tokens=256), one variable: the adapter.

Data design constraint that decides real points: four of the fifteen public questions are
a SINGLE 10-point component whose expected_fact bundles three or four numbers — getting
three of four right scores ZERO. Train the model to emit every sub-clause of a compound
fact in one sentence, in the reference_answer's own shape. Read
training/eval/judge_calibration.md from the base commit before designing the mix: if the
paraphrase arm failed, your target is template reproduction, not natural synthesis.

Also train two behaviours neither obvious nor optional: justified refusal when coverage is
missing (MHQ090's correct answer is a refusal worth 10 points across three graded parts —
"No" alone earns only 3.33; the evidence-boundary reasoning is worth more than the
verdict), and the sentiment path (AFR text + applicable RBA rate -> sentiment + likely
direction, explicitly NO fabricated numeric forecast).

Serving, when you get there: `--served-model-name nemotron-8b-finance`. LiteLLM routes
domain-ft to that name while node1 currently advertises Llama-3.1-Nemotron-Nano-8B-v1, so
this one flag repairs the broken alias with no organizer involvement. Verify with a
domain-ft call through localhost:4000 returning 200, then set DOMAIN_PREDICT_MODE=llm and
confirm the agent actually routes through it.

Done when: the adapter is served, domain-ft returns 200, training/COMPARISON.md and
training/MODEL_SUMMARY.md are written, and DOMAIN_PREDICT_MODE=llm is confirmed in the
running agent.
```

---

## 7. Cadence and escalation

- **Status ping** at definition-of-done, or when blocked >15 min. Not on a timer.
- **Cross-session needs go in `HANDOFF-<letter>.md`** at the repo root, one file per
  session, never into another session's code. The integrator reads all four at merge.
- **A merge conflict means the ownership boundary leaked.** Fix the boundary; do not
  hand-resolve and move on.
- **B1 escalation is a human's job**, not a session's. Whoever owns organizer contact
  chases it from minute zero and reports into session D.

---

## 8. Integration

Merge order is free — the branches are file-disjoint by construction — but merge them **one
at a time** (F14). An octopus merge aborts entirely on any conflict and destroys the signal
about *which* boundary leaked, which is the only diagnostic value a conflict has here.

```bash
git switch main
for b in feat/tools feat/agent feat/serving-eval feat/training; do
  git merge --no-ff "$b" || { echo "BOUNDARY LEAKED: $b"; break; }
done
pytest                                   # full suite
python -m src.eval.run_offline_eval      # the acceptance gate
```

Then the two things that need all four tracks landed, and therefore belong to the
integrator rather than any parallel session:

1. First real end-to-end `POST /query` (harness §10 step 8).
2. Hardening (harness §10 step 10) — three concurrent requests, latency against the 60 s
   threshold, secret scan, `submission.json` with the pinned commit SHA.

**Final gate before submitting**, in order — each is independently capable of zeroing a
whole category:

- [ ] `GET /health` returns 200 **from another machine**, using the IP in `submission.json`
- [ ] `POST /query` returns a non-empty `answer` validating against `validate.json`
- [ ] `DOMAIN_PREDICT_MODE=llm` and the agent demonstrably routes through `domain-ft`
- [ ] Three concurrent `/query` requests, no crossed answers
- [ ] Most responses under 60 s
- [ ] `data set/` untracked; no credentials in any tracked file; `Cognitivo_Labs/` not copied
- [ ] `submission.json`: real IP, 40-char commit SHA, `timeout_seconds: 300`,
      `model.model_name = "nemotron-8b-finance"`, `model.endpoint` on node1 (not localhost)
- [ ] `README.md`, `src/`, `training/`, `logs/` populated per `submission-guide.md`
- [ ] Repo is **public** and clonable without credentials

---

## 9. Resolved review findings

`Agent_Harness_Plan_Review.md` raised items that the earlier version of this file neither
adopted nor rejected. Each now has a decision. **F9–F16 extend the F1–F8 set in §2.**

| # | Finding | Review ref | Decision |
|---|---|---|---|
| **F9** | `MAX_TURNS = 3` vs the review's recommended 5. The handout's best team ran avg 4.65 steps and says hard questions need 3–5 tool calls. | C3 | **Keep 3, because turns ≠ tool calls.** One brain turn can emit several `<tool_call>` blocks, so a 3-turn cap already permits 6+ calls and satisfies the handout. `LOOP_DEADLINE_S` (F10) is the real governor; the turn cap is a runaway backstop. **Document `steps` ≠ `MAX_TURNS` in the response contract** — they are different quantities and the review is right that the plan conflated them. |
| **F10** | Deadline arithmetic crosses the scored boundary: a 45 s wall deadline plus a 15 s synthesis timeout is 60 s **before** FastAPI serialisation, landing on the wrong side of the −20% line. | C2 | **Adopted.** The deadline was bounding the wrong quantity. `LOOP_DEADLINE_S = 40` bounds when the *loop* stops, leaving synthesis its full budget inside the 60 s envelope. Constants and the invariant test are in §4. Measured end-to-end is ~6–10 s, so this costs nothing except in the pathological case — which is exactly where the penalty bites. |
| **F11** | No dependency manifest anywhere, and no test fixtures, so the suite cannot run off this box. | A.2, D.3 | **Adopted into Phase 0** (§4). `requirements.txt` + `pyproject.toml` + a <1 MB `tests/fixtures/dataset/`. Reproducibility is a named 30% rubric item and a judge cloning the repo currently cannot run a single test. |
| **F12** | LangSmith: nothing traces automatically under a hand-rolled loop; `config.py` has no `LANGSMITH_*` block. | E, G1, G2, E.5 | **Partially adopted.** The review's own banner supersedes its §E: the scored path must run with the network down (`Challenge_Brief` § Rules) and must not send hidden question text to a third party. So — **tracing ships OFF.** But `config.py` gets the `LANGSMITH_*` variables and `src/agent/tracing.py` lands in the base commit as no-op passthroughs, because both are frozen files and adding them later breaks the §4 rule. Dev-only observability, one env var to enable. |
| **F13** | Harness §11 still quotes `WARMUP_STEPS=50`, which `FINETUNE_PLAN.md` §8.1 identifies as the most likely silent training failure. | — | **`warmup_ratio = 0.10`.** Corrected in the harness plan. A fixed 50-step warmup on a 60-step run means the LR never reaches target. |
| **F14** | Octopus merge aborts wholesale and loses the per-branch signal. | V12 | **Adopted.** Sequential `--no-ff` merges, §8. |
| **F15** | The `role: "tool"` message may be rejected, and it is on B's critical path. | C1 | **Adopted as a Phase 0 probe** (§4). Two minutes now versus a blocked session B at hour 4. Record the outcome in this row. **Result: _unrun — fill this in._** |
| **F16** | Review §A/§B propose step notes, ADRs, `CHANGELOG.md`, pre-commit + gitleaks, and CI. | A.2, B.1, B.3, V5–V9 | **Mostly rejected on the clock, two exceptions.** `README.md` says `docs/` is not required, and a 12-hour build does not fund an ADR set. **Adopted:** (a) Conventional Commits naming the plan section, already in the §6.1 preamble — it is free and it is what "repository structure" credit reads; (b) `training/MODEL_SUMMARY.md` and `training/COMPARISON.md`, which are already D's deliverables and carry most of the 30% model-quality evidence. **Rejected:** `docs/steps/*`, `docs/DECISIONS/*`, `CHANGELOG.md`, pre-commit, CI. The secret scan survives as a §8 gate item rather than a hook. |

### Two review items promoted into session prompts

- **Peak RSS after the AFR index build** (C4) — session A measures and reports it. The index
  sits in unified memory next to `vllm-brain`'s preallocated KV cache on node0; an unmeasured
  multi-GB index OOMs the brain, not just the agent. Mitigation if tight: keep offsets into an
  mmap'd corpus rather than the text.
- **Thinking-off latency under 3-way concurrency** (C4) — session C measures it. §6 of the
  harness plan has the figure for thinking-**on** (20.7 s) but not off, and 3 concurrent
  requests is the graded condition.
