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
**For current build state and what to pick up next, jump to §11.**

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
| **F6 ✅ RESOLVED** | **Judge calibration was unscheduled.** Whether the FT target is *natural synthesis* or *near-template reproduction of `reference_answer`* was a guess, and it changes the whole training set. `FINETUNE_PLAN.md` §8 started generating data at T+0:15 without it. | `FINETUNE_PLAN.md` | **Ran in Phase 0, 2026-07-31, 108 judge calls. Verdict: `NATURAL_SYNTHESIS`.** Control **1.000**, paraphrase **1.000** (full marks 15/15), negative **0.482** with the deliberately-broken component correctly failing **14/14** — so the judge is tolerant *and* discriminating. **D's answer style is natural synthesis**; sentence order and date format are free. **Caveat the arm did not test:** it reorders whole sentences and never splits one, so §4.5 still binds independently — each compound `required_fact` stays in ONE sentence. Natural synthesis is safe; natural *decomposition* of a compound fact is not. See `training/eval/judge_calibration.md`, which also records three earlier runs whose verdicts were harness bugs. |
| **F7** | **Difficulty distribution is wrong in the harness plan.** §9 says 5 easy / 7 medium / 3 hard. Measured over `public_questions.jsonl`: **4 / 7 / 4**. Cross-dataset count (8) is right. | harness §9 | Cosmetic; correct it in Phase 0 so eval slices are built on the real distribution. |
| **F8** | **The eval plan the harness §12.6 waits for already exists** — it is `Cognitivo_Labs/.../reviews/2026-07-31-evaluation-strategy-review.md`, which specifies the component judge, calibration protocol, frozen-tool-results ablation, failure attribution, and penalised scoring. Session C does not need to wait for anything. | harness §12.6 | C's prompt (§6) points at it directly. No fifth session needed. |

### Still-live blockers, verified 2026-07-31

Re-probed this session; every one of these still holds:

- **B1 — `ssh 10.0.1.11` → `Permission denied (publickey,password)`.** The entire
  fine-tuning critical path queues behind it. Not a coding task; needs a human chasing
  organizers from minute zero.
- ~~**`domain-ft` alias broken**~~ — **RESOLVED 2026-07-31.** The alias now answers from a
  separate vLLM instance (fingerprint `5a3f83cd` vs the brain's `6bc76779`), so
  `DOMAIN_PREDICT_MODE=llm` works today against base Nemotron. When serving the adapter,
  keep `--served-model-name nemotron-8b-finance` so it stays working. See §11.
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
      `FINETUNE_PLAN.md` §4.1, **plus `SENTIMENT_SYSTEM` / `SENTIMENT_USER`** for the
      Role-2 sentiment tool (§10). Imported by `src/agent/synth.py` (B),
      `src/tools/registry.py` (A) and `training/prepare_data.py` (D). **None may edit it.**
- [ ] `src/domain_client.py` — **F17, §10.** The single `DOMAIN_FT_MODEL` caller. Honours
      `DOMAIN_PREDICT_MODE` (`mock` | `llm`), `SYNTH_TIMEOUT_S`, `temperature=0`. Exposes
      `complete(system, user, max_tokens)`. Imported by A's sentiment tool and B's `synth.py`;
      **neither edits it.** Without it, A would have to import B's `synth.py`, which may not
      exist yet — exactly the cross-stream dependency Phase 0 exists to prevent.
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

- [x] **DONE — ACCEPTED (see F15).** Establish whether a `role: "tool"` message is accepted. The brain's XML arrives in
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

- [x] **DONE — verdict `NATURAL_SYNTHESIS` (see F6).** Judge calibration. Three arms over the 15 public questions —
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
| **A — tools** | `feat/tools` | `src/tools/*` (all five incl. `registry.py`) **and the `domain_sentiment` tool (§10)**, `tests/test_{rba,asx,afr,sentiment}.py` | `src/agent/*`, `app.py`, `config.py`, `prompts.py`, `domain_client.py` | MHQ001, MHQ040, MHQ061 reproduce exactly; `ALL_TOOLS` + `BRAIN_SCHEMAS` export; `domain_sentiment` passes in `mock` mode |
| **B — agent** | `feat/agent` | `src/agent/*` (all six), `tests/test_{parser,guard,loop}.py` | `src/tools/*`, `app.py`, `config.py`, `prompts.py`, `domain_client.py` | Real captured XML parses; brain turn ≈1 s; turn cap + deadline + partial-synthesis all tested against a fake brain; **synthesis runs unconditionally and is absent from `ALL_TOOLS`** |
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

  6. domain_sentiment — THE ONE TOOL THAT CALLS NEMOTRON. Read §10 in full before writing it.
     Setup_Instructions.md L95 requires article-grounded sentiment questions to route the
     retrieved AFR text AND the applicable RBA rate through DOMAIN_FT_MODEL, returning a
     sentiment classification (positive/negative/mixed) plus a likely market direction. Worth
     30 of the 150 public points (MHQ058, MHQ067, MHQ080).

     Signature: domain_sentiment(headline, article_text, publication_date, rba_rate) -> str
     Implementation: import src.domain_client and src.prompts (both FROZEN, base commit).
     Never import src/agent/synth.py — that is session B's file and may not exist yet.

     Hard constraints, all of them scored:
     - Returns a CLASSIFICATION, not an answer. Clamp output to 200 chars.
     - NO fabricated numeric forecast — explicitly prohibited by L95. MHQ067's reference
       hedges the direction ("mixed-to-down") and invents no figure.
     - Deny the call when no AFR article has been retrieved in this request's trace. That
       stops Qwen using Nemotron as a general answerer, which is the prohibited pattern.
     - It is a tool like any other: its output is a verified tool result that flows on into
       final synthesis. It does NOT replace final synthesis.

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
trace, so a Nemotron outage degrades to partial credit rather than zero. Import the prompts
from src/prompts.py and the client from src/domain_client.py; do NOT write your own of
either. Point llm mode at node1's base model (http://10.0.1.11:8001/v1) first — that
validates the wiring and collects the base side of the model-quality comparison.

READ §10 BEFORE WRITING loop.py OR synth.py. Nemotron has two roles and you own only one:
- Role 1, YOURS: final synthesis. It is NOT a tool and it is NOT conditional. loop.py calls
  synth.write() after the loop exits, on EVERY request — including deadline breach, zero
  successful tool calls, and total brain failure. Never put synthesis in ALL_TOOLS and never
  let Qwen decide whether it runs. Challenge_Brief § Required Model Roles requires Nemotron
  to receive the accumulated results AFTER the loop; handout/03 has a section titled "Bad:
  Nemotron used as the planner and tool caller". Two 30% categories ride on this.
- Role 2, NOT yours: the domain_sentiment tool is session A's, in src/tools/registry.py.
  Both of you call the same frozen src/domain_client.py, so you never import each other.

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
verdict), and the sentiment path.

On the sentiment path, read §10. Nemotron is called in TWO places and you are training ONE
adapter for both, so the mix must cover both shapes:
- Role 1, final synthesis — SYNTH_SYSTEM/SYNTH_USER, the bulk of the mix.
- Role 2, the domain_sentiment TOOL that Qwen calls mid-loop — SENTIMENT_SYSTEM/
  SENTIMENT_USER, the 12% sentiment slice. Its output is a short CLASSIFICATION (sentiment
  + likely direction, <=200 chars), NOT a full answer, and NEVER a numeric forecast. Train
  it to that exact shape, because A's tool clamps the output and a rambling classification
  gets truncated. Import both prompt pairs from src/prompts.py verbatim.

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
- [ ] **Nemotron's two roles are both live and correctly separated** (§10): final synthesis
      fires on every `/query` and is **absent from `ALL_TOOLS`**; `domain_sentiment` appears in
      `BRAIN_SCHEMAS` and a sentiment question shows it in `tool_trace` followed by a
      synthesised answer. Grep the trace for one of MHQ058/067/080 as the evidence — the FT
      rubric asks for exactly this and the architecture rubric scores the separation.
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
| **F15** | The `role: "tool"` message may be rejected, and it is on B's critical path. | C1 | **RESOLVED 2026-07-31 — `role:"tool"` is ACCEPTED.** Probed against `agent-brain` through LiteLLM with no `tool_call_id` and no preceding `tool_calls` entry: HTTP 200, coherent answer, `finish_reason: stop`. **Session B appends tool results as `role:"tool"` messages directly.** No `<tool_response>` user-message fallback is needed. `budget.py` still owns the single `tool_result_message()` factory so the decision lives in one place. |
| **F16** | Review §A/§B propose step notes, ADRs, `CHANGELOG.md`, pre-commit + gitleaks, and CI. | A.2, B.1, B.3, V5–V9 | **Mostly rejected on the clock, two exceptions.** `README.md` says `docs/` is not required, and a 12-hour build does not fund an ADR set. **Adopted:** (a) Conventional Commits naming the plan section, already in the §6.1 preamble — it is free and it is what "repository structure" credit reads; (b) `training/MODEL_SUMMARY.md` and `training/COMPARISON.md`, which are already D's deliverables and carry most of the 30% model-quality evidence. **Rejected:** `docs/steps/*`, `docs/DECISIONS/*`, `CHANGELOG.md`, pre-commit, CI. The secret scan survives as a §8 gate item rather than a hook. |
| **F17** | Nemotron's two roles were conflated, and the sanctioned one was unbuilt. | — | **See §10.** Synthesis is never a tool; sentiment classification is. |

### Two review items promoted into session prompts

- **Peak RSS after the AFR index build** (C4) — session A measures and reports it. The index
  sits in unified memory next to `vllm-brain`'s preallocated KV cache on node0; an unmeasured
  multi-GB index OOMs the brain, not just the agent. Mitigation if tight: keep offsets into an
  mmap'd corpus rather than the text.
- **Thinking-off latency under 3-way concurrency** (C4) — session C measures it. §6 of the
  harness plan has the figure for thinking-**on** (20.7 s) but not off, and 3 concurrent
  requests is the graded condition.

---

## 10. Where Nemotron is a tool, and where it must not be (F17)

Nemotron has **two** roles. Conflating them is a scored failure; keeping them separate is worth
real points. This is the architecture decision, and it binds sessions A, B and D at once.

### Role 1 — final synthesis. NOT a tool. Unconditional.

```
question → Qwen plans/emits tool calls → runtime executes → results back to Qwen
         → loop until Qwen stops → Nemotron synthesises → POST /query
```

`Challenge_Brief.md` § Required Model Roles: Nemotron *"receives the question and accumulated
verified tool results **after** the Qwen reasoning loop, then synthesizes the final concise
financial-domain answer."* Same doc: *"Do not train Nemotron to replace the supplied Qwen
reasoning brain or **use Nemotron as the primary tool-calling model**."*

`handout/03_scoring_and_examples.md` has a section titled **"Bad: Nemotron used as the planner
and tool caller"** for `question → Nemotron → tool calls → Nemotron → answer`, because *"tool
selection becomes dependent on the Nemotron adapter instead of the stable Qwen agent-brain"*.

**Therefore synthesis is never in `ALL_TOOLS` and Qwen never decides whether it happens.**
`loop.py` calls it after the loop exits, on every request, including deadline breach and total
tool failure. Two scored items ride on this: "correct separation of responsibilities"
(architecture 30%) and "evidence that the final agent uses Qwen for planning and tool-call
generation, then routes the verified tool results through the fine-tuned Nemotron model for
final synthesis" (FT quality 30%).

### Role 2 — sentiment classification. **This one IS a tool Qwen calls.**

`Setup_Instructions.md` L95: *"Route article-grounded sentiment questions through your
fine-tuned domain model using the `DOMAIN_FT_MODEL` alias. The model should receive the
retrieved AFR article text and the applicable RBA rate as context and return a sentiment
classification (positive, negative, or mixed) and a likely market direction. Do not force the
model to emit a made-up numeric return or price forecast."* Restated in L120's checklist.

This is required, it is distinct from synthesis, and it is **currently unbuilt in every plan**.
It is worth 30 of the 150 public points (MHQ058, MHQ067, MHQ080) and Qwen has to orchestrate it
— retrieve the article, look up the as-of rate, then classify — so exposing it as a tool is the
correct shape:

```python
domain_sentiment(headline: str, article_text: str, publication_date: str, rba_rate: str) -> str
```

Constraints that keep it from sliding into Role 1:

- **It returns a classification, not an answer.** Sentiment label + likely direction, ≤200
  chars, no full prose answer. Clamp the output.
- **No fabricated numbers.** Explicitly prohibited by L95. `MHQ067`'s reference hedges the
  *direction* (`mixed-to-down`) and never invents a figure.
- **The guard denies it when the trace holds no retrieved article**, so Qwen cannot use it as a
  general-purpose answerer on a numeric question.
- **Final synthesis still runs afterwards.** The tool's output is just another verified tool
  result flowing into Role 1.

### Ownership — one new frozen file

Both `src/tools/registry.py` (A) and `src/agent/synth.py` (B) now need to talk to
`DOMAIN_FT_MODEL`, so by the §4 rule the client lands in the **base commit**:

- [ ] `src/domain_client.py` — **FROZEN.** The single `DOMAIN_FT_MODEL` caller. Honours
      `DOMAIN_PREDICT_MODE` (`mock` | `llm`), `SYNTH_TIMEOUT_S`, `temperature=0`. Exposes
      `complete(system, user, max_tokens)`. A's sentiment tool and B's `synth.py` both import
      it; **neither edits it.** Without this, A's registry would have to import B's `synth.py`,
      which may not exist yet — the exact cross-stream dependency Phase 0 exists to prevent.
- [ ] `src/prompts.py` gains `SENTIMENT_SYSTEM` / `SENTIMENT_USER` alongside the synthesis
      pair, same freeze, same reason: session D trains against them.

Sessions A and C each carry one measurement promoted out of the review — see the end of §9.

---

## 11. Build status — updated 2026-07-31

`main` at `f715a4e`. **139 tests green**, runnable with no corpus, no models and no
network. Phase 0 and the whole agentic loop are done and merged.

### Done

| Stream | State |
|---|---|
| **Phase 0** | Complete. Six frozen files, test scaffolding, 328K fixture corpus, both probes run |
| **B — agent** | **Complete and merged.** `parser` `brain` `guard` `budget` `synth` `loop`, 97 tests. Verified end-to-end against the live Qwen |
| **D — training** | Active on `feat/training`. 3,773 records generated against the shared `src/prompts.py` |
| **A — tools** | **Complete and merged.** `corpora` `rba` `afr` `asx` `registry`, 119 tests |
| **C — serving/eval** | `app.py` **done** (20 tests, verified over HTTP). `run_offline_eval.py` is the last gap |

### The measurement that should drive the next decisions

The full pipeline ran against live Qwen **and** live Nemotron on MHQ001. The tool
retrieved `41 / 175 / 20 / 21` — perfectly correct. Scored with the frozen
`component_judge`:

| Answer | Score |
|---|---|
| Base Nemotron, `DOMAIN_PREDICT_MODE=llm` | **0 / 10** |
| `reference_answer` | 10 / 10 |

**It had all four numbers right and still scored zero.** Cause isolated by
re-scoring three variants:

| Answer | Score |
|---|---|
| bullet list **+ self-contradiction** | **0/10** |
| same bullet list, contradiction removed | **10/10** |
| perfect prose **+ self-contradiction** | **0/10** |

So the sole cause is the disclaimer *"it is not possible to determine the exact
number"* — **not the formatting.** The judge is format-tolerant, consistent with
F6's `NATURAL_SYNTHESIS` verdict. This is §5.3's predicted base failure mode
(thinking out loud and hedging), and the defence is `SYNTH_SYSTEM`'s "Do not
hedge" plus the adapter, not output shape.

Three things follow. **The harness is not the bottleneck.** The 30% base-vs-FT
comparison now has a real base arm on the hardest question class. And session D's
adapter is what converts a correct retrieval into a scoring answer, which makes it
the highest-leverage remaining work after tools exist at all.

### Corrections to earlier state — do not re-derive these

- **`domain-ft` is NOT broken.** It answers from a separate vLLM instance
  (fingerprint `5a3f83cd` vs the brain's `6bc76779`). §2's blocker line and
  `FINETUNE_PLAN.md` §2.2 are **stale**. `DOMAIN_PREDICT_MODE=llm` works today.
- **`DATASET_DIR` is fixed.** It resolved from `__file__`, so inside a worktree it
  pointed at the worktree — where `data set/` does not exist, because it is
  untracked. Every session works in a worktree, so it was broken for all of them.
  `config.py` now follows the worktree gitfile back to the main checkout. Leave
  `DATASET_DIR` unset.
- **`DOMAIN_BASE_URL` exists** for §9's fallback: point Nemotron at a vLLM
  endpoint directly while the brain keeps using LiteLLM. Defaults to LiteLLM.
- **`role:"tool"` is accepted** (F15) and **judge calibration says
  `NATURAL_SYNTHESIS`** (F6). Both settled; see those rows.
- **ASX files are named by COMPANY, not ticker.** `Tabcorp-ASX-2015-2021.jsonl`
  holds `TAH.AX`, and 7 of 18 stems do not match their ticker prefix.
  `FINETUNE_PLAN.md` §2.4's `<TICKER>-` pattern is **wrong**. Read the `ticker`
  field; deriving it from the filename mis-keys `exclude_tickers` on the one
  ticker excluded in 5 of the 15 public questions.

### Next, in order of leverage

1. ~~**C — `app.py`**~~ **done.** Port stays closed for 30s while corpora warm,
   then `/health` answers in 0.4ms and is proven immune to both models and the
   corpus being gone. `POST /query` answered MHQ001 **10/10 with node1 down**.
2. **C — offline eval.** Two passes never interleaved, penalised score, failure
   attribution. The frozen `component_judge` and the `real_corpus` seam in
   `tests/test_registry.py` are both ready to reuse.
3. **Integration** — first real `POST /query`, then `submission.json`.

Sessions A and B are done. Every tool reproduces its reference answer, and the
loop drives them end to end against the live Qwen.

### Open decision that spans A, B and D

**The evidence-shape contract.** The adapter is trained on *structured* evidence —
`Verified evidence: {"volatility_pct_annualised": 14.46, ...}` plus a
`requested_components` list — but harness §3 says every tool returns `str`. This is
`FINETUNE_DATA_SOURCES.md` Q3: train on one `tool_results` format, serve another,
and the adapter degrades.

`synth.py` currently reads a structured payload off `tool.last_data` when present
and degrades to `{tool_name: result_string}` when absent, flagging the degraded
case in `Synthesis.error`. **The string form is a shape the adapter never saw**, so
it is a correctness risk rather than a neutral fallback.

Pick one before A finishes its tools:

- **A's tools expose `last_data`** (a dict of computed values) alongside their
  string return. Cheapest, preserves what the adapter was trained on, and the loop
  already carries it. **Recommended.**
- **D re-serialises** its 3,773 records against the harness's real string format.
  Truer to Q3's advice, but costs a regeneration and loses the typed evidence that
  makes the assistant targets precise.


---

## 12. Verified tool constants — do not re-derive

Measured on the real corpus 2026-07-31, each one before its assertion was written.
Every value below reproduces its published reference answer exactly.

**RBA** — 175 records, 3 Feb 2010 → 17 Jun 2026, UTF-8 BOM.

| Question | Value |
|---|---|
| MHQ001 | 41 of 175 changed; 20 increases, 21 decreases |
| lowest rate | 0.1, first effective 2020-11-04, 16 records |
| highest rate | 4.75, first effective **2010-11-03**, 11 records |
| longest hold | 1,036 days, 2016-08-03 → 2019-06-05, held 1.5 then 1.25 |
| 2022–23 tightening | 13 hikes, +4.25 pp, 0.1 → 4.35 |
| 2011–13 easing (MHQ035) | 8 cuts, −2.25 pp, 4.75 → 2.50 |

**ASX** — 18 tickers × 1,774 rows, 2015-01-02 → 2021-12-30.

| Question | Value |
|---|---|
| MHQ040 | 18 files, 1,774 rows each, 2 Jan 2015 → 30 Dec 2021 |
| MHQ045 | BHP.AX best 2018 +22.17%; AMP.AX worst −50.04% |
| MHQ049 | AMP.AX highest avg daily volume, 11,635,671.71 shares/day |
| MHQ076 | QBE.AX best non-Tabcorp 2021 return, +35.57% |
| MHQ072 | 5→12 Jun 2019: CBA +0.60, NAB +1.39, ANZ +0.89, BHP +5.89, RIO +2.91 |
| MHQ074 | equal-weighted non-Tabcorp basket +2.88% / +0.24% / −2.17% |

**AFR** — 219,538 records, 85 files, 373,012 tokens, 2015–2021, 92 undated.

| Term / question | Value |
|---|---|
| `unemployment` / `qbe` / `nab` | 5,997 / 1,546 / 7,372 — identical to a `\bword\b` scan |
| MHQ061 | peak year 2020 = 1,452; peak month 202005 = 218 |
| MHQ076 | QBE in 2021 = 369 |

### Semantics that were wrong in the plans, or easy to get wrong

- **ASX filenames are COMPANY names, not tickers.** `Tabcorp-ASX-2015-2021.jsonl`
  holds `TAH.AX`, and 7 of 18 stems mismatch their ticker prefix.
  `FINETUNE_PLAN.md` §2.4's `<TICKER>-` pattern is **wrong** — read the `ticker`
  field. Deriving it from the filename mis-keys `exclude_tickers` on the one
  ticker excluded in 5 of the 15 questions.
- **MHQ061 searches `unemployment`, not `nab`.** The plans quote its numbers
  (1,452 / 218) without naming the term; assuming `nab` gives 1,267 / 184 and
  matches nothing.
- **The MHQ072/074 basket is the equal-weighted MEAN of per-ticker returns**, not
  a price-weighted index.
- **`window_return` must snap to available trading days.** RBA effective dates are
  frequently not trading days, so an exact-match window returns nothing on exactly
  the cross-dataset questions it exists for.
- **The AFR index is mandatory.** A full regex scan measures **36s per pattern**
  and `re` does not release the GIL, so neither caching nor threads help. One
  uncached AFR question would blow the 60s budget alone.
- **AFR index peak RSS is 905 MB** (review C4's measurement). Postings are
  `array("i")` because ~44M (token, record) pairs as boxed ints cost multiple GB,
  and each node's unified memory is SHARED with vLLM — an oversized index can OOM
  the brain, not just the agent.
- **Tools returning prose helps, but is NOT what fixes scoring.** An earlier note
  here claimed a `k=v` tool result caused the 0/10 on MHQ001. That was wrong —
  isolated, the cause is self-contradiction alone (see §11). Prose is still worth
  having as a better exemplar for the model, and `deterministic_fallback` scores
  **10/10** precisely because it never hedges.

### ⚠️ Open organizer question — date basis

`Challenge_Brief.md`'s partial-credit example marks `2010-11-03` wrong for the
4.75 first-effective date and says the judge expected `2010-11-02`. **2 Nov 2010
does not exist anywhere in the approved dataset**, though the brief's record count
(11) matches us. The RBA board meets Tuesday and the rate takes effect Wednesday,
so that example looks graded against the **announcement** date.

We assert the corpus, because the brief's *other* example (`0.1` → `2020-11-04`)
is accepted and we reproduce it exactly — shifting dates a day earlier would break
the case that works. If hidden date components grade against announcement dates
this costs points. `Setup_Instructions.md` closes by telling us to ask in exactly
this situation.

---

## 13. Testing on LangSmith

Two different things, and the second is the one that produces submittable evidence.

### Tracing — works today

```bash
export LANGSMITH_API_KEY='lsv2_pt_...'   # a PAT; read access is needed
unset LANGSMITH_WORKSPACE_ID              # a stale value 403s every READ
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=westpac-agentic-harness
export DOMAIN_PREDICT_MODE=llm
.venv/bin/python scripts/trace_smoke.py --only MHQ001,MHQ040
```

Each request yields one `agent.answer` root with `brain.plan` / `parser.parse` /
`guard.validate` / `tool.invoke` / `synth.write` children, so a lost component
attributes to a **layer** rather than to "the agent". Root metadata carries
`commit_sha`, `question_id`, `difficulty` and `domain_predict_mode`, which is what
makes a trace comparable across commits instead of an anecdote.

### Experiments — what `langchain-basics` actually does, and what we still need

`langchain-basics/evals/run-eval-offline.py` is the working local reference the
review told us to prefer over the docs. Its pattern:

1. `evals/dataset.py` → `to_langsmith_examples()`, upserted into a named dataset.
2. Evaluators live in their own module and are **uploaded and bound to the
   dataset**, so LangSmith applies them to every experiment on it automatically.
3. LLM judges use a `StructuredPrompt` pushed to the prompt hub with a Pydantic
   feedback schema (`reasoning` first, then the score field).
4. `aevaluate(target, data=dataset, experiment_prefix=...)` runs the whole set.

**Ours does not exist yet and belongs to session C.** Built on
`public_questions.jsonl` it gives the base-vs-fine-tuned artefact the 30%
model-quality category asks for: same dataset, same brain, same tools, same
commit, one variable — `experiment_prefix="nemotron-base"` versus
`"nemotron-ft"`. That is the controlled ablation `handout/03` explicitly requires.

Note `langchain-basics` calls `load_dotenv()`; our `config.py` deliberately does
not, so that the scored run's environment is always explicit. The `scripts/`
entry points are the right place to load `.env` if we want the convenience.

**Tracing stays OFF for the scored run** (F12). This whole section is dev-time
evidence gathering on the *public* questions, whose text is already ours.


---

## 14. node1 is down — session D has taken the GPU (2026-07-31)

`10.0.1.11:8001` returns nothing and LiteLLM reports
`InternalServerError ... Connection error` for `domain-ft`. The brain on node0 is
unaffected. This is the event §5.2 and §2 both warned about: **the base control
arm disappears the moment training starts.**

Consequences, in order of how much they cost:

1. **The base-vs-fine-tuned comparison's base arm must already be banked.** If it
   is not, it cannot be collected until node1 is free again, and it is half of a
   30%-weighted deliverable. Session D owns this.
2. **`DOMAIN_PREDICT_MODE=llm` currently degrades to the deterministic fallback.**
   That is not a failure — `POST /query` still answered MHQ001 **10/10** with
   node1 gone, because `deterministic_fallback` never hedges. §7's invariant is
   doing exactly what it was designed for.
3. **`DOMAIN_PREDICT_MODE=mock` must not be what ships.** The fallback scoring
   well is a safety net, not a substitute: shipping without a served adapter
   forfeits the entire 30% model-quality category regardless of the harness score.

Nothing in `src/` needs changing for this. It is recorded so nobody re-diagnoses
a 500 on `domain-ft` as a harness bug.
