# Review — Agent Harness Implementation Plan

**Reviewed artefact:** `src/Agent_Harness_Implementation_Plan.md` (513 lines, dated 2026-07-31)
**Reviewer:** Claude (Opus 5) session, 2026-07-31
**Review scope requested:** (1) per-step version control, (2) a per-step `.md` note recording what was
done and what gaps remain, (3) best-practice references, (4) full test coverage — unit, integration,
smoke, regression, e2e, (5) LangSmith evaluation on every agentic path, cross-checked against
`langchain-basics/`.

> ## ⚠️ Read this first — status of this document
>
> **This is a historical review, not an instruction set. `src/SESSION_KICKOFF.md` is the source
> of truth, and every finding below has been dispositioned in its §9** — adopted or rejected,
> with the reason. Read §9 first; come here only for the argument behind a finding.
>
> **Adopted:** C1 (the `role:"tool"` probe → kickoff F15), C2 (latency decomposition → F10),
> V12 (sequential merges → F14), A.2/D.3 (`requirements.txt`, `pyproject.toml`, tiny test
> fixtures → F11), E.3's points-weighted `component_recall`, G2/E.5's structural point that
> `config.py` and `tracing.py` must be in the base commit → F12.
> **Rejected on the 12-hour clock:** `docs/steps/*`, `docs/DECISIONS/*` (ADRs), `CHANGELOG.md`,
> pre-commit hooks, CI → F16. **Overruled:** C3's `MAX_TURNS = 5` (kickoff keeps 3, because
> turns ≠ tool calls → F9) and E.5's separate `feat/eval` session (kickoff F8: session C owns
> eval and is not waiting on anything).
>
> **§1's evidence table is pre-migration and now false.** The repo moved to
> `hokydna/hackathon_westpac`: the plan is committed, `.gitignore` exists, `data set/` is
> untracked, and `origin` is ours — so V1, V2, V3, V4 and V12 are all closed. See kickoff §2
> "Repo facts".
>
> **Section E is superseded** by
> `Cognitivo_Labs/Cognitivo_Labs/docs/superpowers/reviews/2026-07-31-evaluation-strategy-review.md`,
> which is deeper and correct where we disagree. Read that instead for anything about measurement.
>
> **The one thing I got wrong:** §E treats LangSmith as the eval harness. It must not be.
> `Challenge_Brief.md` § Rules requires "use only approved local datasets and services during
> official scoring", so the primary offline harness must run with the network down and the scored
> path must not trace to LangSmith. Demote LangSmith to **dev-only observability**; keep only
> `tool_trajectory` and `output_contains` from `langchain-basics/evals/evaluators-offline.py`, as
> pure functions. G3–G6 are superseded by that review's E1/E4; §E.4's "ask the organizers" is
> resolved — the conservative default is already determined.
>
> **Sections A–D and §C1–C4 stood, and have now been reconciled** into
> `src/SESSION_KICKOFF.md` §4 and §9. That reconciliation is done — do not redo it, and do not
> action anything from §A/§B/§G without checking §9 first, or you will spend the clock building
> an ADR set the team decided to skip.
>
> **Also corrected below:** `component_recall` must be points-weighted (see §E.3), and the plan's
> "5 easy / 7 medium / 3 hard" in §9 is wrong — measured: **4 easy / 7 medium / 4 hard**.

**Verdict:** the plan is strong on the two things that usually sink a 12-hour build — it has measured
numbers instead of estimates, and it has a file-ownership scheme that actually permits parallel work.
It is materially incomplete on the five things above. **LangSmith appears zero times in the plan**, and
the plan's own choice of a hand-rolled loop means nothing traces automatically, so this is not a
one-line fix. Three defects in the plan body (§C1–C3 below) can cost points directly.

---

## Contents

- [0. What the plan gets right](#0-what-the-plan-gets-right)
- [1. Evidence base for this review](#1-evidence-base-for-this-review)
- [A. Version control — gaps and the per-step protocol](#a-version-control--gaps-and-the-per-step-protocol)
- [B. Per-step notes — template and conventions](#b-per-step-notes--template-and-conventions)
- [C. Defects in the plan body](#c-defects-in-the-plan-body)
- [D. Test coverage — the missing layers](#d-test-coverage--the-missing-layers)
- [E. LangSmith — current state, gaps, wiring](#e-langsmith--current-state-gaps-wiring)
- [F. Best-practice references](#f-best-practice-references)
- [G. Prioritised actions](#g-prioritised-actions)

---

## 0. What the plan gets right

Worth stating, because the recommendations below are additive, not a rewrite.

| Strength | Where | Why it matters |
|---|---|---|
| Measured, not estimated | §6, §8 | `15.0s → 0.9s` for `enable_thinking:false`, `6.5s → <1ms` for the AFR index, and three reproduced reference answers. This is the difference between a plan and a wish. |
| One hazard, one file | §2 | XML parsing, thinking suppression, budget, model mode each land in exactly one module. Makes hour-9 debugging tractable. |
| Frozen contracts before forking | §3, §12.1 | The rule "every shared file must be complete in the base commit" is the correct rule for parallel sessions and is stated explicitly. |
| Never return a non-answer | §7 | Correctly derived from `validate.json` (`answer` is the only required field, `minLength: 1`) and the brief's rules. Degrades to partial credit instead of zero. |
| Tokenizer equivalence proved, not assumed | §8 | `[a-z0-9]+` vs `[a-z0-9']+` giving 7,372 vs 6,903 on `\bnab\b` is exactly the class of silent-wrong-answer bug that reproducibility rules exist to prevent. |
| §9 marks what the execution guide omits | §9 | `cycle_summary`, `describe`, `window_return`, `retrieve_by_headline`, `coverage` — five tools the metric list doesn't mention but the question bank needs. |

Keep all of it. Everything below is what is missing or wrong.

---

## 1. Evidence base for this review

> **Stale as of the repo migration.** Rows about `git log`, `git remote`, `.gitignore`,
> `git ls-files "data set"` and the untracked plan describe
> `cognitivo-aifactory/AI_Industry_Training_Hackathon` as it was on 2026-07-31 **before** the
> move to `hokydna/hackathon_westpac`. All of those conditions are resolved. Kept as the record
> of why the base commit looks the way it does.

Commands run and their results, so you can re-verify:

| Check | Result |
|---|---|
| `git log --oneline` in `AI_Industry_Training_Hackathon` | 5 commits, latest `4e91fe5` |
| `git status --short` | `?? src/Agent_Harness_Implementation_Plan.md` — **the plan itself is untracked** |
| `git remote -v` | `origin` → `git@github.com:cognitivo-aifactory/AI_Industry_Training_Hackathon.git` — organizer's repo only, no team remote |
| `git ls-files "data set" \| wc -l` | **106 files tracked** (~780MB) |
| `[ -f .gitignore ]` | **absent** |
| `grep -rni langsmith Participant_Package/ README.md src/` | **0 hits** |
| `curl https://api.smith.langchain.com/info` | **HTTP 200 in 0.22s** — egress to LangSmith cloud works |
| `LANGSMITH_API_KEY` in shell | **not set** |
| `langchain-basics/.env` | **absent** (only `.env.example`) |
| `langchain-basics/.venv` | **absent** |
| `~/team.env` | **absent** |
| `langchain-basics/requirements.txt` | has `langsmith>=0.9.8,<1.0`, `openevals`, `pytest~=8.3`, `pytest-asyncio~=0.24` |
| LangSmith hackathon workspace | `3cf529b1-af25-43e1-bb6d-06a2ce295d40` (from `langchain-basics/.env.example`) |

Documents read in full: the plan; `Challenge_Brief.md` (scoring + required roles); `Setup_Instructions.md`;
`handout/03_scoring_and_examples.md`; `submission-guide.md` (structure + checklist); `validate.json`;
`submission.json`; `langchain-basics/evals/{dataset,run-eval-offline,evaluators-offline,run-eval-online}.py`;
`langchain-basics/tests/conftest.py` and all test-module docstrings.

---

## A. Version control — gaps and the per-step protocol

### A.1 Gaps

| # | Gap | Consequence | Fix |
|---|---|---|---|
| V1 | The plan is untracked; nothing from this session is committed | The single planning artefact can be lost by one bad `rm` | Commit it as the first act of the base commit |
| V2 | No `.gitignore` exists | `data set/`, `.venv/`, `__pycache__/`, `.env` all commitable by accident; §12.2 assumes it exists | Create in base commit, exactly as §12.2 specifies |
| V3 | `data set/` is tracked, 106 files | ~2.3GB across three worktrees; must not ship in a public repo | `git rm -r --cached "data set"` in base commit |
| V4 | Only remote is the organizer's repo | One careless `git push` publishes to the organizers' repo | `git remote add team …` **and** `git remote set-url --push origin no_push` as a guard |
| V5 | Plan says "commit" per TDD step but defines no message convention, no branch→step mapping, no PR step | Judges assess "repository structure" and "reproducibility" (30%); an unstructured history reads as one big dump | Conventional Commits + step id; PR per owner branch even when solo |
| V6 | No CI | Nothing enforces green tests before merge; a broken merge is discovered at hour 11 | GitHub Actions running the unit+integration markers on push (needs D.3 fixtures to work without the dataset) |
| V7 | No secret-scan gate before the commit is pinned | Rubric: "no credentials … or machine-specific secrets"; `team.env` values leaking is a direct deduction | `pre-commit` with `gitleaks` + `check-added-large-files` |
| V8 | Eval results are not tied to commits | The 30% model-quality category wants base-vs-FT evidence that is *reproducible*; a score with no SHA is not evidence | Every eval run writes `logs/eval/<UTC>-<short-sha>.json` including the SHA, model aliases, and LangSmith experiment URL |
| V9 | `logs/` and `training/` hold only `.gitkeep` | Both are **required** deliverables per `submission-guide.md` | Populate as work happens, not at hour 11 |
| V10 | No artefact-size policy for adapter weights | A LoRA adapter committed directly bloats a public repo | `training/` holds configs, metrics, logs, and a model card — **not** weights; document where weights live |
| V11 | `submission.json` still contains mock values (`mock-team`, `0123…4567`, `172.20.x.x`) | A stale `commit_sha` means judges review the wrong tree; a stale endpoint fails the `/health` gate → **zero on 40%** | A `make freeze` target that writes the real SHA and re-validates, run last |
| V12 | §12.5 uses an octopus merge (`git merge feat/tools feat/agent feat/serving-eval`) | Octopus aborts entirely on any conflict, and you lose the per-branch signal about *which* boundary leaked | Merge one branch at a time; keeps §12.5's own diagnostic ("the ownership boundary leaked") usable |

### A.2 The per-step protocol

Make "version controlled" mean something checkable. One plan step = one branch commit = one note file.

```
1. git switch feat/<track>                  # already isolated per §12.3
2. write the failing test                   # confirm red
3. implement                                # confirm green
4. write docs/steps/step-NN-<slug>.md       # template in §B
5. git add <code> <tests> docs/steps/step-NN-<slug>.md
6. git commit -m "feat(tools): RBA deterministic metrics [step-3]

   Reproduces MHQ001: 41/175 changed, 20 up, 21 down.
   Gap: cycle_summary not yet implemented (see step note).

   Refs: plan §10 step 3"
7. push; open PR at end of track; squash-merge is fine, keep the step ids in the body
```

Rules that make it enforceable:

- **The step note ships in the same commit as its code.** Not a follow-up commit. If the note is
  missing, the step is not done.
- **Every commit body names the plan section it implements.** `Refs: plan §10 step 3`. Judges
  reading the log can then map history onto the design doc, which is exactly what
  "reproducibility" and "architecture explanation" credit rewards.
- **Every commit body has a `Gap:` line or the words `Gap: none`.** Forces the honest note.
- **No commit crosses an ownership boundary.** If a commit touches both `src/tools/` and
  `src/agent/`, the §12.1 rule has already been violated.
- **Tag the submission.** `git tag -a submission-v1 -m "..."` at the exact SHA in `submission.json`.

Suggested repo additions in the base commit:

```
.gitignore                 # data set/, .worktrees/, __pycache__/, *.pyc, .venv/, .env, .cache/
.pre-commit-config.yaml    # gitleaks, ruff, check-added-large-files --maxkb=5000
.github/workflows/ci.yml   # pytest -m "unit or integration or contract" + ruff
CHANGELOG.md               # Keep a Changelog format
Makefile                   # test / test-all / smoke / eval / freeze
docs/DECISIONS/            # ADRs (see §B.3)
docs/steps/                # per-step notes
requirements.txt           # PINNED — currently absent from the plan's module tree entirely
```

`requirements.txt` deserves a callout: **the plan's §2 module tree has no dependency manifest at
all.** No `requirements.txt`, no `pyproject.toml`, no Python version pin. "Reproducibility" is named
explicitly in the 30% architecture rubric. Copy the `langchain-basics` approach (`requirements.txt`
+ `pyproject.toml` with `requires-python`, `[tool.pytest.ini_options]`, `asyncio_mode = "auto"`).

---

## B. Per-step notes — template and conventions

The plan has exactly one document. That is right for planning and wrong for execution: at hour 9
nobody remembers which of six deliberate shortcuts is the one now biting. Three document types close
the gap.

### B.1 Step notes — `docs/steps/step-NN-<slug>.md`

```markdown
# Step 3 — Corpora loaders + RBA metrics

- **Plan ref:** §10 step 3 · **Owner:** A · **Branch:** `feat/tools`
- **Commits:** `abc1234`, `def5678`
- **Status:** done | done-with-gaps | blocked
- **Elapsed:** 55 min (planned 45)

## What this step delivers
One paragraph, in terms of behaviour, not files. "RBA metrics answer count/changes/extremes/
max_hold_streak/lookup_rate from the 175-row corpus, with as-of date semantics."

## Decisions taken here
- BOM handled by `encoding='utf-8-sig'` at load, so no call site deals with it. → ADR-0004
- RBA values kept as `str` in the corpus and coerced at metric boundaries, because the source is
  `"+0.25"` / `"0.00"`.

## Evidence
Commands and their real output — paste it, do not describe it.

    $ pytest tests/test_rba.py -q
    12 passed in 0.41s
    $ pytest tests/test_rba.py::test_mhq001_reference -q
    1 passed   # 41/175 changed, 20 increases, 21 decreases

LangSmith: n/a (no LLM on this path) — or the experiment URL when there is one.

## Gaps and known limitations
Be specific and blunt. This section is the point of the file.
- `cycle_summary` (§9 ★) NOT implemented. Blocks the 2022–23 tightening-cycle question class.
  Owner A, ~20 min.
- `lookup_rate` as-of semantics tested at exact-match and before-match; **untested at the
  first-record boundary** (a date earlier than 3 Feb 2010). Currently raises; should return a
  "no data" string per §7.
- Tests read the full corpus via `DATASET_DIR`, so they **cannot run in a worktree** (see D.3).

## Follow-ups
- [ ] `cycle_summary` — step 6 or a step 3b
- [ ] Boundary case for `lookup_rate` — 10 min
```

The `Gaps and known limitations` sections roll up into `docs/KNOWN_LIMITATIONS.md`, which the README
links. The architecture rubric names "documented limitations" as an assessed item — this makes it a
by-product of working rather than an hour-11 writing task. It is also the honest answer to MHQ090,
whose correct response *is* a justified refusal.

### B.2 A short README-adjacent set

- `README.md` — summary, architecture diagram, run instructions, endpoint notes, known limitations
  (all four named in the checklist).
- `docs/KNOWN_LIMITATIONS.md` — rolled up from step notes.
- `docs/TRAINING_SUMMARY.md` — data prep, hyperparameters, checkpoint selection rationale,
  base-vs-FT numbers. This is most of the 30% model-quality submission.
- `CHANGELOG.md` — Keep a Changelog.

### B.3 ADRs — `docs/DECISIONS/NNNN-<slug>.md`

The plan already contains five real architecture decisions, but they are buried in prose. Judges
assessing "clear end-to-end architecture" and "correct separation of responsibilities" reward
finding them as discrete, dated records:

| ADR | Decision | Source |
|---|---|---|
| 0001 | Hand-rolled tool loop over `create_agent()`, because `message.tool_calls` is always `null` under `--tool-call-parser hermes` with XML-emitting Qwen3.6 | §1.1 |
| 0002 | `chat_template_kwargs={"enable_thinking": False}` on every brain call — 15.0s → 0.9s | §1.2 |
| 0003 | Inverted index with `[a-z0-9]+` tokens, mandatory not optional — 6.5s → <1ms, and `'` excluded from the token class to match `\b` | §8 |
| 0004 | Tools defined with LangChain `@tool` but dispatched by our loop (hybrid) | §1 |
| 0005 | `/health` depends on neither models nor corpora, because it is a hard gate on the whole 40% | §5 |

Use Nygard's five-heading form (Title, Status, Context, Decision, Consequences). Each is ~15 lines.
**Add ADR-0006 for whatever you decide about LangSmith tracing on the scored path (§E.4)** — that one
has a privacy dimension and needs a written decision, not an implicit one.

---

## C. Defects in the plan body

These are not process gaps. They are things in the plan that will misbehave.

### C1 — The `role: "tool"` message may be rejected, and the plan has no fallback

§5 does `msgs.append(tool_message(budget.clamp(result)))`. But §1.1 establishes that
`message.tool_calls` is always `null` — the brain's XML arrives inside `message.content`. An
OpenAI-compatible `role: "tool"` message conventionally carries a `tool_call_id` that must correspond
to a `tool_calls` entry on the preceding assistant message. There is no such entry to point at.

Whether this 400s depends on how strictly LiteLLM validates and how Qwen's chat template renders tool
turns — Qwen-family templates typically wrap tool results in `<tool_response>` blocks without needing
an id, so it may well work. **The problem is that the plan doesn't know, and it is on the critical
path for owner B's step 4.**

Resolve it with a 2-minute probe before writing `loop.py`:

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

Record the outcome in the step-2 note and in ADR-0001's Consequences. If it 400s, append results as a
`user` message shaped like `<tool_response>…</tool_response>` instead. Either way, `budget.py` should
own a single `tool_result_message()` factory so the decision lives in one place — consistent with the
plan's own "one hazard, one file" principle.

### C2 — The deadline arithmetic can cross the 60s penalty threshold

§6 sets a **45s wall deadline from request start** and a **15s synthesis timeout**. Those compose to
60s+ before FastAPI serialisation, so the worst case lands on the wrong side of the 20%-deduction
line. The deadline is being applied to the wrong quantity: it should bound *when the loop must stop*,
leaving synthesis its full budget inside the 60s envelope.

```python
PENALTY_THRESHOLD_S = 60.0
SYNTH_TIMEOUT_S     = 15.0
SAFETY_MARGIN_S     =  5.0
LOOP_DEADLINE_S     = PENALTY_THRESHOLD_S - SYNTH_TIMEOUT_S - SAFETY_MARGIN_S   # 40.0
```

Assert the invariant in a test (`test_budget_invariants`) so nobody tunes one constant and silently
breaks the sum. Measured end-to-end is ~6–10s, so this costs nothing in the common case; it only
changes the pathological case, which is precisely where the penalty applies.

### C3 — `MAX_TURNS = 3` is probably tuned against the wrong constraint

§6 justifies `MAX_TURNS = 3` from "the handout advises ≤3 tool calls and warns that looping more than
5 times will exceed 60s". But the same handout reports the best team-01 run at **avg steps 4.65** and
says explicitly that "hard questions may need 3-5 tool calls: look up the range, get the specific
metric, verify the result". The alpha team's best checkpoint ran **3.59** steps.

The 60s guidance is a *latency* warning, and at ~1.0s per thinking-off brain turn latency is not what
binds — a 5-turn loop is ~5s of brain time. Capping at 3 turns risks truncating exactly the hard
cross-dataset questions where the marginal points are.

Recommend `MAX_TURNS = 5`, with wall-clock (C2's `LOOP_DEADLINE_S`) as the real governor and the turn
cap as a runaway backstop. Also note turns ≠ tool calls in the plan: one turn can emit several
`<tool_call>` blocks, so `steps` in the response and `MAX_TURNS` are different quantities and should
be documented as such.

### C4 — Smaller items

| Item | Note |
|---|---|
| **Index rebuild on restart** | §8: the index costs ~32s to build. `/health` is a hard gate checked at run start (§7 correctly makes `/health` corpora-independent, but a restart still means 32s of failing `/query` calls). Persist the index (`pickle`/`marshal`, keyed on a hash of corpus mtimes+sizes) so a restart is seconds. Cheap insurance on a 40% gate. |
| **Memory not measured** | §8 measures 771MB of AFR text in RAM but never the index size. Each node is a single GB10 with **unified memory shared with vLLM**, and `vllm-brain` is on node0 alongside the agent. An unmeasured multi-GB index next to a preallocated KV cache is an OOM risk that takes down the brain, not just the agent. Add "measure peak RSS after index build" to step 5 and record it in the step note. Mitigation if tight: keep only offsets into an mmap'd corpus rather than the text itself. |
| **Concurrency measured only for the wrong mode** | §6 has 20.7s for thinking-**on** under 3-way concurrency but no thinking-**off** figure under concurrency. The harness sends 3 concurrent questions. Measure it; it is the number that determines whether the 60s threshold is safe under load. |
| **No process supervision** | Nothing in the plan says how the service stays up during the scoring window, or how the endpoint is verified *from another machine* — which `handout/03` explicitly instructs ("Test your endpoint from a different machine before submitting"). Needs a systemd unit or tmux plus a remote smoke script (D.2). |
| **`domain-ft` alias mismatch** | §8 flags it correctly. Add it to `KNOWN_LIMITATIONS.md` and to the step-7 note with an owner and a deadline, because `DOMAIN_PREDICT_MODE=mock` at submission forfeits the 30% model-quality category. |

---

## D. Test coverage — the missing layers

### D.1 What the plan has, and what it calls tests

The plan does say "TDD throughout" and names good specific assertions (MHQ001 → 41/175/20/21; MHQ061
→ 2020=1,452, May 2020=218; tokenizer counts equal regex counts). What it lacks is a **taxonomy**: no
layer names, no markers, no separation between tests that need the 780MB corpus and tests that don't,
no coverage measurement, no CI, and no regression concept beyond three hand-picked fixtures.

`langchain-basics` already solves several of these and is the right pattern to copy:
`tests/conftest.py` puts the root on `sys.path` and sets a dummy `OPENAI_API_KEY` before import;
`pyproject.toml` declares `asyncio_mode = "auto"` and an `e2e` marker; `tests/test_e2e_memory.py`
auto-skips the whole module when no server is reachable so the default run stays green.

### D.2 Proposed layers

| Layer | Marker | Scope | Needs | Budget | Gate |
|---|---|---|---|---|---|
| **Unit** | `unit` | `parser` on real captured XML; `guard` coercion (`"2018"`→`2018`) and denial; `budget` clamp/deadline with a fake clock; RBA/ASX arithmetic; AFR tokenizer | nothing — tiny fixtures | <5s | every commit, CI |
| **Integration** | `integration` | `loop` + fake brain + real tools on fixtures; `registry` schema derivation; `synth` mock mode; `app` via `TestClient` with a stubbed loop | fixtures only | <30s | every commit, CI |
| **Contract** | `contract` | response validates against `validate.json` with `jsonschema`; `tool_trace` entries match `answer_template.json`; `BRAIN_SCHEMAS` is a valid OpenAI tools array | fixtures only | <5s | every commit, CI |
| **Regression (golden)** | `regression` + `needs_dataset` | all 15 public questions through the **tool layer**, compared to committed golden JSON; the §8 reference fixtures | full corpus | ~60s | pre-merge, pre-submission |
| **Property** | `unit` | index-vs-regex equality over ~200 terms sampled from the vocabulary — generalises the `nab`/`qbe`/`unemployment` spot-checks | full corpus (subset ok) | ~30s | pre-merge |
| **Smoke** | `smoke` | `scripts/smoke.py --base-url <ip>` — `GET /health` 200, one `POST /query`, schema-valid, latency logged. Runnable **from another machine** | live service | <30s | after every deploy, before submitting |
| **E2E** | `e2e` | real brain + real tools + real synth on ~3 questions; auto-skip when LiteLLM unreachable | live models | ~30s | pre-merge when the cluster is up |
| **Perf / concurrency** | `perf` | 3 concurrent `/query` with distinct questions → no crossed answers; p95 asserted `< LOOP_DEADLINE_S + SYNTH_TIMEOUT_S` | live service | ~60s | pre-submission |
| **Eval (LangSmith)** | — | 15 public questions with component graders; the acceptance gate | LangSmith + models | ~3min | pre-submission, and after every prompt change |

`make test` runs `-m "unit or integration or contract"`. `make test-all` adds the rest. CI runs the
first three only — which is why D.3 is load-bearing.

### D.3 The fixture problem — this one is structural

§12.2 untracks `data set/`, and §12.3 creates worktrees. **Worktrees only contain tracked files, so
no worktree will have the corpus.** §12.2 mitigates with a `DATASET_DIR` env var pointing at the main
checkout's absolute path — fine on this box, but it means:

- owner A's tests are unrunnable anywhere except this machine;
- CI can never run them;
- and a judge cloning the public repo cannot run the test suite at all, which undercuts the
  "reproducibility" item in the 30% rubric.

Fix: commit a **tiny** fixture corpus, `tests/fixtures/` (<1MB total) —

```
tests/fixtures/dataset/
├── RBA Rates/RBA-rates.jsonl          # ~20 rows, BOM preserved, spanning ≥1 hike and ≥1 cut
├── ASX/{BHP,TAH}-ASX-fixture.jsonl    # 2 tickers × ~60 rows (one is TAH.AX, for exclude_tickers)
└── AFR/AFR_fixture.jsonl              # ~200 articles incl. known nab/qbe/apostrophe cases
```

Every unit/integration/contract test points `DATASET_DIR` at that. Only `needs_dataset` tests use the
real 780MB corpus. Deliberately include in the fixture: a BOM, a `"+0.25"` string change, an
apostrophe-adjacent `nab` occurrence, a `PUBLICATIONDATE` at a month boundary, and a `TAH.AX` row —
i.e. one row per §8 gotcha, so each gotcha has a permanent regression test rather than a note.

### D.4 Specific tests the plan does not name

Grouped by the failure they'd catch.

**Correctness of the §8 gotchas** — one test each: BOM stripped; RBA `"3 Feb 2010"` parsed without
ISO assumptions; signed-string change values (`"+0.25"`, `"0.00"`) coerced correctly; AFR
`PUBLICATIONDATE` sliced (`[:4]`, `[:6]`) not date-parsed; `.AX` suffix preserved; `exclude_tickers`
honoured by *every* ASX metric (parametrise over the metric list — the plan says it is "a first-class
argument on every ASX metric", so test it on every one).

**`lookup_rate` as-of semantics** — the plan flags that nearest-match can return a *future* decision.
Test: exact match; a date between two decisions (returns the earlier); a date after the last decision;
a date before the first (must return a "no data" string per §7, not raise).

**Parser hostility** — malformed/unclosed `<tool_call>`; two calls in one reply; JSON with trailing
commas or single quotes; unicode in arguments; a call for a tool that doesn't exist; empty content.
Each must produce a structured outcome, never an exception, because §7 promises the brain can replan.

**Guard as a security boundary** — a disallowed tool name executes nothing and returns a structured
denial; args that fail Pydantic after coercion follow the same path; and (worth adding) a tool name
that differs only by case or whitespace is denied, not silently normalised into a match.

**Budget** — clamp at exactly **1,200** chars (`config.TOOL_RESULT_CHAR_CAP`; the 2,000/1,500 this
review was written against is superseded by kickoff F2, because training and inference must clamp
identically); the message list stays under 3,000
tokens by dropping *oldest tool results first* and **never** the system prompt or the question (assert
both survive); deadline breach synthesises from a partial trace. Use a monkeypatched clock, not
`sleep`, so the suite stays fast.

**Failure-mode matrix from §7** — one test per row of that table, each asserting a **non-empty
`answer`**. §7's "never return a non-answer" is the single most valuable invariant in the design and
should be the most heavily tested thing in the repo. A parametrised test over injected failures
(brain timeout, brain 500, synth timeout, synth 500, empty tool result, denied tool, deadline breach)
asserting `len(answer) >= 1` and schema validity is maybe 30 lines and covers the whole table.

**Concurrency** — 3 concurrent `/query` with distinguishable questions; assert each answer matches its
own question (§7 claims safety "by construction"; construction claims still need one test).

**Latency** — assert p95 over the 15 public questions is under the C2 budget. A measurement that isn't
an assertion won't survive a late prompt change.

### D.5 Coverage and determinism

- `pytest-cov` with `fail_under` per package: **90% on `src/tools/`** (pure, deterministic, and where
  the hidden-question points come from) and **80% on `src/agent/`**. Don't chase coverage on `app.py`.
- Judge determinism: any LLM judge runs at `temperature=0` with a **pinned model alias**, and the
  judge model is recorded in the eval output alongside the SHA (V8). Otherwise "our score went from
  74% to 71%" is uninterpretable.
- Golden-file discipline: regenerating goldens is an explicit `make goldens` step, and the diff is
  reviewed in the PR. Never auto-refresh on failure.

---

## E. LangSmith — current state, gaps, wiring

### E.1 Framing first

Nothing in the hackathon documents requires LangSmith — `grep -rni langsmith` over
`Participant_Package/` returns **0 hits**. So this is our own instrumentation choice. It pays into two
scored categories rather than being a requirement of its own:

- **30% model quality** — "Quantitative and qualitative comparison with the supplied base model on
  held-out or validation examples" and "evaluation evidence". Two LangSmith experiments over one
  dataset *is* that artefact, in a form a judge can read in 30 seconds.
- **30% architecture/repo** — "code quality and reliability", "reproducibility", "non-sensitive logs".

It does **not** pay into the 40% hidden-question score directly, and it must not endanger it (see
E.4). That ordering should drive how much time it gets.

The organizers' own `langchain-basics/.env.example` ships a hackathon LangSmith workspace ID
(`3cf529b1-…`), so the platform is clearly sanctioned for the event.

### E.2 Measured state of the plumbing

| Item | State |
|---|---|
| Egress to `api.smith.langchain.com` | **works** — HTTP 200 in 0.22s |
| `LANGSMITH_API_KEY` | **not set** in this shell |
| `langchain-basics/.env` | **absent** — only `.env.example` |
| `langchain-basics/.venv` | **absent** — nothing installed, so nothing has ever run |
| `~/team.env` | **absent** (§11 already flags this) |
| `langsmith` in any src dependency manifest | **no manifest exists** (see A.2) |
| LangSmith in the harness plan | **0 mentions** |

So: the network is fine and the reference code is good; the credential, the venv, and any mention in
the design are all missing. That's ~15 minutes of plumbing plus real design work on tracing.

### E.3 Gaps

**G1 — Nothing will trace automatically. This is the big one.**
ADR-0001's hand-rolled loop is the right call for tool dispatch, but it also means the plan gives up
the automatic tracing that `create_agent()` + `ChatOpenAI` would have provided for free. With a
hand-rolled loop over raw HTTP or a bare OpenAI client, `LANGSMITH_TRACING=true` produces **nothing**.

Fix — one new file, consistent with §2's one-concern-per-file structure:

```
src/agent/tracing.py    # the only place LangSmith is imported
```

- `@traceable(run_type="chain", name="agent.answer")` on `loop.answer` — the root span, one run per
  `/query`.
- `@traceable(run_type="llm", name="brain.plan")` on `brain.plan`, **or** wrap the client with
  `langsmith.wrappers.wrap_openai` so token counts and latency come through as a real LLM span.
- `@traceable(run_type="llm", name="synth.write")` on `synth.write`, plus a separate span for the
  §9 sentiment path — they are different prompts and want separate metrics.
- `@traceable(run_type="parser")` on `parser.parse` — the XML parse is hazard #1; when it silently
  returns `[]` you want to see the raw content that produced it.
- `@traceable(run_type="tool")` on the dispatch site in `guard`/`loop`. LangChain `@tool` objects do
  emit runs, but only inside an active traced parent — which the root `@traceable` provides.
- Attach metadata on the root span: `commit_sha`, `brain_model`, `domain_ft_model`,
  `DOMAIN_PREDICT_MODE`, `turns`, `deadline_breached`. That metadata is what makes a trace comparable
  across commits, and it closes V8 from the LangSmith side.

Make every decorator a **no-op passthrough when `LANGSMITH_TRACING` is falsy**, so the scored path can
be de-instrumented by one env var without touching code (see E.4).

**G2 — Config and dependencies.**
`config.py` (which §12.1 says must be *complete* in the base commit, and which A and B may not edit)
has no LangSmith variables. Add them now or you will have to violate the base-commit rule later:
`LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_API_KEY`,
`LANGSMITH_WORKSPACE_ID`, `EVAL_JUDGE_MODEL`. Add `langsmith>=0.9.8,<1.0` (and optionally
`openevals`) to `requirements.txt`.

Use one project per phase so the scored run isn't polluted by development traffic:
`hackathon-dev`, `hackathon-eval-offline`, `hackathon-scored-run`.

**G3 — `eval/run_public.py` is disconnected from LangSmith.**
§10 step 9 describes a local script with per-component graders. That's a reasonable fallback but it
throws away versioned datasets, experiment comparison, and the shareable artefact. Mirror
`langchain-basics/evals/` — it is a working, non-trivial reference for exactly this:

| Reference file | What to copy |
|---|---|
| `evals/dataset.py` | `Example` dataclass + `to_langsmith_examples()`. Ours is built from `Participant_Package/public_questions.jsonl`, carrying each question's `grading.components` as reference outputs plus `expected_tools`/`expected_tool_args`. |
| `evals/run-eval-offline.py` | Dataset upsert, code+LLM evaluator upsert, **binding evaluators to the dataset via run rules**, then `aevaluate(...)` with `experiment_prefix` and metadata. |
| `evals/evaluators-offline.py` | Standard-library-only code evaluators returning `{"key","score","comment"}` — required because uploaded evaluators run in a restricted environment. `tool_trajectory` is directly reusable. |
| `evals/run-eval-online.py` | Uploading an online code evaluator and attaching it to a **tracing project** with a sampling rate. |

Evaluators we need that the Chinook set doesn't have:

- `component_recall` — **points-weighted**, not fact-counted: `sum(points where verdict==YES) /
  sum(max_points)`. Measured over the real 15, component granularity varies sharply and a plain
  fraction-of-facts metric misreads it:
  - **Four questions (MHQ001, MHQ040, MHQ049, MHQ076) have a single component worth all 10 points**,
    each bundling 2–4 independent numbers. Getting three of four numbers right scores **zero**.
    That is 26.7% of the public points behind four all-or-nothing gates, and it is the strongest
    constraint on fine-tuning data design — the model must emit every sub-clause of a compound fact
    in one sentence, in the reference's own shape.
  - **Four questions (MHQ055, MHQ067, MHQ074, MHQ090) additionally carry `partial_credit` schedules**
    with per-answer point values. MHQ090's refusal is worth 10 points across three graded parts, of
    which "No." alone earns 3.33 — the evidence-boundary reasoning is worth more than the verdict.
  - **`tolerance_note` is tiered across three variants** (strict / numeric-tolerant / sentiment-
    tolerant) and must be read per question, not applied globally.

  Also report **compound-component hit rate** separately: one regression there moves the total more
  than any medium question can.
- `no_hedging` — `handout/03` states plainly that "approximately 41" and "roughly 20" are **not
  accepted**. A regex evaluator over hedge words is cheap and catches a whole failure class.
- `latency_under_60s` — the 20% penalty as a first-class score, not a footnote.
- `tool_used` — asserts `tool_trace` is non-empty; the handout's example 1 shows a 0% run whose only
  fault was calling no tools.
- `schema_valid` — `validate.json` conformance, so a malformed run shows as red in the experiment view.

**G4 — Replicate the organizer's judging protocol, don't approximate it.**
`handout/03` specifies precisely how you will be graded: the judge is **Qwen3.6-35B-A3B-FP8 via the
private `agent-brain` service**, it receives the question, your answer, and **each expected fact one
at a time**, and replies **YES or NO**. That is not the same as a holistic rubric judge like the
Chinook `CorrectnessFeedback` example.

So our LLM judge should be one-fact-at-a-time, YES/NO, pointed at `agent-brain` through LiteLLM, at
`temperature=0`. Then the LangSmith number *predicts* the leaderboard number instead of merely
correlating with it. It also costs nothing — it's a local model. This is the single highest-value
LangSmith item on the list.

**G5 — No online evaluators on the live run.**
Copy `run-eval-online.py` and attach guardrail evaluators (`schema_valid`, `no_hedging`, `tool_used`,
`latency_under_60s`) at `sampling_rate=1.0` to `hackathon-scored-run`. During the actual scored window
that gives you a live read on availability, latency, and whether tools are firing — which is
otherwise invisible until the private report arrives. Subject to E.4.

**G6 — The base-vs-FT comparison has no home.**
§10 step 7 says to point `llm` mode at node1's base model "to collect the base-model side of the
comparison evidence", and §8 notes node1 is a free control endpoint. But the plan never says where
that evidence lands. Concretely:

```
experiment_prefix="nemotron-base"   # DOMAIN_FT_MODEL → Llama-3.1-Nemotron-Nano-8B-v1 (node1 base)
experiment_prefix="nemotron-ft"     # DOMAIN_FT_MODEL → the adapter
```

Same dataset, same brain, same tools, same commit — only the synthesis model differs. That is the
controlled ablation `handout/03` explicitly asks for ("teams must separately compare base Nemotron and
fine-tuned Nemotron while keeping the Qwen routing and tool pipeline fixed"), and it is a direct
answer to the plan's §8 note that the historical table is *not* a controlled model-only ablation.
Export both to `logs/eval/` as CSV/JSON and link the LangSmith comparison view from
`docs/TRAINING_SUMMARY.md`.

**G7 — The sentiment path is unevaluated.**
§9's "second role for the fine-tuned model" (AFR text + applicable RBA rate → positive/negative/mixed
+ likely market direction, and explicitly **no** fabricated numeric forecast) is a distinct prompt
path with distinct failure modes. It needs its own small dataset (~5 examples) and its own evaluators:
label agreement, and a `no_numeric_forecast` check. Neither the plan nor the eval design mentions it.

**G8 — Plumbing not provisioned.** From E.2: get a `LANGSMITH_API_KEY` into the shell (not into
`.env`, per `langchain-basics`'s own instruction and per `Setup_Instructions.md`'s "keep credentials in
environment variables"), create the venv, install `langsmith`, and confirm with one trivial traced
run. ~15 minutes, and it unblocks G1–G7.

Minor: `langchain-basics/evals/` contains `run-evaltied-with-dataset.p__` and
`evlauator-tied-with-dataset.p__` — disabled/misnamed variants. Don't copy those; the live pattern is
`run-eval-offline.py` + `evaluators-offline.py`.

### E.4 The one thing to decide deliberately — tracing on the scored path

Two risks, both worth a written decision (ADR-0006) rather than a default:

1. **Latency.** The LangSmith SDK batches in background threads, so overhead is small — but it is not
   zero, and the 60s threshold is a scored boundary. Rules: never call `client.flush()` on the request
   path; keep `LANGSMITH_TRACING` as a kill switch; and measure `/query` latency **with tracing on and
   off** in step 10, recording both in the step note. If tracing costs anything measurable, ship the
   scored run with it off and keep it on for dev and eval.

2. **Confidentiality.** Tracing the scored run sends **hidden question text and AFR article content**
   to a third-party cloud. `Setup_Instructions.md` says not to "silently replace a missing organizer
   service with an external one", and the rubric prohibits committing "organizer-only evaluation
   material". Neither line squarely forbids tracing, and the organizers ship a LangSmith workspace ID
   themselves — but hidden evaluation questions are a different category of data from your own dev
   traffic. **Ask an organizer and record the answer.** `Setup_Instructions.md` ends with exactly that
   instruction for unclear cases.

   Fallback that needs no permission: trace `hackathon-dev` and `hackathon-eval-offline` fully, and for
   the scored run either disable tracing or trace **metadata only** — latency, turn count, tool names,
   schema validity, `deadline_breached` — with question and article text redacted. You keep the
   operational dashboard and send no organizer content off-box. `evaluators-offline.py`'s `no_pii`
   evaluator shows the shape: it sends *counts*, not text.

### E.5 Ownership

The eval work is ~2–3 hours and is file-disjoint from `src/tools/` and `src/agent/`, so per §12.6 it
should be its own session on `feat/eval`, not squeezed into owner C alongside `app.py` and hardening.
The one cross-boundary piece is `src/agent/tracing.py` plus the `@traceable` decorators, which live in
owner **B**'s files. Handle it the way §12.1 prescribes: land `tracing.py` (with no-op fallbacks) in
the **base commit**, so B merely decorates and never has to edit a shared file.

---

## F. Best-practice references

Grouped by the part of the review they support.

**Version control and change records**
- Conventional Commits — <https://www.conventionalcommits.org> (commit message grammar; the `[step-N]` suffix is our local extension)
- Keep a Changelog — <https://keepachangelog.com>
- Semantic Versioning — <https://semver.org>
- Trunk-based development — <https://trunkbaseddevelopment.com> (short-lived branches; matches §12's worktree-per-track model)
- `pre-commit` — <https://pre-commit.com>; gitleaks — <https://github.com/gitleaks/gitleaks>

**Decision and documentation structure**
- Michael Nygard, *Documenting Architecture Decisions* (2011) — the ADR form used in §B.3
- ADR organisation and tooling — <https://adr.github.io>
- Diátaxis — <https://diataxis.fr> (tutorial / how-to / reference / explanation; the plan is
  *explanation*, step notes are *reference*, README is *how-to* — a useful check that you're not
  writing four documents that all do the same job)
- arc42 — <https://arc42.org> (if you want a fuller architecture-document skeleton for the README)

**Testing**
- Martin Fowler, *TestPyramid* and Ham Vocke, *The Practical Test Pyramid* — <https://martinfowler.com/bliki/TestPyramid.html>
- Martin Fowler, *Mocks Aren't Stubs* — relevant to the plan's "fake brain", which is a stub and should stay one
- pytest markers and fixtures — <https://docs.pytest.org> (`-m` selection, `pytest.importorskip`, module-level skip)
- Hypothesis (property-based testing) — <https://hypothesis.readthedocs.io> — for D.2's index-vs-regex equivalence property
- Approval / golden-file testing — <https://approvaltests.com> — for the 15-question golden regression set
- Consumer-driven contract testing — <https://docs.pact.io> — the concept behind the `validate.json` contract layer, even though a `jsonschema` assertion suffices here
- Google, *Site Reliability Engineering* / *SRE Workbook*, error-budget chapters — the framing behind C2's explicit latency-budget decomposition

**Config, secrets, packaging**
- The Twelve-Factor App, factor III (config in the environment) — <https://12factor.net/config> — matches `Setup_Instructions.md`'s "keep all credentials and endpoint URLs in environment variables"
- OWASP ASVS, V14 (configuration) — if you want a checklist for the secret-scan gate

**LangSmith / LangChain**
- LangSmith docs — <https://docs.langchain.com/langsmith> (evaluation concepts, `@traceable`, `wrap_openai`, datasets and experiments, online evaluators and run rules, experiment comparison)
- `openevals` — <https://github.com/langchain-ai/openevals> — prebuilt judges; already in `langchain-basics/requirements.txt`
- **The strongest reference here is local:** `langchain-basics/evals/` is a working implementation of dataset upsert, code + LLM evaluators, dataset binding via run rules, and online evaluator attachment. Prefer it over docs where they disagree — it is known to work against this event's workspace.
- LangChain `@tool` and structured-output docs — for the §3 "every tool returns `str`" convention and `BRAIN_SCHEMAS` derivation

**Fine-tuning evidence (for `docs/TRAINING_SUMMARY.md`)**
- Model Cards for Model Reporting (Mitchell et al., 2019) — a good skeleton for the model summary the rubric asks for
- `handout/01_training_guide.md` — the authoritative hyperparameter baseline; §11 of the plan already cites the right values (`LORA_RANK=32`, `LR=5e-5`, `MAX_SEQ_LEN=512`, `MAX_STEPS=100`, `WARMUP_STEPS=50`, `CHECKPOINT_EVERY=20`)

---

## G. Prioritised actions

**P0 — before anyone forks a session** (~60 min, one person, extends §12.2's base commit)

| # | Action | Ref |
|---|---|---|
| 1 | Commit the plan. Create `.gitignore`, `git rm -r --cached "data set"`, add the `team` remote, block pushes to `origin` | V1–V4 |
| 2 | `requirements.txt` (pinned) + `pyproject.toml` with `asyncio_mode=auto` and markers `unit, integration, contract, regression, smoke, e2e, perf, needs_dataset` | A.2, D.2 |
| 3 | `config.py` **complete**, including the `LANGSMITH_*` block and C2's derived budget constants | G2, C2 |
| 4 | `src/agent/tracing.py` with no-op fallbacks, so B only decorates | G1, E.5 |
| 5 | `tests/conftest.py` + `tests/fixtures/` tiny corpus (one row per §8 gotcha) | D.3 |
| 6 | `docs/steps/TEMPLATE.md`, `docs/DECISIONS/0001…0005`, `docs/KNOWN_LIMITATIONS.md`, `CHANGELOG.md` | B.1, B.3 |
| 7 | `.pre-commit-config.yaml` (gitleaks, ruff, large-files) + minimal CI | V6, V7 |

**P1 — during the build**

| # | Action | Owner | Ref |
|---|---|---|---|
| 8 | Run the `role:"tool"` curl probe **before** writing `loop.py`; record the result in ADR-0001 | B | C1 |
| 9 | `MAX_TURNS = 5`; wall-clock as the real governor; `test_budget_invariants` | B | C2, C3 |
| 10 | Persist the AFR index to disk; measure and record peak RSS | A | C4 |
| 11 | Decorate `loop`/`brain`/`parser`/`synth` with `@traceable`; root-span metadata incl. `commit_sha` | B | G1 |
| 12 | Ask organizers about tracing hidden questions; write ADR-0006 either way | C | E.4 |
| 13 | A step note per completed step, in the same commit as the code | all | B.1 |
| 14 | The §7 failure-matrix parametrised test — one row per failure, all asserting non-empty `answer` | B | D.4 |
| 15 | Measure thinking-off latency under 3-way concurrency | C | C4 |

**P2 — the eval track (`feat/eval`, own session, ~2–3h)**

| # | Action | Ref |
|---|---|---|
| 16 | LangSmith plumbing: API key in shell, venv, `langsmith` installed, one traced run verified | G8 |
| 17 | `eval/dataset.py` from `public_questions.jsonl`, carrying `grading.components` as reference outputs | G3 |
| 18 | Code evaluators: `component_recall`, `no_hedging`, `tool_used`, `latency_under_60s`, `schema_valid` (stdlib only) | G3 |
| 19 | LLM judge replicating the organizer protocol — one fact at a time, YES/NO, `agent-brain`, `temperature=0` | **G4 — highest value** |
| 20 | `eval/run-eval-offline.py` on the `langchain-basics` pattern; results to `logs/eval/<UTC>-<sha>.json` | G3, V8 |
| 21 | Base-vs-FT experiments (`nemotron-base` / `nemotron-ft`), exported and linked from `TRAINING_SUMMARY.md` | G6 |
| 22 | Online evaluators attached to `hackathon-scored-run` (subject to #12) | G5 |
| 23 | Sentiment-path dataset + `no_numeric_forecast` evaluator | G7 |
| 24 | `scripts/smoke.py`, run from a different machine | D.2, C4 |
| 25 | `make freeze` — real SHA into `submission.json`, tag `submission-v1`, re-validate, secret scan | V11 |

### The three things most likely to cost points if skipped

1. **C1 (the `role:"tool"` probe)** — 2 minutes now, or a blocked owner B at hour 4.
2. **G4 (organizer-protocol judge)** — without it you are optimising against a proxy metric and won't
   know your real score until it's final.
3. **V11 (`make freeze`)** — a stale `commit_sha` gets the wrong tree judged; a stale endpoint fails
   the `/health` gate and **zeroes the entire 40% category**. It is the cheapest catastrophic failure
   in the whole project.

---

*This review assesses the plan as written. It does not change any file under `src/`.*
