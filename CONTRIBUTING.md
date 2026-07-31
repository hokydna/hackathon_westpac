# Version Control Workflow

This repository **is** the submission. Scoring pins a single commit SHA from `main`, and the
repo must stay **public** — no private repos, no collaborator-only access.

## Branches — one per stream of work

`src/SESSION_KICKOFF.md` §5 is the authority on who owns what. Four streams, four branches,
**file-disjoint by construction** — that disjointness is what lets four sessions run at once
without merge conflicts, so the ownership boundary matters more than the branch name.

| Branch | Stream | Owns | Never touches |
|---|---|---|---|
| `main` | — | Always submission-ready. The pinned submission SHA only ever comes from here. | — |
| `feat/tools` | **A** — tools | `src/tools/*` (incl. `registry.py`), `tests/test_{rba,asx,afr}.py` | `src/agent/*`, `src/app.py` |
| `feat/agent` | **B** — agent runtime | `src/agent/*` (not `tracing.py`), `tests/test_{parser,guard,loop}.py` | `src/tools/*`, `src/app.py` |
| `feat/serving-eval` | **C** — serving & eval | `src/app.py`, `src/eval/*` (not `component_judge.py`), `tests/test_app.py` | `src/agent/*`, `src/tools/*` |
| `feat/training` | **D** — fine-tuning | `training/*` entirely | everything under `src/` |

**Frozen files — every stream imports, none edits:** `src/config.py`, `src/prompts.py`,
`src/contracts.py`, `src/agent/tracing.py`, `src/eval/component_judge.py`,
`tests/conftest.py`. They land in the base commit on `main` before anyone forks. Changing one
mid-flight silently invalidates another stream's work — `prompts.py` in particular is what the
LoRA adapter is trained against.

Each stream works in its own worktree (`.worktrees/` is git-ignored). The branches already
exist, so no `-b`:

```bash
git worktree add .worktrees/tools feat/tools
```

**Never run two sessions in the same working tree** — they fight over the git index and
failures become unattributable.

Need a change in a file you don't own? Append it to `HANDOFF-<your-letter>.md` at the repo
root and keep going. Never "just quickly fix" another stream's file.

## Commits

Conventional Commits, imperative mood, subject ≤ 72 chars:

```
<type>(<scope>): <subject>

<why the change was needed, if not obvious from the subject>
```

Types: `feat`, `fix`, `docs`, `train`, `chore`, `refactor`, `test`, `perf`.
Scopes used here: `agent`, `retrieval`, `harness`, `training`, `package`, `submission`, `repo`.

`git commit` opens `.gitmessage` as a reminder — it is wired up via `commit.template`.

## Integration

Merge order is free — the branches are file-disjoint — but merge **one at a time**, never as an
octopus. An octopus merge aborts wholesale on any conflict and destroys the only diagnostic a
conflict carries here: *which* boundary leaked.

```bash
git switch main
for b in feat/tools feat/agent feat/serving-eval feat/training; do
  git merge --no-ff "$b" || { echo "BOUNDARY LEAKED: $b"; break; }
done
pytest
python -m src.eval.run_offline_eval      # the acceptance gate
```

**A merge conflict means the ownership boundary leaked. Fix the boundary — do not
hand-resolve and move on.**

PRs are optional on this clock; if you open one, say what changed and how it was verified
(pytest output, eval numbers, curl output), and keep the plan refs in the body.

## What must never be committed

- `data set/` — ~785M, ignored on purpose. Copy it in locally.
- Model weights or checkpoints — commit configs, scripts, metrics and a model summary instead.
- Endpoints containing real credentials, `.env` files, keys, tokens.

If git refuses to add something, that is the `.gitignore` doing its job. Do not `git add -f`.

## Before submitting

```bash
git switch main && git pull
git status --porcelain          # must be empty
git rev-parse HEAD              # -> commit_sha in submission.json
```

Update `submission.json` (`team_id`, `team_name`, `github_url`, `commit_sha`, `agent`, `model`),
commit that, then take the **new** SHA — the pinned SHA is the one that contains the final
`submission.json`, so it is always the last commit on `main`.

The full pre-submission gate — each item independently capable of zeroing a whole scoring
category — is `src/SESSION_KICKOFF.md` §8. Two that get forgotten: the repo must be **public**,
and `GET /health` must answer 200 **from another machine** using the IP in `submission.json`.
