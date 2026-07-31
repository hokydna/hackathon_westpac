# Version Control Workflow

This repository **is** the submission. Scoring pins a single commit SHA from `main`, and the
repo must stay **public** — no private repos, no collaborator-only access.

## Branches

| Branch | Rule |
|---|---|
| `main` | Always submission-ready. Every commit here should leave `submission.json` valid and the agent runnable. This is the only branch a submission SHA is ever taken from. |
| `feat/<slug>` | New capability — agent endpoints, tools, retrieval, harness wiring. |
| `fix/<slug>` | Bug fixes. |
| `train/<slug>` | Fine-tuning runs, data prep, eval sweeps. |
| `docs/<slug>` | Plans, reviews, handout notes. |

Branch off `main`, keep it short-lived, merge back via PR, delete after merge.

```bash
git switch main && git pull
git switch -c feat/query-endpoint
# ... work ...
git push -u origin feat/query-endpoint   # then open a PR
```

Long-running parallel work goes in a worktree (`.worktrees/` is ignored):

```bash
git worktree add .worktrees/train-lora train/nemotron-lora
```

## Commits

Conventional Commits, imperative mood, subject ≤ 72 chars:

```
<type>(<scope>): <subject>

<why the change was needed, if not obvious from the subject>
```

Types: `feat`, `fix`, `docs`, `train`, `chore`, `refactor`, `test`, `perf`.
Scopes used here: `agent`, `retrieval`, `harness`, `training`, `package`, `submission`, `repo`.

`git commit` opens `.gitmessage` as a reminder — it is wired up via `commit.template`.

## Pull requests

- One logical change per PR; rebase on `main` rather than merging `main` in.
- Squash-merge to keep `main` linear and easy to pin.
- PR description states what changed and how it was verified (harness run, eval numbers, curl output).

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
