"""Every environment variable and budget constant, in one place.

FROZEN in the base commit (SESSION_KICKOFF.md §4). Every session imports from
here; no session edits it. If you need a new constant, it goes in the base
commit before forking, not into your branch.

Nothing here hard-codes an endpoint or a credential -- Setup_Instructions.md
requires all of those to come from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Repo / data locations
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# `data set/` is untracked (785 MB), so a git worktree does NOT contain it.
# DATASET_DIR is therefore absolute and defaults to the main checkout, letting
# every worktree read one shared copy. Unit tests override it to point at
# tests/fixtures/dataset/ instead -- see tests/conftest.py.
def _default_dataset_dir() -> Path:
    """Resolve `data set/` even from inside a linked git worktree.

    `data set/` is untracked, so a worktree does not contain it. REPO_ROOT is
    derived from __file__, which inside `.worktrees/agent/` points at the
    worktree — where the corpus does not exist. Every session works in a
    worktree, so the naive default broke the corpus for all of them.

    In a linked worktree `.git` is a FILE containing
    `gitdir: /path/to/main/.git/worktrees/<name>`, so the main checkout is the
    parent of the `.git` directory in that path. Walk to it and look there.
    """
    local = REPO_ROOT / "data set"
    if local.is_dir():
        return local

    gitfile = REPO_ROOT / ".git"
    if gitfile.is_file():
        for line in gitfile.read_text().splitlines():
            if line.startswith("gitdir:"):
                gitdir = Path(line.split(":", 1)[1].strip())
                for ancestor in gitdir.parents:
                    if ancestor.name == ".git":
                        candidate = ancestor.parent / "data set"
                        if candidate.is_dir():
                            return candidate
    return local


DATASET_DIR = Path(os.getenv("DATASET_DIR") or _default_dataset_dir())

RBA_PATH = DATASET_DIR / "RBA Rates" / "RBA-rates.jsonl"
ASX_DIR = DATASET_DIR / "ASX"
AFR_DIR = DATASET_DIR / "AFR"

# --------------------------------------------------------------------------
# Model endpoints and aliases
# --------------------------------------------------------------------------

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_KEY", "")

BRAIN_MODEL = os.getenv("BRAIN_MODEL", "agent-brain")
DOMAIN_FT_MODEL = os.getenv("DOMAIN_FT_MODEL", "domain-ft")

# `mock` is the organizer bootstrap default. Challenge_Brief.md requires `llm`
# before official evaluation -- shipping `mock` forfeits the whole 30%
# fine-tuned-model-quality category and drags architecture credit with it.
DOMAIN_PREDICT_MODE = os.getenv("DOMAIN_PREDICT_MODE", "mock")

# Defaults to LiteLLM, so nothing changes unless it is set. Exists because
# FINETUNE_PLAN §9's fallback tree requires pointing DOMAIN_FT_MODEL directly at
# a vLLM endpoint (e.g. http://10.0.1.11:8001/v1) if the LiteLLM alias cannot be
# repaired -- and the brain must keep using LiteLLM regardless.
DOMAIN_BASE_URL = os.getenv("DOMAIN_BASE_URL") or LITELLM_BASE_URL

# --------------------------------------------------------------------------
# Latency budget  (kickoff F10)
#
# Derived, not hand-picked. 45s of looping plus a 15s synthesis timeout is 60s
# before FastAPI even serialises, which lands on the wrong side of the scored
# penalty line. LOOP_DEADLINE_S bounds when the LOOP must stop, leaving
# synthesis its full budget inside the 60s envelope.
#
# tests/test_budget_invariants.py asserts the sum, so tuning one constant
# cannot silently break the total.
# --------------------------------------------------------------------------

PENALTY_THRESHOLD_S = 60.0  # >60s costs 20% of that question's points
SYNTH_TIMEOUT_S = 15.0
SAFETY_MARGIN_S = 5.0
LOOP_DEADLINE_S = PENALTY_THRESHOLD_S - SYNTH_TIMEOUT_S - SAFETY_MARGIN_S  # 40.0

BRAIN_TIMEOUT_S = 5.0  # measured: ~1.0s per turn with thinking off

# Runaway backstop, NOT the primary governor -- LOOP_DEADLINE_S is (kickoff F9).
# Note turns != tool calls: one brain turn can emit several <tool_call> blocks,
# so 3 turns already permits 6+ calls. `steps` in the response and MAX_TURNS
# are different quantities.
MAX_TURNS = 3

# --------------------------------------------------------------------------
# Context budget  (kickoff F2)
#
# 1200, not 2000. FINETUNE_PLAN.md §4.4 caps training-time tool_results at
# 1200 against max_seq_len=512, and training and inference MUST clamp
# identically or the adapter is served a context shape it never saw.
# Imported by src/agent/budget.py (B) and training/prepare_data.py (D).
# --------------------------------------------------------------------------

TOOL_RESULT_CHAR_CAP = 1200
AFR_TEXT_CHAR_CAP = 1200
MAX_CONTEXT_TOKENS = 3000  # both models are max_model_len 4096

# Role 2 (kickoff §10): domain_sentiment returns a CLASSIFICATION, not an
# answer. Clamping is what keeps it from becoming a general-purpose answerer.
SENTIMENT_CHAR_CAP = 200

# --------------------------------------------------------------------------
# Observability  (kickoff F12)
#
# Tracing ships OFF: Challenge_Brief.md § Rules requires the scored path to use
# only approved local services, and hidden question text must not go to a third
# party. These live here anyway because config.py is frozen -- adding them
# later would violate the base-commit rule.
# --------------------------------------------------------------------------

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hackathon-dev")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", BRAIN_MODEL)
