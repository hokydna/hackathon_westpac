"""Send real public questions through the agentic harness with LangSmith tracing on.

Dev tooling. **Not for the scored run** — `Challenge_Brief.md` § Rules requires
only approved local services during official scoring, and tracing ships hidden
question text to a third-party cloud. This is for debugging the loop on the 15
*public* calibration questions, whose text is already ours.

Usage::

    set -a; . ./.env; set +a
    export LANGSMITH_TRACING=true
    export DATASET_DIR="$PWD/data set"          # or the main checkout's path
    .venv/bin/python scripts/trace_smoke.py --limit 3

Every span the loop emits shows up as its own run, so a failure is attributable
to a layer rather than to "the agent":

    agent.answer   (chain)   one run per question, the root
      brain.plan   (llm)     per turn — latency and completion tokens
      parser.parse (parser)  hazard #1: when it returns [] you see the raw content
      guard.validate (chain) denials, with the reason
      tool.invoke  (tool)    the actual dispatch, args and result
      synth.write  (llm)     Role 1, the final answer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.agent import loop  # noqa: E402

QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"


class RbaTool:
    """A real deterministic RBA tool, so the traces carry genuine tool output.

    Session A owns the real tool layer; this is the minimum needed to exercise
    dispatch end to end without depending on their branch.
    """

    name = "query_data"
    args_schema = None

    def __init__(self) -> None:
        self.last_data: dict | None = None

    async def ainvoke(self, args: dict) -> str:
        rows = [
            json.loads(line)
            for line in open(config.RBA_PATH, encoding="utf-8-sig")
            if line.strip()
        ]
        changed = [r for r in rows if r["Change % points"] not in ("0.00", "0.0")]
        ups = sum(1 for r in changed if r["Change % points"].startswith("+"))
        self.last_data = {
            "total_records": len(rows),
            "changed": len(changed),
            "increases": ups,
            "decreases": len(changed) - ups,
        }
        return (
            f"{len(changed)} of {len(rows)} records changed the rate: "
            f"{ups} increases and {len(changed) - ups} decreases."
        )


BRAIN_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Query the approved local datasets deterministically. "
                "RBA metrics: count_changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "enum": ["rba", "asx", "afr"]},
                    "metric": {"type": "string", "enum": ["count_changes"]},
                },
                "required": ["dataset", "metric"],
            },
        },
    }
]


def commit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    args = ap.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not exported. Run: set -a; . ./.env; set +a")
        return 1
    if not config.LANGSMITH_TRACING:
        print("LANGSMITH_TRACING is falsy, so nothing will be traced.")
        print("Run: export LANGSMITH_TRACING=true")
        return 1
    os.environ.setdefault("LANGCHAIN_PROJECT", config.LANGSMITH_PROJECT)

    questions = [json.loads(line) for line in QUESTIONS.open() if line.strip()]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    questions = questions[: args.limit]

    sha = commit_sha()
    print(f"project     : {config.LANGSMITH_PROJECT}")
    print(f"commit      : {sha}")
    print(f"brain       : {config.BRAIN_MODEL} via {config.LITELLM_BASE_URL}")
    print(f"synth mode  : {config.DOMAIN_PREDICT_MODE}")
    print(f"questions   : {len(questions)}\n")

    for q in questions:
        res = await loop.answer(
            q["prompt"],
            {"query_data": RbaTool()},
            BRAIN_SCHEMAS,
            # Static run metadata makes traces comparable across commits — the
            # thing that turns a trace into evidence rather than an anecdote.
            langsmith_extra={
                "metadata": {
                    "question_id": q["id"],
                    "difficulty": q["difficulty"],
                    "commit_sha": sha,
                    "brain_model": config.BRAIN_MODEL,
                    "domain_ft_model": config.DOMAIN_FT_MODEL,
                    "domain_predict_mode": config.DOMAIN_PREDICT_MODE,
                },
                "tags": [f"difficulty:{q['difficulty']}", f"sha:{sha}"],
            },
        )
        print(f"{q['id']:8} steps={res.steps} tools={len(res.tool_trace)}")
        print(f"         answer: {res.answer[:110]}")

    # Flush before exit, or short runs lose their traces. Never call this on the
    # request path -- it blocks.
    try:
        from langsmith import Client

        Client().flush()
    except Exception:  # noqa: BLE001
        pass

    print(f"\nTraces: https://smith.langchain.com  (project: {config.LANGSMITH_PROJECT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
