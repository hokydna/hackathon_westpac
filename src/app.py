"""FastAPI surface: `GET /health` and `POST /query`. Deliberately thin.

**`/health` is a hard gate on 40% of the score.** `Challenge_Brief.md`
§ Response-Time Rules: if it does not return 200 during the pre-evaluation check,
the team is skipped and receives no hidden-question points at all. So it touches
nothing — no model, no corpus, no disk, no network. It answers while the process
lives, and that is its entire contract.

The way that is kept true is **startup ordering**: corpora load at import, before
uvicorn binds the port. The AFR index costs ~25s to build, so the port simply does
not open until the agent can actually answer. That removes the whole class of
"healthy but not ready" failure without `/health` needing to know anything about
readiness.

`POST /query` never propagates an exception. `validate.json` requires only
`answer` with `minLength: 1`, and a crash scores zero on that question where a
degraded answer scores partial credit — so every failure path is caught here and
turned into a valid response that states its limitation.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import loop
from .tools.registry import BRAIN_SCHEMAS, REGISTRY

log = logging.getLogger("agent")

app = FastAPI(
    title="Westpac Market Signal Agent",
    description="Qwen plans and calls tools; fine-tuned Nemotron synthesises the answer.",
    version="1.0.0",
)


def warm_corpora() -> None:
    """Load every corpus at import, BEFORE uvicorn binds the port.

    Harness §5 calls this load-bearing. Two consequences, both deliberate:

    * the port opens only once the ~25s AFR index build has finished, so the
      harness can never reach a process that is listening but cannot answer;
    * `/health` therefore never has to check corpus state, which keeps it a pure
      liveness check — and it is a hard gate on the entire 40%.

    A failure here should crash the process loudly at startup rather than
    degrade every subsequent request silently.
    """
    from .tools import corpora

    corpora.rba_rows()
    corpora.asx_series()
    corpora.afr_index()


@app.get("/health")
async def health() -> dict:
    """Process liveness. Touches nothing.

    No model call, no corpus access, no disk, no network. Anything added here
    becomes a way to fail the gate that zeroes 40% of the score.
    """
    return {"status": "ok"}


@app.post("/query")
async def query(request: Request) -> JSONResponse:
    """Answer one question. Always 200, always schema-valid.

    Returns 200 even for a malformed body: the harness sends `{"question": ...}`,
    but a 422 scores zero for that case, whereas an answer stating the limitation
    can still earn partial credit.
    """
    question = ""
    try:
        body = await request.json()
        if isinstance(body, dict):
            # `question` is the documented field; the others are cheap tolerance
            # in case the harness or a teammate's client words it differently.
            for key in ("question", "prompt", "query", "input"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    question = value.strip()
                    break
    except Exception:  # noqa: BLE001 - a bad body must not 500
        question = ""

    if not question:
        return JSONResponse(
            {
                "answer": (
                    "No question was supplied in the request, so there is nothing to "
                    "answer. Send a JSON object with a non-empty 'question' field."
                ),
                "steps": 0,
                "tool_trace": [],
            }
        )

    try:
        result = await loop.answer(question, REGISTRY, BRAIN_SCHEMAS)
        payload = result.to_response()
    except Exception:  # noqa: BLE001 - §7: never return a non-answer
        log.exception("loop failed for question: %s", question[:120])
        payload = {
            "answer": (
                "An internal error prevented this question from being answered "
                "from the approved datasets."
            ),
            "steps": 0,
            "tool_trace": [],
        }

    # Last line of defence on the one field that is scored. An empty `answer`
    # fails validate.json's minLength and scores zero.
    if not str(payload.get("answer") or "").strip():
        payload["answer"] = (
            "No answer could be produced for this question from the approved datasets."
        )

    return JSONResponse(payload)


# Runs at import, before uvicorn binds. Unconditional and deliberately not
# behind a flag: a flag would let the port open before the corpora are ready,
# which is exactly the failure startup ordering exists to prevent.
#
# Tests are unaffected because conftest points DATASET_DIR at
# tests/fixtures/dataset/, where warming is instant.
warm_corpora()
