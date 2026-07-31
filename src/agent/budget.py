"""Turn cap, wall deadline, context clamp. Hazards #3 and #4, one file.

Three rules, each protecting a hard limit that sits just outside the happy path.
Measured end-to-end is ~6-10s, so none of this fires in the common case — it
exists entirely for the pathological one, which is exactly where the penalty
applies.

**1. Wall deadline — `LOOP_DEADLINE_S = 40`, not 45.** Derived, not chosen. It
bounds when the *loop* must stop, leaving synthesis its full 15s inside the 60s
envelope: 40 + 15 + 5 margin = 60. The original 45 + 15 summed to exactly 60
before FastAPI serialisation, i.e. on the wrong side of the -20% line.
`tests/test_base_commit.py` asserts the sum so tuning one constant cannot
silently break the total.

**2. Turn cap — `MAX_TURNS = 3`, as a backstop only.** Wall-clock is the real
governor. Note `turns != steps`: one brain turn can emit several `<tool_call>`
blocks, so a 3-turn cap already permits 6+ calls, which satisfies the handout's
"hard questions may need 3-5 tool calls".

**3. Context clamp — 1,200 chars per tool result.** Not a nicety: both models are
`max_model_len 4096`, and the fine-tuning run caps training-time `tool_results` at
the same 1,200 against `max_seq_len 512`. If inference clamped at a different
number the adapter would be served a context shape it never saw.

Truncation is always announced in-band. A silently cut result teaches the brain
the data ended there, which is worse than a shorter honest one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config

_TRUNC = " ... [result truncated]"


def clamp(result: Any, cap: int | None = None) -> str:
    """Bound one tool result and say so when it was cut."""
    cap = config.TOOL_RESULT_CHAR_CAP if cap is None else cap
    if result is None:
        return ""
    text = result if isinstance(result, str) else str(result)
    if len(text) <= cap:
        return text
    return text[: max(0, cap - len(_TRUNC))].rstrip() + _TRUNC


def tool_result_message(tool: str, result: Any) -> dict[str, Any]:
    """The single factory for feeding a tool result back to the brain.

    Uses `role: "tool"`. That was an open question — the brain's XML arrives in
    `message.content`, so there is no `tool_calls` entry for a `tool_call_id` to
    reference, and an OpenAI-compatible `role: "tool"` message conventionally
    needs one. Probed live against `agent-brain` through LiteLLM with no
    `tool_call_id` and no preceding `tool_calls`: HTTP 200, coherent reply,
    `finish_reason: "stop"`. Accepted. No `<tool_response>` fallback needed.

    It stays a factory anyway so that if a future server tightens validation,
    the shape changes in exactly one place.
    """
    return {"role": "tool", "name": tool, "content": clamp(result)}


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap upper-ish estimate: ~4 chars per token.

    Deliberately not a real tokenizer. The brain's tokenizer is on the far side
    of an HTTP call, and spending a round trip per turn to measure the context
    would cost more than the headroom it buys. We keep a 1,096-token cushion
    under 4096 instead.
    """
    total = 0
    for m in messages:
        total += len(str(m.get("content") or "")) // 4 + 4  # +4 for role framing
    return total


def trim(
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Drop oldest tool results until the conversation fits.

    The system prompt and the question are never dropped — without the question
    the request is unanswerable, and §7 promises we always return an answer. If
    even they exceed the ceiling we return them anyway: a too-long prompt is the
    server's problem to reject, whereas an empty message list is ours.
    """
    max_tokens = config.MAX_CONTEXT_TOKENS if max_tokens is None else max_tokens

    if estimate_tokens(messages) <= max_tokens:
        return messages

    protected: list[dict[str, Any]] = []
    droppable: list[dict[str, Any]] = []
    for m in messages:
        # First system message and the user question are structural.
        if m.get("role") in ("system", "user"):
            protected.append(m)
        else:
            droppable.append(m)

    # Keep the newest droppable messages: the most recent tool result is the one
    # the brain is actively reasoning about.
    kept: list[dict[str, Any]] = []
    for m in reversed(droppable):
        candidate = [m, *kept]
        if estimate_tokens(protected + candidate) > max_tokens:
            break
        kept = candidate

    # Preserve original ordering.
    order = {id(m): i for i, m in enumerate(messages)}
    return sorted(protected + kept, key=lambda m: order[id(m)])


@dataclass
class Budget:
    """Per-request turn and deadline tracking.

    `clock` is injectable so deadline behaviour is tested with a fake clock
    rather than `sleep`, keeping the suite at milliseconds.
    """

    clock: Callable[[], float] = time.monotonic
    deadline_s: float = config.LOOP_DEADLINE_S
    max_turns: int = config.MAX_TURNS
    turns: int = 0
    steps: int = 0
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def elapsed_s(self) -> float:
        return self.clock() - self.started

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_s - self.elapsed_s())

    def breached(self) -> bool:
        return self.elapsed_s() >= self.deadline_s

    def may_take_turn(self) -> bool:
        """Wall-clock first: it is the real governor, the cap is the backstop."""
        return not self.breached() and self.turns < self.max_turns

    def record_turn(self) -> None:
        self.turns += 1

    def record_calls(self, n: int) -> None:
        """Executed tool calls. This is what `steps` reports, not turns."""
        self.steps += n
