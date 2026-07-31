"""One Qwen planning turn. Hazard #2, isolated in one file.

`enable_thinking: false` is not in effect server-side — the server's
`--override-generation-config` does not govern the chat template — so it must
travel per request as `chat_template_kwargs`. Measured on this cluster with tools
attached and a tool call expected, median of 3:

    thinking ON  : 6.83s, 300 completion tokens (capped; 800 → ~17.6s)
    thinking OFF : 1.17s,  43 completion tokens

That is the difference between a 3-turn loop costing ~3.5s and costing ~50s, and
the 60s scored boundary sits in between. The flag is therefore **not a tunable**:
`plan()` overwrites any caller-supplied `chat_template_kwargs` rather than merging,
because a caller that switches thinking back on has broken the budget for the
whole request and should not be able to do so by accident.

`plan()` never raises. §7 of the harness plan: on brain timeout or 5xx the loop
stops and synthesizes from the partial trace, so failure is data, not an
exception.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .. import config

# Generous relative to the measured 43 tokens for a tool call. Exists so a turn
# that decides to write prose instead cannot spend the whole deadline.
MAX_PLAN_TOKENS = 512


@dataclass
class BrainReply:
    """One turn's outcome. `ok is False` means the loop should stop looping."""

    content: str
    ok: bool = True
    error: str = ""
    completion_tokens: int = 0
    elapsed_s: float = 0.0


Transport = Callable[[dict, float], dict]


def _http(payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{config.LITELLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {config.LITELLM_KEY}"} if config.LITELLM_KEY else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def plan(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    timeout: float | None = None,
    transport: Transport | None = None,
    **_ignored: Any,
) -> BrainReply:
    """Ask the brain for its next move.

    Returns the raw assistant content — `parser.parse()` turns it into calls.
    We do not look at `message.tool_calls`: it is always `None` under
    `--tool-call-parser hermes` with an XML-emitting model.

    `**_ignored` swallows caller kwargs (including any attempt to pass
    `chat_template_kwargs`) so the thinking-off guarantee cannot be overridden.
    """
    send = transport or _http
    timeout = config.BRAIN_TIMEOUT_S if timeout is None else timeout

    payload: dict[str, Any] = {
        "model": config.BRAIN_MODEL,
        "messages": messages,
        "max_tokens": MAX_PLAN_TOKENS,
        "temperature": 0,
        # Set last and unconditionally. Never merged from caller input.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Omit the key entirely when there is nothing to send: session A's registry
    # starts empty, and some servers reject `tools: []`.
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    started = time.monotonic()
    try:
        data = send(payload, timeout)
    except Exception as exc:  # noqa: BLE001 - deliberate: no failure may escape
        return BrainReply(
            content="",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )
    elapsed = time.monotonic() - started

    try:
        content = data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        return BrainReply(
            content="",
            ok=False,
            error=f"unexpected response shape: {type(exc).__name__}",
            elapsed_s=elapsed,
        )

    return BrainReply(
        content=content or "",
        ok=True,
        completion_tokens=(data.get("usage") or {}).get("completion_tokens", 0),
        elapsed_s=elapsed,
    )
