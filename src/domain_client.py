"""The single DOMAIN_FT_MODEL caller.

FROZEN in the base commit (SESSION_KICKOFF.md §4, §10 / F17).

Nemotron is reached from two places, in two different sessions:

  * src/agent/synth.py     (B) -- Role 1, final synthesis, after the loop
  * src/tools/registry.py  (A) -- Role 2, the domain_sentiment tool, mid-loop

By the Phase 0 rule, a file two sessions both need is complete before either
forks. Without this module, A's tool would have to import B's synth.py, which
may not exist yet -- exactly the cross-stream dependency Phase 0 prevents.

DOMAIN_PREDICT_MODE governs both roles at once: `mock` returns a deterministic
echo so A/B/C can build and test with no adapter served, `llm` calls the real
endpoint. Challenge_Brief.md requires `llm` before official evaluation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class DomainUnavailable(RuntimeError):
    """The domain model could not be reached or returned no usable content.

    Callers must degrade, never propagate. §7 of the harness plan: every
    failure path still produces a valid answer string. For Role 1 that means
    synth.py falls back to a deterministic template built from the trace; for
    Role 2 it means the tool returns a "sentiment unavailable" string so the
    brain can carry on and the rate-lookup component still scores.
    """


def _post(payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{config.DOMAIN_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {config.LITELLM_KEY}"}
                if config.LITELLM_KEY
                else {}
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 256,
    timeout: float | None = None,
    mode: str | None = None,
) -> str:
    """One Nemotron completion. Returns the content string.

    temperature=0 always: the base-vs-fine-tuned comparison is only a
    controlled ablation if decoding is identical across both arms, and a
    non-deterministic judge makes "74% -> 71%" uninterpretable.

    Raises DomainUnavailable on any failure. Callers degrade.
    """
    mode = mode or config.DOMAIN_PREDICT_MODE
    timeout = config.SYNTH_TIMEOUT_S if timeout is None else timeout

    if mode == "mock":
        # Deterministic, offline, and obviously synthetic so it can never be
        # mistaken for a real answer in a trace or a scored run.
        return f"[mock:{config.DOMAIN_FT_MODEL}] {user.strip()[:200]}"

    payload = {
        "model": config.DOMAIN_FT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    try:
        data = _post(payload, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise DomainUnavailable(f"{config.DOMAIN_FT_MODEL} unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DomainUnavailable(f"{config.DOMAIN_FT_MODEL} returned non-JSON") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        # LiteLLM returns {"error": {...}} on a bad alias -- e.g. the known
        # domain-ft -> nemotron-8b-finance mismatch before the adapter is
        # served under --served-model-name nemotron-8b-finance.
        raise DomainUnavailable(f"unexpected response shape: {str(data)[:200]}") from exc

    if not content or not content.strip():
        raise DomainUnavailable("empty content")

    return content.strip()


def is_live(mode: str | None = None) -> bool:
    """Cheap reachability check for eval and the pre-submission gate.

    Never call this from the /query path -- /health must not depend on any
    model being reachable (it is a hard gate on the whole 40%).
    """
    mode = mode or config.DOMAIN_PREDICT_MODE
    if mode == "mock":
        return False
    try:
        complete("Reply with OK.", "Reply with OK.", max_tokens=5, timeout=5.0, mode="llm")
        return True
    except DomainUnavailable:
        return False
