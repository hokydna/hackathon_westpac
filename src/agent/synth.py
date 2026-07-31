"""Final answer synthesis — Nemotron's Role 1. NOT a tool.

Invoked unconditionally by `loop.py` after the Qwen loop exits: on every request,
including deadline breach, zero successful tool calls, and total brain failure.
It is never registered in `ALL_TOOLS` and Qwen never decides whether it runs.

`Challenge_Brief.md` § Required Model Roles requires Nemotron to receive the
accumulated verified results *after* the reasoning loop, and
`handout/03_scoring_and_examples.md` has a section titled "Bad: Nemotron used as
the planner and tool caller". Two 30% categories score this separation.

Three modes, in descending order of preference:

    llm       -> the served adapter (or base Nemotron) via domain_client
    mock      -> deterministic stub, so A/B/C build with no adapter served
    fallback  -> prompts.deterministic_fallback(), no model in the loop

The fallback is not decoration. §7's single most valuable invariant is **never
return a non-answer**: `validate.json` requires only `answer` with minLength 1,
and a Nemotron outage must degrade to partial credit rather than zero.

--------------------------------------------------------------------------
The evidence-shape contract, and where it is still soft
--------------------------------------------------------------------------

The adapter was trained on *structured* evidence. A real training record's user
message looks like::

    Question:
    How volatile was CBA.AX in 2017 on an annualised basis?

    Requested components:
    [ "volatility_pct_annualised", "basis" ]

    Verified evidence:
    { "ticker": "CBA.AX", "year": 2017,
      "volatility_pct_annualised": 14.464909945587081, ... }

    Limitations:
    []

But the harness contract (harness plan §3) says **every tool returns `str`**. So
there is a genuine gap between what the tools produce and what the adapter was
trained to read — this is exactly `FINETUNE_DATA_SOURCES.md` Q3 ("train on a
tool_results format that differs from what the live harness feeds Nemotron and
the adapter degrades").

This module therefore accepts structured evidence when the trace carries it and
degrades to `{tool_name: result_string}` when it does not. **The string form is a
shape the adapter never saw**, so it is a correctness risk, not a neutral
fallback. Resolving it properly needs a cross-session decision: either tool
results carry structured data through the trace, or session D re-serialises.
Tracked in the handoff note rather than papered over here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .. import config, domain_client, prompts
from .tracing import traceable

# Keys that describe the query rather than answer it. Excluded when we have to
# guess `requested_components`, so the model is not told the ticker it was given
# is a fact it must restate.
_CONTEXT_KEYS = frozenset(
    {"ticker", "tickers", "year", "years", "dataset", "date", "date_from", "date_to",
     "start", "end", "metric", "pattern", "exclude_tickers"}
)


@dataclass
class Synthesis:
    answer: str
    mode: str  # "llm" | "mock" | "fallback"
    error: str = ""


def _structured_evidence(trace: Iterable[Any]) -> tuple[dict[str, Any], bool]:
    """Merge structured tool payloads into one evidence dict.

    Returns (evidence, is_structured). `is_structured` is False when we had to
    fall back to raw strings, which callers log because it means the adapter is
    being fed an untrained shape.
    """
    evidence: dict[str, Any] = {}
    structured = False
    strings: dict[str, str] = {}

    for i, entry in enumerate(trace or []):
        tool = getattr(entry, "tool", None) or (
            entry.get("tool") if isinstance(entry, Mapping) else None
        ) or f"tool_{i}"
        data = getattr(entry, "data", None)
        if data is None and isinstance(entry, Mapping):
            data = entry.get("data")

        if isinstance(data, Mapping):
            evidence.update(data)
            structured = True
            continue

        result = getattr(entry, "result", None)
        if result is None and isinstance(entry, Mapping):
            result = entry.get("result")
        if result:
            # Distinct key per call so two calls to the same tool don't collide.
            key = tool if tool not in strings else f"{tool}_{i}"
            strings[key] = str(result)

    if strings and not structured:
        return strings, False
    evidence.update(strings)
    return evidence, structured


def _infer_components(evidence: Mapping[str, Any]) -> list[str]:
    """Best-effort `requested_components` when the caller supplies none.

    At inference the grading components are hidden, so this is an approximation:
    every evidence key that looks like a result rather than a query parameter.
    Training records carried the real list, so this is one more strand of the Q3
    format gap — documented, not hidden.
    """
    return [k for k in evidence if k not in _CONTEXT_KEYS] or list(evidence)


@traceable(run_type="llm", name="synth.write")
def write(
    question: str,
    trace: Sequence[Any],
    *,
    requested_components: Sequence[str] | None = None,
    limitations: Sequence[Mapping[str, Any]] = (),
    mode: str | None = None,
) -> Synthesis:
    """Produce the final answer. Never returns an empty string.

    Failure order is deliberate: try the model, then the deterministic template,
    then a bare statement of insufficiency. Each step still yields a valid
    `answer`, because the alternative is a zero on a question we may have
    computed perfectly.
    """
    mode = mode or config.DOMAIN_PREDICT_MODE
    evidence, structured = _structured_evidence(trace)
    components = list(requested_components or _infer_components(evidence))

    if not evidence:
        # No tool ever succeeded. Still answer -- state the limitation.
        return Synthesis(
            answer=(
                "I could not answer this question because no verified data was "
                "retrieved from the approved datasets."
            ),
            mode="fallback",
            error="empty trace",
        )

    messages = prompts.render_synthesis_messages(
        question=question,
        requested_components=components,
        verified_evidence=evidence,
        limitations=list(limitations),
    )

    try:
        answer = domain_client.complete(
            messages[0]["content"],
            messages[1]["content"],
            max_tokens=256,
            timeout=config.SYNTH_TIMEOUT_S,
            mode=mode,
        )
    except domain_client.DomainUnavailable as exc:
        answer, err = "", str(exc)
    else:
        err = "" if structured else "evidence passed as strings, not structured"

    if answer.strip():
        return Synthesis(answer=answer.strip(), mode=mode, error=err)

    # Model unreachable or empty. Build the answer from evidence alone.
    fallback = prompts.deterministic_fallback(
        requested_components=components,
        verified_evidence=evidence,
        limitations=list(limitations),
    ).strip()

    if fallback:
        return Synthesis(answer=fallback, mode="fallback", error=err or "empty completion")

    return Synthesis(
        answer=(
            "I retrieved data for this question but could not compose a final "
            "answer. The requested values could not be determined."
        ),
        mode="fallback",
        error=err or "empty fallback",
    )
