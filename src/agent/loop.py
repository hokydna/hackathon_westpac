"""The orchestrator: turn cap, deadline, trace assembly, unconditional synthesis.

Data flow (harness plan §5)::

    msgs = [system(tool docs + answer rules), user(question)]
    for turn in 1..MAX_TURNS:
        if deadline breached: break
        reply = brain.plan(msgs)          # thinking off
        calls = parser.parse(reply.content)
        if not calls: break               # brain is done reasoning
        for call in guard.validate(calls):
            result = await tool.ainvoke(call.args)
            msgs.append(budget.tool_result_message(...))
            trace.append({tool, args, result})
    return synth.write(question, trace)   # ALWAYS

**Synthesis is unconditional.** It runs on every exit path — turn cap reached,
deadline breached, brain unreachable, every tool denied. It is never in
`ALL_TOOLS` and Qwen never decides whether it happens: `Challenge_Brief.md`
§ Required Model Roles requires Nemotron to receive the accumulated results
*after* the loop, and `handout/03` titles the inverse "Bad: Nemotron used as the
planner and tool caller".

**Concurrency is safe by construction, not by locking.** `msgs`, `trace` and the
`Budget` are per-request locals; corpora are read-only after startup. The harness
sends up to three concurrent `/query` requests and state must not bleed between
them, so there is a test for it rather than only a claim.

`answer()` is async because FastAPI awaits it and because tools expose `ainvoke`.
The blocking HTTP calls to the brain and to Nemotron are pushed to threads —
otherwise one in-flight request would stall the event loop and serialise the
three concurrent questions the harness sends.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from .. import config
from ..contracts import AgentResult, TraceEntry
from . import brain, budget, guard, parser, synth
from .tracing import traceable

SYSTEM_PROMPT = (
    "You are a financial research planner with access to deterministic data tools "
    "over three approved local datasets: RBA cash-rate decisions, ASX daily prices, "
    "and the AFR news corpus.\n\n"
    "Call a tool to obtain every value the question asks for. Use the structured "
    "data tools for any count, date, ranking or calculation — never estimate a "
    "number yourself and never answer a numeric question from memory. When you have "
    "all the values you need, reply with a short plain-text confirmation and no "
    "further tool calls.\n\n"
    "Prefer as few calls as possible; three or fewer is normally enough."
)


async def _invoke(tool: Any, args: Mapping[str, Any]) -> str:
    """Execute one tool. Prefers the async path, threads the sync one.

    A tool must never raise (harness §3: return a valid "no results" string), but
    this is a boundary to session A's code, so a raise here is contained and
    turned into a trace entry rather than a failed request.
    """
    if hasattr(tool, "ainvoke"):
        return str(await tool.ainvoke(dict(args)))
    if hasattr(tool, "invoke"):
        return str(await asyncio.to_thread(tool.invoke, dict(args)))
    return str(await asyncio.to_thread(tool, **dict(args)))


@traceable(run_type="chain", name="agent.answer")
async def answer(
    question: str,
    registry: Mapping[str, Any] | None = None,
    brain_schemas: list[dict] | None = None,
    *,
    clock=None,
) -> AgentResult:
    """One `/query`. Always returns a valid AgentResult with a non-empty answer."""
    registry = registry or {}
    b = budget.Budget(clock=clock) if clock else budget.Budget()

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[TraceEntry] = []
    seen_calls: dict[tuple[str, str], str] = {}

    while b.may_take_turn():
        b.record_turn()

        reply = await asyncio.to_thread(
            brain.plan, budget.trim(msgs), tools=brain_schemas or None
        )
        if not reply.ok:
            # Brain timeout or 5xx: stop looping, synthesize from what we have.
            break

        calls = parser.parse(reply.content)
        if not calls:
            break  # brain has finished reasoning

        accepted, denied = guard.validate(calls, registry)

        # Denials go back to the brain so it can replan on the next turn.
        for d in denied:
            msgs.append(budget.tool_result_message(d.call.name, d.as_tool_result()))
            trace.append(
                TraceEntry(tool=d.call.name, args=d.call.args, result=d.as_tool_result())
            )

        for call in accepted:
            # Deterministic tools over a read-only corpus: the same (name, args)
            # cannot produce a different answer, so re-executing wastes a turn.
            # Observed live: the brain re-emitted an identical query_data call on
            # all three turns and spent the whole cap on one question. Serving
            # the repeat from the trace keeps the answer identical while leaving
            # turns for the follow-up calls that hard cross-dataset questions
            # actually need.
            key = (call.name, repr(sorted(call.args.items())))
            if key in seen_calls:
                cached = seen_calls[key]
                msgs.append(budget.tool_result_message(call.name, cached))
                b.record_calls(1)
                continue

            try:
                result = await _invoke(registry[call.name], call.args)
            except Exception as exc:  # noqa: BLE001 - a tool bug must not 500 the request
                result = f"ERROR: tool '{call.name}' failed: {type(exc).__name__}: {exc}"
                entry = TraceEntry(tool=call.name, args=call.args, result=result)
            else:
                entry = TraceEntry(tool=call.name, args=call.args, result=result)
                # Structured payload for synthesis when the tool offers one —
                # the adapter was trained on typed evidence, not strings.
                data = getattr(registry[call.name], "last_data", None)
                if isinstance(data, Mapping):
                    entry.data = dict(data)  # type: ignore[attr-defined]

            seen_calls[key] = result
            msgs.append(budget.tool_result_message(call.name, result))
            trace.append(entry)
            b.record_calls(1)

        b.record_calls(len(denied))

        if b.breached():
            break

    limitations = []
    if b.breached():
        limitations.append(
            {"message": "The time budget was reached before all data could be gathered."}
        )

    result = await asyncio.to_thread(
        synth.write, question, trace, limitations=limitations
    )

    return AgentResult(answer=result.answer, steps=b.steps, tool_trace=trace)
