"""Allowlist + argument coercion. The security boundary before dispatch.

The brain is an LLM, so its output is untrusted input. This is the only place a
tool *name* turns into a *dispatch*, which makes it an allowlist rather than a
lookup: an exact match against the registry, or nothing runs.

Two decisions worth stating, because both are load-bearing and neither is
obvious:

**Near-miss names are denied, never normalised.** `QUERY_DATA`, `query_data `
and `Query_Data` are all rejected. Quietly mapping them onto the real tool would
make the model's mistake invisible in `tool_trace` — and session C's failure
attribution reads that trace to decide whether a lost component is a
routing/execution bug (sessions A/B) or a synthesis bug (session D). An honest
trace is worth more than a salvaged call.

**A denial is data, not an exception.** §7 of the harness plan promises the brain
can replan, which only works if the denial travels back to it as a tool result.
`Denial.as_tool_result()` renders that string.

Coercion happens here and nowhere else: `parser.py` deliberately leaves scalars
as strings (`year` → `"2018"`) because the XML carries no types, and Pydantic
owns the conversion. One concern, one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import ToolCall
from .tracing import traceable


@dataclass
class Denial:
    """A rejected call, shaped to travel back to the brain."""

    call: ToolCall
    reason: str

    def as_tool_result(self) -> str:
        """What the brain sees. Names the tool so it can correct itself."""
        return f"ERROR: call to '{self.call.name}' was not executed. {self.reason}"


def _coerce(tool: Any, args: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate args against the tool's Pydantic schema, if it has one.

    Returns (coerced_args, "") on success or (None, reason) on failure.

    Only keys the caller actually supplied are returned. Injecting schema
    defaults would put arguments into `tool_trace` that the brain never
    requested — misleading for both the tool and anyone reading the trace.
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return dict(args), ""

    try:
        model = schema.model_validate(dict(args))
    except Exception as exc:  # pydantic.ValidationError, or anything it raises
        return None, _summarise(exc)

    dumped = model.model_dump()
    return {k: v for k, v in dumped.items() if k in args}, ""


def _summarise(exc: Exception) -> str:
    """Compact, single-line error naming the offending fields.

    Pydantic's default rendering is multi-line and includes a docs URL; that
    goes into the brain's 4096-token context, so it gets trimmed.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = [
                f"{'.'.join(str(p) for p in e.get('loc', ())) or '?'}: {e.get('msg', '')}"
                for e in errors()
            ]
            if parts:
                return "Invalid arguments — " + "; ".join(parts[:4])
        except Exception:  # noqa: BLE001 - never let error handling raise
            pass
    return f"Invalid arguments — {type(exc).__name__}"


@traceable(run_type="chain", name="guard.validate")
def validate(
    calls: list[ToolCall],
    registry: Mapping[str, Any],
) -> tuple[list[ToolCall], list[Denial]]:
    """Split proposed calls into (executable, denied).

    Every input yields one output in exactly one list, so the loop can append a
    trace entry for each and the brain always learns what happened. Nothing
    raises: a malformed call is a `Denial`, not a crash.
    """
    accepted: list[ToolCall] = []
    denied: list[Denial] = []

    for call in calls or []:
        name = getattr(call, "name", "") or ""
        args = getattr(call, "args", None)
        if not isinstance(args, Mapping):
            args = {}

        # Exact match only. No strip(), no casefold() -- see module docstring.
        tool = registry.get(name) if name else None
        if tool is None:
            allowed = ", ".join(sorted(registry)) or "none registered"
            denied.append(
                Denial(
                    call=ToolCall(name=name, args=dict(args)),
                    reason=f"No such tool. Available tools: {allowed}.",
                )
            )
            continue

        coerced, reason = _coerce(tool, args)
        if coerced is None:
            denied.append(Denial(call=ToolCall(name=name, args=dict(args)), reason=reason))
            continue

        accepted.append(ToolCall(name=name, args=coerced))

    return accepted, denied
