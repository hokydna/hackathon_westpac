"""XML `<tool_call>` → `[ToolCall]`. Hazard #1, isolated in one file.

`vllm-brain` runs `--tool-call-parser hermes`, but Qwen3.6 emits XML. The
consequence, verified live on this cluster: `message.tool_calls` is **always**
`None`, `finish_reason` is `"stop"` rather than `"tool_calls"`, and the call text
arrives inside `message.content`. LangChain's `create_agent()` routes on
`AIMessage.tool_calls`, so it would never dispatch a tool. Hence this module.

Verbatim capture from `agent-brain` (2026-07-31)::

    <tool_call>
    <function=query_data>
    <parameter=dataset>
    asx
    </parameter>
    <parameter=exclude_tickers>
    ["TAB.AX"]
    </parameter>
    </function>
    </tool_call>

Two properties of that shape drive the implementation:

* values sit on their own lines, so everything is stripped — an unstripped
  ``"asx\\n"`` fails an enum check downstream;
* array arguments arrive as JSON-array *strings*, and ``exclude_tickers`` is a
  first-class argument on every ASX metric. Left as a string the tool would
  exclude nothing and return a silently wrong number.

Scalars are deliberately left as strings (``year`` → ``"2018"``). Type coercion
belongs to ``guard.py``, where Pydantic owns it — one concern, one file.

**Nothing here raises.** §7 of the harness plan promises the brain can always
replan, so malformed input yields fewer calls, never an exception.
"""

from __future__ import annotations

import json
import re

from ..contracts import ToolCall
from .tracing import traceable

# Non-greedy so consecutive calls don't merge. DOTALL because the real payload is
# newline-separated.
_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Tolerates an unclosed </function> — the closing tag carries no information and
# the model occasionally drops it.
_FUNCTION_RE = re.compile(r"<function=([^>\s]*)>(.*?)(?:</function>|\Z)", re.DOTALL)

_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|\Z)", re.DOTALL
)


def _decode(raw: str) -> object:
    """Strip, then decode JSON containers only.

    Scalars stay strings for the guard to coerce. Containers must be decoded
    here because no downstream layer can recover a list from ``'["TAH.AX"]'``
    without re-parsing, and a silently-unexcluded ticker is a wrong answer
    rather than an error.
    """
    value = raw.strip()
    if value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Malformed container: hand the guard a string it can reject
            # cleanly. Better than raising in the parser.
            return value
    return value


@traceable(run_type="parser", name="parser.parse")
def parse(content: str | None) -> list[ToolCall]:
    """Extract every tool call from one brain reply.

    An empty list means the brain emitted no calls, which the loop reads as
    "done reasoning" and takes as its cue to synthesize. That is the same signal
    as a total parse failure, deliberately: either way the correct next move is
    to answer from whatever trace exists rather than to error.
    """
    if not content:
        return []

    calls: list[ToolCall] = []

    for block in _CALL_RE.findall(content):
        calls.extend(_parse_block(block))

    # An unclosed <tool_call> is invisible to _CALL_RE. Sweep the tail so a
    # broken call cannot swallow a valid one that follows it.
    if not calls and "<function=" in content:
        calls.extend(_parse_block(content))

    return calls


def _parse_block(block: str) -> list[ToolCall]:
    out: list[ToolCall] = []

    # Hermes JSON form. Kept because it is what the server would emit if the
    # tool-call parser ever started working, and we must not regress then.
    stripped = block.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                return [ToolCall(name=str(obj["name"]).strip(), args=args)]
        return []

    for name, body in _FUNCTION_RE.findall(block):
        name = name.strip()
        if not name:
            continue  # <function=> carries no dispatchable target
        args = {key.strip(): _decode(val) for key, val in _PARAM_RE.findall(body)}
        out.append(ToolCall(name=name, args=args))

    return out
