"""The four frozen contracts, as importable stubs.

FROZEN in the base commit (SESSION_KICKOFF.md §4). These exist so sessions A, B
and C can import each other's interfaces on minute one, before any of the real
implementations exist. Session B codes the loop against ToolCall/AgentResult and
the empty registry exports; session A fills the registry in later.

`tool_trace` entries match Participant_Package/answer_template.json exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation, parser -> guard -> registry.

    Args arrive from Qwen's XML as strings (year='2018'), so the guard coerces
    them via Pydantic at the boundary. Never trust arg types here.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEntry:
    """One executed call, as it appears in the response's tool_trace."""

    tool: str
    args: dict[str, Any]
    result: str


@dataclass
class AgentResult:
    """loop -> app. Shaped to validate.json.

    `answer` is the only field the hidden-question grader scores, and the only
    required one (minLength 1). `steps` and `tool_trace` are optional organizer
    diagnostics -- strongly recommended, never load-bearing.

    NOTE `steps` != config.MAX_TURNS (kickoff F9). One brain turn can emit
    several <tool_call> blocks, so steps counts executed calls, not turns.
    """

    answer: str
    steps: int = 0
    tool_trace: list[TraceEntry] = field(default_factory=list)

    def to_response(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "tool_trace": [
                {"tool": e.tool, "args": e.args, "result": e.result}
                for e in self.tool_trace
            ],
        }


# --------------------------------------------------------------------------
# Session A fills these in src/tools/registry.py. Session B imports them and
# never edits registry.py -- that ownership call is kickoff §4.
#
# Empty here so `from src.tools.registry import ALL_TOOLS` resolves before A
# has written a line.
# --------------------------------------------------------------------------

ALL_TOOLS: list[Any] = []
BRAIN_SCHEMAS: list[dict[str, Any]] = []
