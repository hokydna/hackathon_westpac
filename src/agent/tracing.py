"""LangSmith tracing decorators -- no-ops unless explicitly enabled.

FROZEN in the base commit (SESSION_KICKOFF.md §4, F12). Session B decorates its
functions with these and never edits this file.

**Tracing ships OFF.** Challenge_Brief.md § Rules requires the scored path to
"use only approved local datasets and services during official scoring", and
tracing the scored run would send hidden question text and AFR article content
to a third-party cloud. This module exists so that (a) enabling dev-time
observability is one env var rather than a code change, and (b) adding it later
would not have meant editing a frozen file mid-flight.

The hand-rolled loop is why this is needed at all: LangChain's create_agent()
would have traced for free, but ADR-0001 rejected it because Qwen's XML means
message.tool_calls is always null. With a hand-rolled loop, LANGSMITH_TRACING
alone produces nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .. import config

F = TypeVar("F", bound=Callable[..., Any])

if config.LANGSMITH_TRACING:  # pragma: no cover - dev-only path
    try:
        from langsmith import traceable as _traceable

        def traceable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
            return _traceable(*args, **kwargs)

        TRACING_ACTIVE = True
    except ImportError:
        # Enabled but the package is missing: degrade silently rather than
        # crash the agent. /health is a hard gate; observability is not.
        def traceable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
            def _decorator(fn: F) -> F:
                return fn

            return _decorator

        TRACING_ACTIVE = False
else:

    def traceable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """Passthrough. Accepts and ignores every @traceable kwarg.

        Usable bare (@traceable) or called (@traceable(run_type="llm")).
        """
        # Bare usage: @traceable with no parentheses.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]  # type: ignore[return-value]

        def _decorator(fn: F) -> F:
            return fn

        return _decorator

    TRACING_ACTIVE = False
