"""Guard tests — the security boundary between the brain and our tools.

The brain is an LLM: its output is untrusted input. `guard` is the only place a
tool name becomes a dispatch, so it is an allowlist, not a lookup.

Two behaviours matter more than they look:

* **A denial is structured data, not an exception.** §7 promises the brain can
  replan, which requires the denial to travel back as a tool result.
* **Near-miss names are DENIED, not normalised.** Silently mapping `QUERY_DATA`
  or `query_data ` onto the real tool means the model's mistake is invisible in
  the trace, and failure attribution (which workstream to fix) depends on that
  trace being honest.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from src.agent import guard
from src.contracts import ToolCall

pytestmark = pytest.mark.unit


class AsxArgs(BaseModel):
    dataset: str
    metric: str
    year: int | None = None
    exclude_tickers: list[str] = Field(default_factory=list)


class FakeTool:
    """Stands in for a LangChain @tool object (which exposes args_schema)."""

    def __init__(self, name: str, schema: type[BaseModel] | None = AsxArgs):
        self.name = name
        self.args_schema = schema


@pytest.fixture
def registry():
    return {"query_data": FakeTool("query_data"), "coverage": FakeTool("coverage", None)}


class TestAllowlist:
    def test_known_tool_is_accepted(self, registry):
        accepted, denied = guard.validate(
            [ToolCall("query_data", {"dataset": "asx", "metric": "describe"})], registry
        )
        assert len(accepted) == 1 and not denied

    def test_unknown_tool_is_denied_and_nothing_executes(self, registry):
        accepted, denied = guard.validate([ToolCall("rm_rf", {})], registry)
        assert accepted == []
        assert len(denied) == 1
        # The rejected name reaches the brain via the rendered result, and the
        # reason lists what it could have called instead so it can replan.
        assert "rm_rf" in denied[0].as_tool_result()
        assert "query_data" in denied[0].reason

    @pytest.mark.parametrize("name", ["QUERY_DATA", "query_data ", " query_data", "Query_Data"])
    def test_near_miss_names_are_denied_not_normalised(self, registry, name):
        """Normalising hides the model's error from the trace, and failure
        attribution reads the trace to decide which workstream to fix."""
        accepted, denied = guard.validate([ToolCall(name, {"dataset": "asx", "metric": "m"})], registry)
        assert accepted == []
        assert len(denied) == 1

    def test_empty_registry_denies_everything_without_crashing(self):
        """Session A's registry starts empty; B must still run."""
        accepted, denied = guard.validate([ToolCall("query_data", {})], {})
        assert accepted == []
        assert len(denied) == 1

    def test_denial_carries_the_original_call_for_the_trace(self, registry):
        _, denied = guard.validate([ToolCall("nope", {"a": 1})], registry)
        assert denied[0].call.name == "nope"
        assert denied[0].call.args == {"a": 1}


class TestCoercion:
    def test_string_ints_from_xml_are_coerced(self, registry):
        """The parser leaves scalars as strings on purpose: year='2018'."""
        accepted, denied = guard.validate(
            [ToolCall("query_data", {"dataset": "asx", "metric": "annual_return", "year": "2018"})],
            registry,
        )
        assert not denied
        assert accepted[0].args["year"] == 2018
        assert isinstance(accepted[0].args["year"], int)

    def test_already_decoded_list_survives_coercion(self, registry):
        accepted, _ = guard.validate(
            [
                ToolCall(
                    "query_data",
                    {"dataset": "asx", "metric": "m", "exclude_tickers": ["TAH.AX"]},
                )
            ],
            registry,
        )
        assert accepted[0].args["exclude_tickers"] == ["TAH.AX"]

    def test_uncoercible_argument_is_denied_not_raised(self, registry):
        accepted, denied = guard.validate(
            [ToolCall("query_data", {"dataset": "asx", "metric": "m", "year": "not-a-year"})],
            registry,
        )
        assert accepted == []
        assert len(denied) == 1
        assert "year" in denied[0].reason

    def test_missing_required_argument_is_denied(self, registry):
        accepted, denied = guard.validate([ToolCall("query_data", {"dataset": "asx"})], registry)
        assert accepted == []
        assert "metric" in denied[0].reason

    def test_tool_without_a_schema_passes_args_through(self, registry):
        accepted, denied = guard.validate([ToolCall("coverage", {"anything": "goes"})], registry)
        assert not denied
        assert accepted[0].args == {"anything": "goes"}

    def test_defaults_are_not_injected_into_the_call(self, registry):
        """Only what the brain asked for is dispatched. Injecting
        exclude_tickers=[] would silently claim the brain requested it, which
        misleads the trace and the tool."""
        accepted, _ = guard.validate(
            [ToolCall("query_data", {"dataset": "asx", "metric": "describe"})], registry
        )
        assert "exclude_tickers" not in accepted[0].args


class TestNeverRaises:
    @pytest.mark.parametrize(
        "calls",
        [
            [],
            [ToolCall("", {})],
            [ToolCall("query_data", None)],  # type: ignore[arg-type]
        ],
    )
    def test_degenerate_input_returns_structured_results(self, registry, calls):
        accepted, denied = guard.validate(calls, registry)
        assert isinstance(accepted, list) and isinstance(denied, list)

    def test_a_bad_call_does_not_block_a_good_one_beside_it(self, registry):
        accepted, denied = guard.validate(
            [
                ToolCall("bogus", {}),
                ToolCall("query_data", {"dataset": "asx", "metric": "describe"}),
            ],
            registry,
        )
        assert len(accepted) == 1
        assert len(denied) == 1

    def test_denial_renders_to_a_tool_result_string_for_the_brain(self, registry):
        _, denied = guard.validate([ToolCall("bogus", {})], registry)
        rendered = denied[0].as_tool_result()
        assert isinstance(rendered, str) and rendered
        # Must name the tool so the brain can correct itself.
        assert "bogus" in rendered
