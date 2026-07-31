"""Parser tests — hazard #1 of the whole build.

`vllm-brain` runs `--tool-call-parser hermes` but Qwen3.6 emits XML, so
`message.tool_calls` is ALWAYS null and the call text leaks into
`message.content`. LangChain's `create_agent()` routes on
`AIMessage.tool_calls` and would therefore never dispatch a tool. We parse the
XML ourselves.

The payloads below are VERBATIM captures from `agent-brain` through LiteLLM on
this cluster, not invented. `REAL_SINGLE_CALL` was captured with
`chat_template_kwargs={"enable_thinking": false}`, `tools=[query_data]`,
`tool_choice="auto"`, and came back with `finish_reason: "stop"` and
`tool_calls: None`.

§7 of the harness plan promises the brain can always replan, so **no input may
raise.** Every hostile case must return a structured outcome instead.
"""

from __future__ import annotations

import pytest

from src.agent import parser

pytestmark = pytest.mark.unit


# Verbatim capture. Note: values sit on their own lines with surrounding
# whitespace, and the array argument arrives as a JSON-array *string*.
REAL_SINGLE_CALL = """<tool_call>
<function=query_data>
<parameter=dataset>
asx
</parameter>
<parameter=metric>
annual_return
</parameter>
<parameter=ticker>
BHP.AX
</parameter>
<parameter=year>
2018
</parameter>
<parameter=exclude_tickers>
["TAB.AX"]
</parameter>
</function>
</tool_call>"""

# The inline form the harness plan documents. Both shapes must parse.
REAL_INLINE_CALL = (
    "<tool_call><function=query_data><parameter=dataset>asx</parameter>"
    "<parameter=metric>describe</parameter></function></tool_call>"
)


class TestRealCaptures:
    def test_parses_the_real_multiline_capture(self):
        calls = parser.parse(REAL_SINGLE_CALL)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "query_data"
        assert call.args["dataset"] == "asx"
        assert call.args["metric"] == "annual_return"
        assert call.args["ticker"] == "BHP.AX"

    def test_strips_the_whitespace_around_values(self):
        """Values arrive on their own lines. An unstripped "asx\\n" fails an enum."""
        call = parser.parse(REAL_SINGLE_CALL)[0]
        for value in call.args.values():
            if isinstance(value, str):
                assert value == value.strip()
                assert "\n" not in value

    def test_parses_the_inline_form_too(self):
        calls = parser.parse(REAL_INLINE_CALL)
        assert len(calls) == 1
        assert calls[0].args == {"dataset": "asx", "metric": "describe"}

    def test_numbers_stay_strings_for_the_guard_to_coerce(self):
        """The parser does not guess types. `year` arrives as '2018'.

        Coercion belongs at the guard boundary, where Pydantic owns it — one
        concern, one file.
        """
        call = parser.parse(REAL_SINGLE_CALL)[0]
        assert call.args["year"] == "2018"

    def test_json_array_argument_is_decoded(self):
        """`exclude_tickers` is a first-class argument on every ASX metric and it
        arrives as the string '["TAB.AX"]'. Left as a string, the tool would
        exclude nothing and silently return wrong numbers."""
        call = parser.parse(REAL_SINGLE_CALL)[0]
        assert call.args["exclude_tickers"] == ["TAB.AX"]


class TestMultipleCalls:
    def test_two_calls_in_one_reply(self):
        """One brain TURN can emit several calls — which is why `steps` and
        MAX_TURNS are different quantities (kickoff F9)."""
        payload = REAL_INLINE_CALL + "\n" + REAL_INLINE_CALL.replace("describe", "count")
        calls = parser.parse(payload)
        assert len(calls) == 2
        assert [c.args["metric"] for c in calls] == ["describe", "count"]

    def test_prose_around_the_calls_is_ignored(self):
        payload = f"I'll look that up.\n{REAL_INLINE_CALL}\nThen I will compare."
        assert len(parser.parse(payload)) == 1


class TestHostileInput:
    """§7: every failure path must let the brain replan. Nothing raises."""

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "   ",
            "No tool needed, the answer is 41.",
            "<tool_call>",
            "<tool_call><function=query_data>",
            "<tool_call></tool_call>",
            "<tool_call><function=></function></tool_call>",
            "<tool_call><function=q><parameter=a></function></tool_call>",
            "<tool_call>{not xml at all}</tool_call>",
        ],
    )
    def test_never_raises_and_always_returns_a_list(self, payload):
        result = parser.parse(payload)
        assert isinstance(result, list)

    def test_no_calls_means_the_brain_is_done_reasoning(self):
        """An empty list is the loop's signal to stop and synthesize — it must be
        distinguishable from a parse failure only by being empty, never by an
        exception."""
        assert parser.parse("The answer is 41 of 175.") == []

    def test_unclosed_call_does_not_swallow_a_later_valid_one(self):
        payload = f"<tool_call><function=broken\n{REAL_INLINE_CALL}"
        calls = parser.parse(payload)
        assert any(c.name == "query_data" for c in calls)

    def test_none_content_is_tolerated(self):
        """LiteLLM can return content=None. The loop must not crash on it."""
        assert parser.parse(None) == []

    def test_unicode_in_arguments_survives(self):
        payload = (
            "<tool_call><function=query_data>"
            "<parameter=pattern>café</parameter>"
            "<parameter=dataset>afr</parameter>"
            "</function></tool_call>"
        )
        call = parser.parse(payload)[0]
        assert call.args["pattern"] == "café"

    def test_malformed_json_array_falls_back_to_the_raw_string(self):
        """Better a string the guard can reject than an exception here."""
        payload = (
            "<tool_call><function=query_data>"
            "<parameter=exclude_tickers>[\"TAH.AX\",]</parameter>"
            "<parameter=dataset>asx</parameter>"
            "</function></tool_call>"
        )
        call = parser.parse(payload)[0]
        assert isinstance(call.args["exclude_tickers"], (str, list))

    def test_a_parameter_with_no_value_becomes_empty_not_missing(self):
        payload = (
            "<tool_call><function=query_data>"
            "<parameter=ticker></parameter>"
            "<parameter=dataset>asx</parameter>"
            "</function></tool_call>"
        )
        call = parser.parse(payload)[0]
        assert call.args["ticker"] == ""
        assert call.args["dataset"] == "asx"


class TestHermesJsonForm:
    """If the server-side hermes parser ever starts working, or the model emits
    the JSON form, we must not regress. Both shapes are seen in the wild."""

    def test_json_tool_call_body_is_parsed(self):
        payload = (
            '<tool_call>{"name": "query_data", '
            '"arguments": {"dataset": "rba", "metric": "count_changes"}}</tool_call>'
        )
        calls = parser.parse(payload)
        assert len(calls) == 1
        assert calls[0].name == "query_data"
        assert calls[0].args == {"dataset": "rba", "metric": "count_changes"}
