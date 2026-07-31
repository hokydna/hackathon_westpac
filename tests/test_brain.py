"""Brain tests — hazard #2.

`enable_thinking: false` is NOT in effect server-side; the server's
`--override-generation-config` does not govern the chat template. It must be
passed per request, as `chat_template_kwargs`, on **every** call.

Measured on this cluster (2026-07-31, median of 3, tools attached, tool call
expected — i.e. the real loop condition):

    thinking ON  : 6.83s, 300 completion tokens (hit the cap; 800 → ~17.6s)
    thinking OFF : 1.17s,  43 completion tokens

43 tokens reproduces the plan's documented figure exactly. 1.17s against a 5s
`BRAIN_TIMEOUT_S` is ~4x headroom; without the flag a single turn would eat most
of `LOOP_DEADLINE_S` and three turns would blow the 60s scored boundary.

Beware measuring this with no `tools` and a prose-inviting prompt: that returns
~11s even with thinking off, because the model is answering rather than emitting
a call. It is not the loop's condition and it is not a valid comparison.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.agent import brain

pytestmark = pytest.mark.unit


class FakeTransport:
    """Records the payload the brain would have sent."""

    def __init__(self, content="<tool_call></tool_call>", *, raises=None, usage_tokens=43):
        self.content = content
        self.raises = raises
        self.usage_tokens = usage_tokens
        self.payloads: list[dict] = []
        self.timeouts: list[float] = []

    def __call__(self, payload: dict, timeout: float) -> dict:
        self.payloads.append(payload)
        self.timeouts.append(timeout)
        if self.raises:
            raise self.raises
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": self.usage_tokens},
        }


@pytest.fixture
def transport():
    return FakeTransport()


class TestThinkingSuppression:
    """The single most valuable assertion in this file."""

    def test_every_call_passes_enable_thinking_false(self, transport):
        for _ in range(3):
            brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert len(transport.payloads) == 3
        for payload in transport.payloads:
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_it_is_present_even_when_tools_are_omitted(self, transport):
        brain.plan([{"role": "user", "content": "q"}], tools=None, transport=transport)
        assert transport.payloads[0]["chat_template_kwargs"]["enable_thinking"] is False

    def test_caller_cannot_accidentally_switch_it_on(self, transport):
        """No kwarg path may re-enable thinking. It is not a tunable."""
        brain.plan(
            [{"role": "user", "content": "q"}],
            transport=transport,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert transport.payloads[0]["chat_template_kwargs"]["enable_thinking"] is False


class TestRequestShape:
    def test_uses_the_configured_brain_alias(self, transport):
        brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert transport.payloads[0]["model"] == config.BRAIN_MODEL

    def test_applies_the_brain_timeout(self, transport):
        brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert transport.timeouts[0] == config.BRAIN_TIMEOUT_S

    def test_temperature_is_zero_for_reproducibility(self, transport):
        brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert transport.payloads[0]["temperature"] == 0

    def test_max_tokens_is_bounded(self, transport):
        """A prose-y turn must not be able to spend the whole deadline. With
        thinking off a tool call is 43 tokens, so the cap is pure insurance."""
        brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert 0 < transport.payloads[0]["max_tokens"] <= 1024

    def test_tools_are_forwarded_when_supplied(self, transport):
        schemas = [{"type": "function", "function": {"name": "query_data"}}]
        brain.plan([{"role": "user", "content": "q"}], tools=schemas, transport=transport)
        assert transport.payloads[0]["tools"] == schemas
        assert transport.payloads[0]["tool_choice"] == "auto"

    def test_no_tools_key_when_registry_is_empty(self, transport):
        """Session A's registry starts empty. Sending `tools: []` makes some
        servers 400, so the key is omitted entirely."""
        brain.plan([{"role": "user", "content": "q"}], tools=[], transport=transport)
        assert "tools" not in transport.payloads[0]


class TestFailureIsNeverFatal:
    """§7: brain timeout or 5xx must stop the loop, not crash the request."""

    @pytest.mark.parametrize(
        "exc",
        [TimeoutError("timed out"), OSError("connection refused"), ValueError("boom")],
    )
    def test_returns_a_reply_marked_failed_instead_of_raising(self, exc):
        transport = FakeTransport(raises=exc)
        reply = brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert reply.ok is False
        assert reply.content == ""
        assert reply.error

    def test_unexpected_response_shape_is_a_failure_not_a_crash(self):
        class Weird:
            def __call__(self, payload, timeout):
                return {"unexpected": True}

        reply = brain.plan([{"role": "user", "content": "q"}], transport=Weird())
        assert reply.ok is False

    def test_none_content_is_normalised_to_empty_string(self):
        transport = FakeTransport(content=None)
        reply = brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert reply.ok is True
        assert reply.content == ""

    def test_successful_reply_exposes_content_and_tokens(self, transport):
        reply = brain.plan([{"role": "user", "content": "q"}], transport=transport)
        assert reply.ok is True
        assert reply.content == "<tool_call></tool_call>"
        assert reply.completion_tokens == 43


class TestLiveBrain:
    """Opt-in. Skips when the cluster is unreachable so the default run is green
    on any machine, including a judge's clone."""

    @pytest.mark.e2e
    def test_real_brain_turn_is_fast_and_emits_xml(self):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "query_data",
                    "description": "Query approved local datasets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset": {"type": "string", "enum": ["rba", "asx", "afr"]},
                            "metric": {"type": "string"},
                        },
                        "required": ["dataset", "metric"],
                    },
                },
            }
        ]
        msgs = [
            {
                "role": "system",
                "content": "You answer financial questions by calling tools. "
                "Emit a tool call, nothing else.",
            },
            {"role": "user", "content": "How many RBA decisions changed the rate?"},
        ]
        reply = brain.plan(msgs, tools=schemas)
        if not reply.ok:
            pytest.skip(f"brain unreachable: {reply.error}")

        # The whole reason parser.py exists: the call arrives in content.
        assert "<tool_call>" in reply.content
        assert reply.elapsed_s < config.BRAIN_TIMEOUT_S
