"""Budget tests — turn cap, wall deadline, context clamp.

All three exist because two hard limits sit just outside the happy path: the
60s scored penalty boundary, and `max_model_len 4096` on both models.

Deadlines are tested with an injected clock, never `sleep`, so the suite stays
at milliseconds.
"""

from __future__ import annotations

import pytest

from src import config
from src.agent import budget

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


class TestDeadline:
    def test_fresh_budget_is_not_breached(self, clock):
        b = budget.Budget(clock=clock)
        assert not b.breached()
        assert b.remaining_s() == pytest.approx(config.LOOP_DEADLINE_S)

    def test_breaches_exactly_at_the_loop_deadline(self, clock):
        b = budget.Budget(clock=clock)
        clock.advance(config.LOOP_DEADLINE_S - 0.01)
        assert not b.breached()
        clock.advance(0.02)
        assert b.breached()

    def test_the_deadline_bounds_the_loop_not_the_request(self):
        """F10. 45s of looping + a 15s synthesis timeout is 60s before FastAPI
        even serialises, which lands on the wrong side of the -20% line. The
        loop stops at 40 so synthesis keeps its full budget inside the envelope.
        """
        assert config.LOOP_DEADLINE_S == 40.0
        assert (
            config.LOOP_DEADLINE_S + config.SYNTH_TIMEOUT_S + config.SAFETY_MARGIN_S
            == config.PENALTY_THRESHOLD_S
        )

    def test_remaining_never_goes_negative(self, clock):
        b = budget.Budget(clock=clock)
        clock.advance(999.0)
        assert b.remaining_s() == 0.0

    def test_elapsed_is_reported_for_the_penalised_score(self, clock):
        b = budget.Budget(clock=clock)
        clock.advance(3.5)
        assert b.elapsed_s() == pytest.approx(3.5)


class TestTurnCap:
    def test_allows_exactly_max_turns(self, clock):
        b = budget.Budget(clock=clock)
        allowed = 0
        while b.may_take_turn():
            b.record_turn()
            allowed += 1
            if allowed > 10:
                pytest.fail("turn cap never engaged")
        assert allowed == config.MAX_TURNS

    def test_a_breached_deadline_stops_turns_even_under_the_cap(self, clock):
        b = budget.Budget(clock=clock)
        b.record_turn()
        clock.advance(config.LOOP_DEADLINE_S + 1)
        assert not b.may_take_turn()

    def test_turns_are_not_steps(self, clock):
        """kickoff F9: one brain turn can emit several <tool_call> blocks, so
        `steps` in the response and MAX_TURNS are different quantities."""
        b = budget.Budget(clock=clock)
        b.record_turn()
        b.record_calls(4)
        assert b.turns == 1
        assert b.steps == 4


class TestClamp:
    def test_clamps_to_the_configured_cap(self):
        out = budget.clamp("x" * 5000)
        assert len(out) <= config.TOOL_RESULT_CHAR_CAP

    def test_short_results_are_untouched(self):
        assert budget.clamp("41 of 175") == "41 of 175"

    def test_truncation_is_announced_not_silent(self):
        """A silently cut result teaches the brain the data ended there."""
        out = budget.clamp("x" * 5000)
        assert "truncated" in out.lower()

    def test_uses_the_same_cap_as_training(self):
        """F2: training clamps tool_results at 1200 against max_seq_len 512.
        Inference must match or the adapter sees a shape it never trained on."""
        assert config.TOOL_RESULT_CHAR_CAP == 1200

    def test_none_and_non_strings_survive(self):
        assert budget.clamp(None) == ""
        assert budget.clamp(41) == "41"


class TestToolResultMessage:
    def test_uses_role_tool(self):
        """F15, probed live against agent-brain: a role:"tool" message with no
        tool_call_id and no preceding tool_calls entry is ACCEPTED (HTTP 200,
        coherent reply). So no <tool_response> user-message fallback is needed.
        """
        msg = budget.tool_result_message("rba", "41 of 175 changed")
        assert msg["role"] == "tool"
        assert "41 of 175 changed" in msg["content"]

    def test_content_is_clamped(self):
        msg = budget.tool_result_message("afr", "x" * 9000)
        assert len(msg["content"]) <= config.TOOL_RESULT_CHAR_CAP + 64

    def test_names_the_tool_so_the_brain_can_attribute_the_result(self):
        msg = budget.tool_result_message("asx", "ok")
        assert "asx" in str(msg)


class TestContextTrim:
    def _msgs(self, n_tool: int, tool_len: int = 400):
        msgs = [
            {"role": "system", "content": "SYSTEM PROMPT with tool docs"},
            {"role": "user", "content": "THE QUESTION"},
        ]
        for i in range(n_tool):
            msgs.append({"role": "tool", "name": f"t{i}", "content": f"{i}" * tool_len})
        return msgs

    def test_small_conversation_is_unchanged(self):
        msgs = self._msgs(1, 10)
        assert budget.trim(msgs) == msgs

    def test_system_prompt_and_question_always_survive(self):
        """Non-negotiable: drop the question and the answer is unanswerable."""
        trimmed = budget.trim(self._msgs(40, 900), max_tokens=200)
        assert trimmed[0]["content"] == "SYSTEM PROMPT with tool docs"
        assert any(m["content"] == "THE QUESTION" for m in trimmed)

    def test_oldest_tool_results_are_dropped_first(self):
        trimmed = budget.trim(self._msgs(6, 900), max_tokens=400)
        kept = [m["name"] for m in trimmed if m["role"] == "tool"]
        assert kept, "at least one tool result should survive"
        # The most recent result is the one the brain is reasoning about.
        assert "t5" in kept
        assert "t0" not in kept

    def test_trims_below_the_token_ceiling(self):
        trimmed = budget.trim(self._msgs(30, 900), max_tokens=500)
        assert budget.estimate_tokens(trimmed) <= 500

    def test_ceiling_defaults_to_config(self):
        assert config.MAX_CONTEXT_TOKENS == 3000
        assert config.MAX_CONTEXT_TOKENS < 4096  # both models' max_model_len

    def test_dropping_everything_still_leaves_a_usable_conversation(self):
        """Even an absurd ceiling must not produce an empty message list."""
        trimmed = budget.trim(self._msgs(10, 5000), max_tokens=1)
        assert len(trimmed) >= 2
        assert trimmed[0]["role"] == "system"
