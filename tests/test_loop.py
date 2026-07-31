"""Loop tests — driven by a fake brain and stub tools.

Session B never needs session A's real tools to make progress: the loop is
exercised by a fake brain returning canned XML against a stub registry.

The invariants under test, in order of how much they cost if broken:

1. Synthesis runs on EVERY exit path (two 30% categories score the separation).
2. The answer is never empty (§7; `validate.json` minLength 1).
3. Three concurrent requests do not cross state (the graded condition).
4. The turn cap and the wall deadline both terminate the loop.
"""

from __future__ import annotations

import asyncio

import pytest

from src import config, domain_client
from src.agent import brain, loop

pytestmark = pytest.mark.integration


XML = (
    "<tool_call><function=query_data>"
    "<parameter=dataset>rba</parameter>"
    "<parameter=metric>count_changes</parameter>"
    "</function></tool_call>"
)


class FakeBrain:
    """Returns canned replies in order, then stops emitting calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen_messages = []

    def __call__(self, messages, tools=None, timeout=None, transport=None, **kw):
        self.calls += 1
        self.seen_messages.append(list(messages))
        content = self.replies.pop(0) if self.replies else "All values gathered."
        return brain.BrainReply(content=content, ok=True)


class StubTool:
    def __init__(self, result="41 of 175 changed", data=None, raises=None):
        self.result = result
        self.last_data = data
        self.raises = raises
        self.args_schema = None
        self.invocations = []

    async def ainvoke(self, args):
        self.invocations.append(args)
        if self.raises:
            raise self.raises
        return self.result


@pytest.fixture(autouse=True)
def mock_domain(monkeypatch):
    """Synthesis via mock mode -- no adapter needed."""
    monkeypatch.setattr(config, "DOMAIN_PREDICT_MODE", "mock")
    yield


@pytest.fixture
def fake_brain(monkeypatch):
    fb = FakeBrain([XML])
    monkeypatch.setattr(brain, "plan", fb)
    return fb


class TestSynthesisIsUnconditional:
    async def test_runs_on_the_normal_path(self, fake_brain):
        res = await loop.answer("How many RBA decisions changed?", {"query_data": StubTool()})
        assert res.answer.strip()

    async def test_runs_when_the_brain_is_unreachable(self, monkeypatch):
        monkeypatch.setattr(
            brain, "plan", lambda *a, **k: brain.BrainReply("", ok=False, error="timeout")
        )
        res = await loop.answer("q", {"query_data": StubTool()})
        assert res.answer.strip()
        assert res.tool_trace == []

    async def test_runs_when_every_call_is_denied(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML]))
        res = await loop.answer("q", {})  # empty registry -> all denied
        assert res.answer.strip()
        assert res.tool_trace and "not executed" in res.tool_trace[0].result

    async def test_runs_when_a_tool_raises(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML]))
        res = await loop.answer("q", {"query_data": StubTool(raises=RuntimeError("boom"))})
        assert res.answer.strip()
        assert "ERROR" in res.tool_trace[0].result

    async def test_is_never_registered_as_a_tool(self):
        """Role 1 must not be callable by Qwen. handout/03: "Bad: Nemotron used
        as the planner and tool caller"."""
        from src import contracts

        names = {getattr(t, "name", "") for t in contracts.ALL_TOOLS}
        assert "synth" not in names
        assert not any("synth" in n or "nemotron" in n.lower() for n in names)


class TestTermination:
    async def test_stops_when_the_brain_emits_no_calls(self, monkeypatch):
        fb = FakeBrain(["Nothing to look up; the answer is 41."])
        monkeypatch.setattr(brain, "plan", fb)
        res = await loop.answer("q", {"query_data": StubTool()})
        assert fb.calls == 1
        assert res.tool_trace == []

    async def test_honours_the_turn_cap(self, monkeypatch):
        # Always emits a call, so only the cap can stop it.
        fb = FakeBrain([XML] * 50)
        monkeypatch.setattr(brain, "plan", fb)
        await loop.answer("q", {"query_data": StubTool()})
        assert fb.calls == config.MAX_TURNS

    async def test_deadline_breach_synthesizes_from_the_partial_trace(self, monkeypatch):
        fb = FakeBrain([XML] * 50)
        monkeypatch.setattr(brain, "plan", fb)

        class Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                self.t += config.LOOP_DEADLINE_S  # breach after the first turn
                return self.t

        res = await loop.answer("q", {"query_data": StubTool()}, clock=Clock())
        assert res.answer.strip()
        assert fb.calls <= config.MAX_TURNS


class TestTraceShape:
    async def test_matches_the_answer_template(self, fake_brain):
        res = await loop.answer("q", {"query_data": StubTool()})
        payload = res.to_response()
        assert set(payload) == {"answer", "steps", "tool_trace"}
        assert set(payload["tool_trace"][0]) == {"tool", "args", "result"}

    async def test_steps_counts_executed_calls_not_turns(self, monkeypatch):
        two = XML + XML  # one TURN emitting two calls
        monkeypatch.setattr(brain, "plan", FakeBrain([two]))
        res = await loop.answer("q", {"query_data": StubTool()})
        assert res.steps == 2

    async def test_coerced_args_reach_the_tool(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML]))
        tool = StubTool()
        await loop.answer("q", {"query_data": tool})
        assert tool.invocations == [{"dataset": "rba", "metric": "count_changes"}]

    async def test_structured_tool_data_is_carried_for_synthesis(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML]))
        tool = StubTool(data={"changes": 41, "increases": 20, "decreases": 21})
        res = await loop.answer("q", {"query_data": tool})
        assert getattr(res.tool_trace[0], "data", None) == {
            "changes": 41,
            "increases": 20,
            "decreases": 21,
        }

    async def test_tool_results_are_fed_back_to_the_brain(self, monkeypatch):
        fb = FakeBrain([XML, "done"])
        monkeypatch.setattr(brain, "plan", fb)
        await loop.answer("q", {"query_data": StubTool(result="41 of 175")})
        second_turn = fb.seen_messages[1]
        assert any(m.get("role") == "tool" and "41 of 175" in m["content"] for m in second_turn)


class TestConcurrency:
    async def test_three_concurrent_requests_do_not_cross_answers(self, monkeypatch):
        """The graded condition: the harness sends up to three at once and state
        must not bleed. §7 claims safety by construction — construction claims
        still need a test."""

        def per_question_brain(messages, tools=None, **kw):
            question = next(m["content"] for m in messages if m["role"] == "user")
            if any(m.get("role") == "tool" for m in messages):
                return brain.BrainReply(content="done", ok=True)
            return brain.BrainReply(
                content=(
                    "<tool_call><function=query_data>"
                    f"<parameter=dataset>{question}</parameter>"
                    "<parameter=metric>m</parameter>"
                    "</function></tool_call>"
                ),
                ok=True,
            )

        monkeypatch.setattr(brain, "plan", per_question_brain)

        class EchoTool:
            args_schema = None
            last_data = None

            async def ainvoke(self, args):
                await asyncio.sleep(0.01)  # force interleaving
                return f"result-for-{args['dataset']}"

        results = await asyncio.gather(
            *[loop.answer(q, {"query_data": EchoTool()}) for q in ("alpha", "beta", "gamma")]
        )

        for question, res in zip(("alpha", "beta", "gamma"), results):
            assert res.tool_trace, f"{question} produced no trace"
            assert res.tool_trace[0].args["dataset"] == question
            assert f"result-for-{question}" in res.tool_trace[0].result


class TestDuplicateCallElision:
    """Observed live: the brain re-emitted an identical query_data call on all
    three turns and spent the entire cap on one question. Tools are
    deterministic over a read-only corpus, so a repeat cannot return a different
    answer — serving it from the trace preserves the answer and leaves turns for
    the follow-up calls hard cross-dataset questions need."""

    async def test_identical_repeat_is_not_re_executed(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML, XML, XML]))
        tool = StubTool()
        res = await loop.answer("q", {"query_data": tool})
        assert len(tool.invocations) == 1, "tool should run once, not once per turn"
        assert res.answer.strip()

    async def test_the_cached_result_still_reaches_the_brain(self, monkeypatch):
        fb = FakeBrain([XML, XML])
        monkeypatch.setattr(brain, "plan", fb)
        await loop.answer("q", {"query_data": StubTool(result="41 of 175")})
        # Every turn after the first must still see the result in context.
        assert any(
            m.get("role") == "tool" and "41 of 175" in m["content"]
            for m in fb.seen_messages[-1]
        )

    async def test_different_args_are_still_executed(self, monkeypatch):
        other = XML.replace("count_changes", "extremes")
        monkeypatch.setattr(brain, "plan", FakeBrain([XML + other]))
        tool = StubTool()
        await loop.answer("q", {"query_data": tool})
        assert len(tool.invocations) == 2

    async def test_trace_records_the_call_once(self, monkeypatch):
        monkeypatch.setattr(brain, "plan", FakeBrain([XML, XML, XML]))
        res = await loop.answer("q", {"query_data": StubTool()})
        assert len(res.tool_trace) == 1
