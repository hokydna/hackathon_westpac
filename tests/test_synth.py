"""Synthesis tests — Role 1, and the never-return-a-non-answer invariant.

§7's most valuable promise is that every failure path still yields a valid
`answer`. `validate.json` requires only `answer` with minLength 1, so a Nemotron
outage must cost partial credit, never the whole question. That invariant is the
most heavily tested thing here.
"""

from __future__ import annotations

import pytest

from src import config, domain_client, prompts
from src.agent import synth
from src.contracts import TraceEntry

pytestmark = pytest.mark.unit


def entry(tool="query_data", args=None, result="ok", data=None):
    e = TraceEntry(tool=tool, args=args or {}, result=result)
    if data is not None:
        e.data = data  # type: ignore[attr-defined]
    return e


STRUCTURED = [
    entry(
        result="CBA.AX 2017 annualised volatility 14.46%",
        data={
            "ticker": "CBA.AX",
            "year": 2017,
            "volatility_pct_annualised": 14.464909945587081,
            "basis": "daily simple close-to-close returns, sample stdev",
        },
    )
]


class TestNeverReturnsANonAnswer:
    """One test per failure path. Every one asserts a non-empty answer."""

    @pytest.mark.parametrize(
        "exc",
        [
            domain_client.DomainUnavailable("timeout"),
            domain_client.DomainUnavailable("404 nemotron-8b-finance does not exist"),
        ],
    )
    def test_model_failure_falls_back_to_the_deterministic_template(self, monkeypatch, exc):
        def boom(*a, **k):
            raise exc

        monkeypatch.setattr(domain_client, "complete", boom)
        out = synth.write("How volatile was CBA.AX in 2017?", STRUCTURED)
        assert out.answer.strip()
        assert out.mode == "fallback"
        # The template must carry the real number, not a placeholder.
        assert "14.46" in out.answer

    def test_empty_completion_falls_back(self, monkeypatch):
        monkeypatch.setattr(domain_client, "complete", lambda *a, **k: "   ")
        out = synth.write("q", STRUCTURED)
        assert out.answer.strip()
        assert out.mode == "fallback"

    def test_empty_trace_still_answers_and_states_the_limitation(self):
        out = synth.write("q", [])
        assert out.answer.strip()
        assert "could not" in out.answer.lower()

    def test_total_failure_of_both_model_and_template_still_answers(self, monkeypatch):
        monkeypatch.setattr(
            domain_client, "complete", lambda *a, **k: (_ for _ in ()).throw(
                domain_client.DomainUnavailable("down")
            )
        )
        monkeypatch.setattr(prompts, "deterministic_fallback", lambda **k: "")
        out = synth.write("q", STRUCTURED)
        assert out.answer.strip()

    def test_answer_is_never_empty_across_a_matrix_of_failures(self, monkeypatch):
        cases = [
            lambda *a, **k: (_ for _ in ()).throw(domain_client.DomainUnavailable("x")),
            lambda *a, **k: "",
            lambda *a, **k: None or "",
        ]
        for fn in cases:
            monkeypatch.setattr(domain_client, "complete", fn)
            assert len(synth.write("q", STRUCTURED).answer) >= 1


class TestTrainedShape:
    """The adapter was trained on structured evidence + a component list."""

    def test_structured_tool_data_becomes_verified_evidence(self, monkeypatch):
        seen = {}

        def spy(system, user, **kwargs):
            seen["system"], seen["user"] = system, user
            return "CBA.AX's annualised volatility in 2017 was 14.46%."

        monkeypatch.setattr(domain_client, "complete", spy)
        synth.write("How volatile was CBA.AX in 2017?", STRUCTURED)

        assert seen["system"] == prompts.SYNTH_SYSTEM
        for block in ("Question:", "Requested components:", "Verified evidence:", "Limitations:"):
            assert block in seen["user"]
        # The typed value, not a stringified summary.
        assert "14.464909945587081" in seen["user"]

    def test_context_keys_are_not_offered_as_requested_components(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            domain_client,
            "complete",
            lambda s, u, **k: seen.setdefault("user", u) or "answer",
        )
        synth.write("q", STRUCTURED)
        components_block = seen["user"].split("Requested components:")[1].split("Verified evidence:")[0]
        assert "volatility_pct_annualised" in components_block
        # `ticker` and `year` describe the query, not the answer.
        assert '"ticker"' not in components_block
        assert '"year"' not in components_block

    def test_caller_supplied_components_win_over_inference(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            domain_client,
            "complete",
            lambda s, u, **k: seen.setdefault("user", u) or "answer",
        )
        synth.write("q", STRUCTURED, requested_components=["basis"])
        block = seen["user"].split("Requested components:")[1].split("Verified evidence:")[0]
        assert "basis" in block
        assert "volatility_pct_annualised" not in block


class TestUnstructuredDegradation:
    """String-only evidence is a shape the adapter never trained on. It must
    still work, and it must be flagged so the risk is visible in logs."""

    def test_string_results_are_used_but_flagged(self, monkeypatch):
        monkeypatch.setattr(domain_client, "complete", lambda s, u, **k: "an answer")
        out = synth.write("q", [entry(result="41 of 175 changed")])
        assert out.answer == "an answer"
        assert "structured" in out.error

    def test_two_calls_to_the_same_tool_do_not_collide(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            domain_client,
            "complete",
            lambda s, u, **k: seen.setdefault("user", u) or "a",
        )
        synth.write("q", [entry(result="first"), entry(result="second")])
        assert "first" in seen["user"] and "second" in seen["user"]


class TestModes:
    def test_mock_mode_needs_no_network_and_is_obviously_synthetic(self):
        out = synth.write("q", STRUCTURED, mode="mock")
        assert out.answer
        assert "mock" in out.answer.lower()

    def test_mode_defaults_to_config(self, monkeypatch):
        monkeypatch.setattr(config, "DOMAIN_PREDICT_MODE", "mock")
        assert synth.write("q", STRUCTURED).mode == "mock"

    def test_synthesis_timeout_is_passed_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            domain_client,
            "complete",
            lambda s, u, **k: seen.update(k) or "a",
        )
        synth.write("q", STRUCTURED)
        assert seen["timeout"] == config.SYNTH_TIMEOUT_S
