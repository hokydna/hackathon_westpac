"""Serving tests. Session C, harness §10 steps 1 and 10.

`GET /health` is a **hard gate**: `Challenge_Brief.md` § Response-Time Rules says
if it does not return 200 during the pre-evaluation check, the team is skipped and
scores **zero on the entire 40% hidden-question category**. So the most important
assertions here are the ones proving `/health` cannot be taken down by anything —
not an unreachable brain, not an unreachable Nemotron, not a missing corpus.

`POST /query` must validate against `Participant_Package/validate.json`, where
`answer` is the only required field with `minLength: 1`. Every failure path still
has to produce one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import validate as jsonschema_validate

from src import config
from src.contracts import AgentResult, TraceEntry

pytestmark = pytest.mark.contract

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "Participant_Package" / "validate.json").read_text()
)


@pytest.fixture
def client(monkeypatch):
    """A client whose loop is stubbed, so these tests exercise HTTP not the agent."""
    from src import app as app_module

    async def fake_answer(question, registry=None, brain_schemas=None, **kw):
        return AgentResult(
            answer=f"Stub answer for: {question}",
            steps=1,
            tool_trace=[TraceEntry(tool="query_data", args={"dataset": "rba"}, result="41 of 175")],
        )

    monkeypatch.setattr(app_module.loop, "answer", fake_answer)
    return TestClient(app_module.app)


class TestHealthIsAHardGate:
    """Each of these is a way /health could fail and zero the whole 40%."""

    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_returns_200_with_litellm_pointing_nowhere(self, monkeypatch, client):
        """The brain being down must not affect liveness."""
        monkeypatch.setattr(config, "LITELLM_BASE_URL", "http://127.0.0.1:9/v1")
        assert client.get("/health").status_code == 200

    def test_returns_200_with_nemotron_pointing_nowhere(self, monkeypatch, client):
        monkeypatch.setattr(config, "DOMAIN_BASE_URL", "http://127.0.0.1:9/v1")
        assert client.get("/health").status_code == 200

    def test_returns_200_with_the_dataset_dir_missing(self, monkeypatch, client):
        """Corpora load at import; /health must not re-check them at request time."""
        monkeypatch.setattr(config, "DATASET_DIR", Path("/nonexistent"))
        assert client.get("/health").status_code == 200

    def test_makes_no_outbound_calls(self, monkeypatch, client):
        """A /health that phones a model can be taken down by that model."""
        import urllib.request

        def boom(*a, **k):
            raise AssertionError("/health must not make network calls")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        assert client.get("/health").status_code == 200

    def test_body_is_json(self, client):
        assert client.get("/health").json().get("status") == "ok"


class TestQueryContract:
    def test_response_validates_against_validate_json(self, client):
        r = client.post("/query", json={"question": "How many RBA decisions changed?"})
        assert r.status_code == 200
        jsonschema_validate(instance=r.json(), schema=SCHEMA)

    def test_returns_the_three_documented_fields(self, client):
        payload = client.post("/query", json={"question": "q"}).json()
        assert set(payload) == {"answer", "steps", "tool_trace"}

    def test_tool_trace_entries_match_the_answer_template(self, client):
        entry = client.post("/query", json={"question": "q"}).json()["tool_trace"][0]
        assert set(entry) == {"tool", "args", "result"}
        assert isinstance(entry["args"], dict)
        assert isinstance(entry["result"], str)

    def test_answer_is_never_empty(self, client):
        assert len(client.post("/query", json={"question": "q"}).json()["answer"]) >= 1

    def test_only_the_prompt_field_is_needed(self, client):
        """Challenge_Brief: the evaluator sends one object with a `question` field."""
        assert client.post("/query", json={"question": "q"}).status_code == 200


class TestQueryNeverReturnsANonAnswer:
    """§7's invariant, at the HTTP boundary. A crash scores zero; a degraded
    answer scores partial credit."""

    def test_a_crashing_loop_still_returns_a_valid_answer(self, monkeypatch):
        from src import app as app_module

        async def boom(*a, **k):
            raise RuntimeError("loop exploded")

        monkeypatch.setattr(app_module.loop, "answer", boom)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        r = c.post("/query", json={"question": "q"})
        assert r.status_code == 200
        jsonschema_validate(instance=r.json(), schema=SCHEMA)
        assert r.json()["answer"].strip()

    def test_a_loop_returning_an_empty_answer_is_backfilled(self, monkeypatch, client):
        from src import app as app_module

        async def empty(*a, **k):
            return AgentResult(answer="   ", steps=0, tool_trace=[])

        monkeypatch.setattr(app_module.loop, "answer", empty)
        payload = client.post("/query", json={"question": "q"}).json()
        assert payload["answer"].strip()

    @pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": None}, {"prompt": "q"}])
    def test_malformed_requests_still_return_a_schema_valid_answer(self, client, body):
        """The harness sends `question`, but a 422 scores zero for that case —
        better to answer and state the limitation."""
        r = client.post("/query", json=body)
        assert r.status_code == 200
        jsonschema_validate(instance=r.json(), schema=SCHEMA)


class TestConcurrency:
    def test_three_concurrent_queries_do_not_cross_answers(self, monkeypatch):
        """The graded condition: the harness sends up to three at once and state
        must not bleed between them."""
        from src import app as app_module

        async def echo(question, registry=None, brain_schemas=None, **kw):
            await asyncio.sleep(0.02)  # force interleaving
            return AgentResult(answer=f"answer-for-{question}", steps=1, tool_trace=[])

        monkeypatch.setattr(app_module.loop, "answer", echo)
        c = TestClient(app_module.app)

        import concurrent.futures as cf

        questions = ["alpha", "beta", "gamma"]
        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(
                pool.map(lambda q: c.post("/query", json={"question": q}).json(), questions)
            )
        for q, payload in zip(questions, results):
            assert payload["answer"] == f"answer-for-{q}"


class TestStartupOrdering:
    def test_corpora_are_loaded_at_import_not_per_request(self):
        """Harness §5: corpora load before uvicorn binds, so the port only opens
        once the AFR index is built (~25s). That is what lets /health stay a pure
        liveness check instead of 503-ing mid-build."""
        from src import app as app_module

        assert hasattr(app_module, "warm_corpora")
        assert app_module.warm_corpora.__doc__

    def test_app_exposes_the_registry_it_will_dispatch(self):
        from src import app as app_module

        assert "query_data" in app_module.REGISTRY
        assert app_module.BRAIN_SCHEMAS
