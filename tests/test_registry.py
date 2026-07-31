"""Registry tests — the tool surface the brain actually sees.

Two properties matter more than the individual metrics, because both are scored
and both fail silently:

1. **Tools return prose, not `k=v`.** An earlier draft returned
   `"changed=41; increases=20"` and base Nemotron echoed that shape into the
   answer, which the judge scored 0/10 on MHQ001 despite every number being right.
2. **`domain_sentiment` stays a classifier.** It must never become a
   general-purpose answerer — that is the prohibited "Nemotron as planner and tool
   caller" pattern.
"""

from __future__ import annotations

import re

import pytest

from src import config
from src.tools import registry

pytestmark = pytest.mark.unit


@pytest.fixture
def real_corpus(real_dataset, monkeypatch):
    """Point the CACHED module accessors at the real corpus.

    The registry deliberately calls `corpora.rba_rows()` / `asx_series()` /
    `afr_index()` with no injection point, because that is the production path —
    so a test cannot just pass different rows in. conftest sets DATASET_DIR to the
    fixtures for the whole session, and `needs_dataset` only controls SKIPPING, it
    does not repoint anything. Hence: repoint config and clear the lru_caches.

    Session C will need the same seam for the offline eval. It belongs in
    conftest, but conftest is frozen in the base commit, so it lives here until
    that can be changed on main.
    """
    from src import config
    from src.tools import corpora

    monkeypatch.setattr(config, "DATASET_DIR", real_dataset)
    monkeypatch.setattr(config, "RBA_PATH", real_dataset / "RBA Rates" / "RBA-rates.jsonl")
    monkeypatch.setattr(config, "ASX_DIR", real_dataset / "ASX")
    monkeypatch.setattr(config, "AFR_DIR", real_dataset / "AFR")
    for cached in (corpora.rba_rows, corpora.asx_series, corpora.afr_index):
        cached.cache_clear()
    yield real_dataset
    for cached in (corpora.rba_rows, corpora.asx_series, corpora.afr_index):
        cached.cache_clear()


def test_exports_are_consistent():
    assert {t.name for t in registry.ALL_TOOLS} == set(registry.REGISTRY)
    assert len(registry.BRAIN_SCHEMAS) == len(registry.ALL_TOOLS)


def test_synthesis_is_not_registered_as_a_tool():
    """Role 1 must not be callable by Qwen (kickoff §10). handout/03: "Bad:
    Nemotron used as the planner and tool caller"."""
    names = set(registry.REGISTRY)
    assert "synth" not in names and "synthesize" not in names
    assert names == {"query_data", "coverage", "domain_sentiment"}


class TestSchemas:
    def test_every_schema_is_a_valid_openai_tool_entry(self):
        for s in registry.BRAIN_SCHEMAS:
            assert s["type"] == "function"
            fn = s["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_query_data_advertises_exclude_tickers(self):
        """5 of 15 questions need it; if it is not in the schema the brain cannot
        ask for it."""
        props = registry.REGISTRY["query_data"].parameters["properties"]
        assert "exclude_tickers" in props
        assert "TAH.AX" in props["exclude_tickers"]["description"]

    def test_metric_description_names_every_implemented_metric(self):
        desc = registry.REGISTRY["query_data"].parameters["properties"]["metric"]["description"]
        for m in ("count_changes", "cycle_summary", "window_return", "avg_volume",
                  "count_by_month", "retrieve_by_headline", "max_drawdown"):
            assert m in desc, m


class TestProseOutput:
    """The 0/10 regression guard."""

    @pytest.mark.needs_dataset
    @pytest.mark.parametrize(
        "args",
        [
            {"dataset": "rba", "metric": "count_changes"},
            {"dataset": "rba", "metric": "extremes"},
            {"dataset": "asx", "metric": "describe"},
            {"dataset": "asx", "metric": "avg_volume", "exclude_tickers": ["Tabcorp"]},
            {"dataset": "afr", "metric": "count_by_month", "pattern": "unemployment"},
        ],
    )
    async def test_results_are_sentences_not_key_value_dumps(self, real_corpus, args):
        out = await registry.REGISTRY["query_data"].ainvoke(args)
        assert out.endswith("."), out
        assert "=" not in out, "k=v dump — base Nemotron will echo this shape"
        assert re.search(r"[a-z]{4,} [a-z]{4,}", out), "should read as prose"

    @pytest.mark.needs_dataset
    async def test_mhq001_output_carries_all_three_numbers_in_one_sentence(self, real_corpus):
        """A single 10-point all-or-nothing component."""
        out = await registry.REGISTRY["query_data"].ainvoke(
            {"dataset": "rba", "metric": "count_changes"}
        )
        first = out.split(". ")[0]
        for n in ("41", "175", "20", "21"):
            assert n in first, f"{n} missing from the first sentence"

    @pytest.mark.needs_dataset
    async def test_typed_payload_rides_alongside_the_prose(self, real_corpus):
        """The adapter was trained on structured evidence, so last_data matters."""
        tool = registry.REGISTRY["query_data"]
        await tool.ainvoke({"dataset": "rba", "metric": "count_changes"})
        assert tool.last_data["changed"] == 41
        assert tool.last_data["increases"] == 20

    @pytest.mark.needs_dataset
    async def test_results_fit_the_context_clamp(self, real_corpus):
        """window_return over 17 tickers is the most verbose result we produce."""
        out = await registry.REGISTRY["query_data"].ainvoke(
            {"dataset": "asx", "metric": "window_return",
             "date_from": "2019-06-05", "date_to": "2019-06-12",
             "exclude_tickers": ["Tabcorp"]}
        )
        assert len(out) <= config.TOOL_RESULT_CHAR_CAP


class TestUnknownInput:
    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"dataset": "nope", "metric": "count"},
            {"dataset": "rba", "metric": "nope"},
            {"dataset": "asx", "metric": "nope"},
            {"dataset": "afr", "metric": "nope"},
        ],
    )
    async def test_never_raises_and_names_the_alternatives(self, args):
        """§7: the brain must be able to replan, which needs the valid options."""
        out = await registry.REGISTRY["query_data"].ainvoke(args)
        assert isinstance(out, str) and out
        assert "Available" in out or "Unknown" in out


class TestDomainSentiment:
    """Role 2 — the one tool that calls Nemotron."""

    async def test_denied_without_a_retrieved_article(self):
        """Stops Qwen using Nemotron as a general answerer on a numeric question."""
        out = await registry.REGISTRY["domain_sentiment"].ainvoke({})
        assert out.startswith("ERROR")
        assert "retrieve_by_headline" in out

    async def test_output_is_clamped_to_a_classification(self, monkeypatch):
        from src import domain_client

        monkeypatch.setattr(domain_client, "complete", lambda *a, **k: "x" * 5000)
        out = await registry.REGISTRY["domain_sentiment"].ainvoke({"headline": "Banks rally"})
        assert len(out) <= config.SENTIMENT_CHAR_CAP

    async def test_degrades_to_a_note_when_nemotron_is_down(self, monkeypatch):
        """The deterministic rate-lookup component of a sentiment question must
        still be answerable."""
        from src import domain_client

        def boom(*a, **k):
            raise domain_client.DomainUnavailable("unreachable")

        monkeypatch.setattr(domain_client, "complete", boom)
        out = await registry.REGISTRY["domain_sentiment"].ainvoke({"headline": "Banks rally"})
        assert "unavailable" in out.lower()
        assert not out.startswith("ERROR:")

    async def test_uses_the_frozen_sentiment_renderer(self, monkeypatch):
        from src import domain_client, prompts

        seen = {}
        monkeypatch.setattr(
            domain_client, "complete",
            lambda s, u, **k: seen.update(system=s, user=u) or "Positive. Higher.",
        )
        await registry.REGISTRY["domain_sentiment"].ainvoke(
            {"headline": "Banks rally", "article_text": "Shares rose", "rba_rate": "0.10"}
        )
        assert seen["system"] == prompts.SENTIMENT_SYSTEM
        assert "Banks rally" in seen["user"] and "0.10" in seen["user"]


class TestCoverageTool:
    @pytest.mark.needs_dataset
    async def test_states_the_evidence_boundary_explicitly(self, real_corpus):
        """MHQ090's refusal is worth 10 points across three components, and "No"
        alone earns 3.33 — the reasoning carries the rest."""
        out = await registry.REGISTRY["coverage"].ainvoke({})
        assert "2021" in out
        assert "unsupported" in out.lower()
