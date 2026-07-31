"""Guards on the frozen base commit.

These are not ceremony. Each one catches a defect that silently costs points
and that no other test in any session would catch, because each spans a
boundary between two sessions' files.
"""

from __future__ import annotations

import json
import re

import pytest

from src import config, contracts, domain_client, prompts
from src.agent import tracing

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# The latency budget (kickoff F10)
# --------------------------------------------------------------------------


def test_budget_invariant_sums_to_the_penalty_threshold():
    """45 + 15 = 60 was the original bug: it landed exactly ON the -20% line.

    If someone retunes one constant, this fails rather than silently pushing
    the pathological case across a scored boundary.
    """
    assert (
        config.LOOP_DEADLINE_S + config.SYNTH_TIMEOUT_S + config.SAFETY_MARGIN_S
        == config.PENALTY_THRESHOLD_S
    )
    assert config.LOOP_DEADLINE_S == 40.0
    assert config.PENALTY_THRESHOLD_S == 60.0


def test_loop_deadline_leaves_room_for_synthesis():
    assert config.LOOP_DEADLINE_S + config.SYNTH_TIMEOUT_S < config.PENALTY_THRESHOLD_S


def test_brain_timeout_is_well_under_the_loop_deadline():
    assert config.BRAIN_TIMEOUT_S * config.MAX_TURNS < config.LOOP_DEADLINE_S


# --------------------------------------------------------------------------
# The tool-result cap (kickoff F2) -- training and inference must agree
# --------------------------------------------------------------------------


def test_tool_result_cap_is_1200_not_2000():
    """FINETUNE_PLAN §4.4 caps training-time tool_results at 1200 against
    max_seq_len=512. If inference clamps at a different number, the adapter is
    served a context shape it never saw. This is the F2 reconciliation."""
    assert config.TOOL_RESULT_CHAR_CAP == 1200
    assert config.AFR_TEXT_CHAR_CAP <= config.TOOL_RESULT_CHAR_CAP


def test_sentiment_output_is_clamped_tighter_than_a_full_answer():
    """Role 2 returns a classification, not an answer (kickoff §10)."""
    assert config.SENTIMENT_CHAR_CAP < config.TOOL_RESULT_CHAR_CAP


# --------------------------------------------------------------------------
# Frozen prompts -- session D trains against these exact bytes
# --------------------------------------------------------------------------


def test_synth_prompt_carries_the_single_sentence_clause():
    """The compound-component defence. Four public questions are a single
    10-point all-or-nothing component bundling 3-4 numbers (26.7% of points);
    splitting one across two sentences scores zero."""
    assert "single sentence" in prompts.SYNTH_SYSTEM
    assert "Do not hedge" in prompts.SYNTH_SYSTEM


def test_synth_user_has_both_placeholders():
    filled = prompts.SYNTH_USER.format(question="q", tool_results="r")
    assert "q" in filled and "r" in filled


def test_sentiment_prompt_forbids_numeric_forecasts():
    """Setup_Instructions.md L95: "Do not force the model to emit a made-up
    numeric return or price forecast." """
    low = prompts.SENTIMENT_SYSTEM.lower()
    assert "never forecast a numeric value" in low
    assert "three sentences" in low


def test_sentiment_user_has_all_four_placeholders():
    filled = prompts.SENTIMENT_USER.format(
        headline="h", publication_date="20150102", article_text="t", rba_rate="2.25"
    )
    for token in ("h", "20150102", "t", "2.25"):
        assert token in filled


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_agent_result_shape_matches_the_answer_template():
    """tool_trace entries must match Participant_Package/answer_template.json."""
    result = contracts.AgentResult(
        answer="a",
        steps=1,
        tool_trace=[contracts.TraceEntry(tool="rba", args={"metric": "count"}, result="41")],
    )
    payload = result.to_response()
    assert set(payload) == {"answer", "steps", "tool_trace"}
    assert set(payload["tool_trace"][0]) == {"tool", "args", "result"}
    # Must survive a JSON round-trip -- the harness parses this.
    assert json.loads(json.dumps(payload)) == payload


def test_registry_exports_are_importable_before_session_a_writes_anything():
    """Session B codes against these on minute one."""
    assert contracts.ALL_TOOLS == []
    assert contracts.BRAIN_SCHEMAS == []


# --------------------------------------------------------------------------
# Tracing must be inert (kickoff F12) -- the scored path runs network-down
# --------------------------------------------------------------------------


def test_tracing_is_off_and_decorator_is_transparent():
    assert tracing.TRACING_ACTIVE is False

    @tracing.traceable(run_type="chain", name="x")
    def called(a, b=2):
        return a + b

    @tracing.traceable
    def bare(a):
        return a * 2

    assert called(1) == 3
    assert bare(4) == 8


# --------------------------------------------------------------------------
# domain_client -- mock mode must work with no adapter served
# --------------------------------------------------------------------------


def test_domain_client_mock_mode_needs_no_network():
    out = domain_client.complete("sys", "user text", mode="mock")
    assert out.startswith("[mock:")
    assert "user text" in out


def test_domain_client_mock_is_obviously_synthetic():
    """A mock answer must never be mistakable for a real one in a trace."""
    assert "mock" in domain_client.complete("s", "u", mode="mock").lower()


# --------------------------------------------------------------------------
# The fixture corpus -- one row per documented gotcha (kickoff F11)
# --------------------------------------------------------------------------


def test_rba_fixture_has_a_bom_and_needs_utf8_sig(fixture_dataset):
    path = fixture_dataset / "RBA Rates" / "RBA-rates.jsonl"
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf", "BOM must be preserved"
    with path.open(encoding="utf-8-sig") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows
    assert set(rows[0]) == {"Effective Date", "Change % points", "Cash rate target%"}
    # Dates are "3 Feb 2010", not ISO. All values are strings.
    assert not re.match(r"^\d{4}-\d{2}-\d{2}$", rows[0]["Effective Date"])
    assert all(isinstance(v, str) for v in rows[0].values())


def test_rba_fixture_contains_a_signed_string_change(fixture_dataset):
    """Change values are strings like "+0.25" / "-0.25" / "0.00", never floats."""
    with (fixture_dataset / "RBA Rates" / "RBA-rates.jsonl").open(
        encoding="utf-8-sig"
    ) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert any(r["Change % points"].startswith(("+", "-")) for r in rows)


def test_config_resolves_dataset_dir_to_the_fixture_corpus():
    """conftest sets DATASET_DIR before src.config imports. If this fails, every
    session's unit tests would silently read the untracked 785 MB corpus and
    stop working in a worktree."""
    assert config.DATASET_DIR.name == "dataset"
    assert config.RBA_PATH.is_file()
    assert config.ASX_DIR.is_dir() and config.AFR_DIR.is_dir()


def test_asx_fixture_filenames_are_company_based_not_ticker_based(fixture_dataset):
    """The gotcha that would break exclude_tickers silently.

    Tabcorp-ASX-2015-2021.jsonl holds ticker TAH.AX. 7 of the 18 real files
    have a stem that does not match their ticker prefix, so a loader that
    derives the ticker from the filename is wrong -- and TAH.AX is excluded in
    5 of the 15 public questions. Read the `ticker` field.
    """
    path = fixture_dataset / "ASX" / "Tabcorp-ASX-2015-2021.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert rows[0]["ticker"] == "TAH.AX"
    assert not path.stem.upper().startswith("TAH")


def test_afr_fixture_publicationdate_is_a_yyyymmdd_string(fixture_dataset):
    path = fixture_dataset / "AFR" / "AFR_fixture.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert set(rows[0]) >= {
        "HEADLINE",
        "SUBHEAD",
        "INTRO",
        "TEXT",
        "NEWSPAPER",
        "PUBLICATIONDATE",
    }
    d = rows[0]["PUBLICATIONDATE"]
    assert isinstance(d, str) and len(d) == 8 and d.isdigit()


def test_afr_fixture_contains_the_apostrophe_tokenizer_trap(fixture_dataset):
    """[a-z0-9']+ gives 6,903 for \\bnab\\b where the correct answer is 7,372,
    because \\b treats an apostrophe as a boundary. The fixture must contain a
    nab-with-apostrophe article so this stays a regression test, not a note."""
    path = fixture_dataset / "AFR" / "AFR_fixture.jsonl"
    blobs = []
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        blobs.append(
            " ".join(
                str(r.get(k) or "")
                for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT")
            ).lower()
        )
    corpus = "\n".join(blobs)
    assert re.search(r"nab['’]", corpus), "need an apostrophe-adjacent 'nab'"

    # The equivalence the whole AFR index rests on.
    good = sum(1 for b in blobs if "nab" in re.findall(r"[a-z0-9]+", b))
    regex = sum(1 for b in blobs if re.search(r"\bnab\b", b))
    assert good == regex, "[a-z0-9]+ tokens must equal \\bnab\\b counts"
