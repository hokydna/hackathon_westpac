"""RBA tool tests. Session A, harness §10 step 3.

Every assertion here is a measured value from the real corpus or a published
reference answer — not an invention. The §8 constants exist so implementers do
not re-derive them, and MHQ001 is the blocking gate before anything downstream
matters.

Unit tests run against `tests/fixtures/dataset/` (10 rows, BOM preserved, signed
string changes). The `needs_dataset` tests run against the real 175-row corpus and
skip automatically when it is absent, so the default run stays green in a worktree
and in a judge's fresh clone.
"""

from __future__ import annotations

import pytest

from src.tools import corpora, rba

pytestmark = pytest.mark.unit


class TestCorpusLoading:
    def test_bom_is_handled(self, fixture_dataset):
        """RBA-rates.jsonl is UTF-8 BOM and needs `encoding="utf-8-sig"`.

        The failure is LOUD, not silent: read as plain utf-8, the leading
        `\\ufeff` reaches json.loads, which raises
        `JSONDecodeError: Unexpected UTF-8 BOM`. Worth pinning both halves — the
        trap and our immunity — so this stays a regression test if anyone drops
        the encoding argument.
        """
        import json

        path = fixture_dataset / "RBA Rates" / "RBA-rates.jsonl"
        assert path.read_bytes()[:3] == b"\xef\xbb\xbf", "fixture must keep the BOM"

        with open(path, encoding="utf-8") as fh:
            first_line = fh.readline()
        with pytest.raises(json.JSONDecodeError, match="BOM"):
            json.loads(first_line)

        # Our loader is immune.
        rows = corpora.load_rba(path)
        assert rows and rows[0].rate > 0

    def test_dates_parse_from_the_non_iso_format(self, fixture_dataset):
        """Dates are '3 Feb 2010', not ISO. Naive date-parsing raises."""
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        assert rows[0].date.year == 2010
        assert rows[0].date.month == 2

    def test_rows_are_sorted_ascending_by_date(self, fixture_dataset):
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        assert [r.date for r in rows] == sorted(r.date for r in rows)

    def test_signed_string_changes_become_floats(self, fixture_dataset):
        """Values are strings including '+0.25', '-0.25', '0.00'."""
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        assert all(isinstance(r.change, float) for r in rows)
        assert any(r.change > 0 for r in rows)
        assert any(r.change < 0 for r in rows)

    def test_rate_becomes_float(self, fixture_dataset):
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        assert all(isinstance(r.rate, float) for r in rows)


class TestCountChanges:
    """MHQ001 — the blocking gate.

    A single 10-point all-or-nothing component bundling three numbers. Getting
    two of three right scores ZERO, so this is tested against the real corpus.
    """

    @pytest.fixture
    def real_rows(self, real_dataset):
        """The full 175-row corpus. conftest points DATASET_DIR at the fixtures
        globally, so needs_dataset tests must ask for the real thing by name."""
        return corpora.load_rba(real_dataset / "RBA Rates" / "RBA-rates.jsonl")

    @pytest.mark.needs_dataset
    def test_mhq001_reference_reproduces_exactly(self, real_rows):
        result = rba.count_changes(rows=real_rows)
        assert result["total_records"] == 175
        assert result["changed"] == 41
        assert result["increases"] == 20
        assert result["decreases"] == 21

    @pytest.mark.needs_dataset
    def test_increases_plus_decreases_equals_changed(self, real_rows):
        r = rba.count_changes(rows=real_rows)
        assert r["increases"] + r["decreases"] == r["changed"]

    def test_zero_change_rows_are_not_counted_as_changes(self, fixture_dataset):
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        r = rba.count_changes(rows=rows)
        holds = sum(1 for x in rows if x.change == 0.0)
        assert r["changed"] == len(rows) - holds

    def test_returns_structured_data_for_synthesis(self, fixture_dataset):
        """The adapter is trained on typed evidence, not strings, so every tool
        must expose a dict as well as its human-readable return."""
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        r = rba.count_changes(rows=rows)
        assert isinstance(r, dict)
        assert all(isinstance(v, (int, float, str)) for v in r.values())


class TestLookupRateAsOf:
    """As-of semantics: the rate in force ON or BEFORE a date.

    Nearest-match is wrong — it can return a FUTURE decision, which silently
    answers a different question. `handout/03` calls this out explicitly.
    """

    @pytest.fixture
    def rows(self, fixture_dataset):
        return corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")

    def test_exact_match_returns_that_decision(self, rows):
        target = rows[2]
        assert rba.lookup_rate(target.date.isoformat(), rows=rows)["rate"] == target.rate

    def test_a_date_between_decisions_returns_the_earlier_one(self, rows):
        from datetime import timedelta

        mid = rows[1].date + timedelta(days=1)
        assert mid < rows[2].date
        assert rba.lookup_rate(mid.isoformat(), rows=rows)["rate"] == rows[1].rate

    def test_never_returns_a_future_decision(self, rows):
        from datetime import timedelta

        mid = rows[1].date + timedelta(days=1)
        got = rba.lookup_rate(mid.isoformat(), rows=rows)
        assert got["effective_date"] <= mid.isoformat()

    def test_a_date_after_the_last_decision_returns_the_last(self, rows):
        assert rba.lookup_rate("2099-01-01", rows=rows)["rate"] == rows[-1].rate

    def test_a_date_before_the_first_returns_no_data_not_an_exception(self, rows):
        """§7: a tool never raises on an empty result set — it returns a valid
        'no results' payload so the brain can try another query."""
        got = rba.lookup_rate("1990-01-01", rows=rows)
        assert got.get("rate") is None
        assert "no" in str(got.get("note", "")).lower()

    def test_accepts_the_corpus_date_format_too(self, rows):
        """Qwen may echo '3 Feb 2010' straight back from a tool result."""
        native = rows[0].date.strftime("%-d %b %Y") if hasattr(rows[0].date, "strftime") else None
        if native:
            assert rba.lookup_rate(native, rows=rows)["rate"] == rows[0].rate

    def test_unparseable_date_returns_a_note_not_an_exception(self, rows):
        got = rba.lookup_rate("not-a-date", rows=rows)
        assert got.get("rate") is None


class TestExtremes:
    @pytest.fixture
    def real_rows(self, real_dataset):
        return corpora.load_rba(real_dataset / "RBA Rates" / "RBA-rates.jsonl")

    @pytest.mark.needs_dataset
    def test_lowest_rate_matches_the_published_example(self, real_rows):
        """Challenge_Brief.md's full-marks easy example: the lowest cash-rate
        target is 0.1, first effective 2020-11-04, shown by 16 records.

        Note this one the brief ACCEPTS, and we reproduce it exactly — which is
        what makes the 4.75 date discrepancy below worth flagging rather than
        assuming our dates are systematically off by a day.
        """
        r = rba.extremes(rows=real_rows)
        assert r["min_rate"] == 0.1
        assert r["min_rate_first_date"] == "2020-11-04"
        assert r["min_rate_record_count"] == 16

    @pytest.mark.needs_dataset
    def test_highest_rate_and_its_record_count(self, real_rows):
        """4.75, 11 records, first effective 2010-11-03 — measured from the corpus.

        ⚠️ KNOWN DISCREPANCY, flagged not papered over. Challenge_Brief.md's
        partial-credit example marks 2010-11-03 WRONG and says "the judge expected
        2010-11-02". **2 Nov 2010 does not exist anywhere in the approved
        dataset** — the only 4.75 effective date is 3 Nov 2010, though the brief's
        record count of 11 matches us exactly.

        The RBA board meets on a Tuesday and the rate takes effect Wednesday, so
        that example looks graded against the ANNOUNCEMENT date while the corpus
        carries EFFECTIVE dates. Note the brief's other example is consistent with
        us: 0.1 first took effect 2020-11-04, which it accepts and we reproduce.

        We assert the corpus, because § Technical Reference says scores come from
        "running the same tool calls against the same data", and shifting dates a
        day earlier would break the accepted 2020-11-04 case. If hidden date
        components grade against announcement dates this costs points — it is a
        question for the organizers, per Setup_Instructions.md's closing advice.
        """
        r = rba.extremes(rows=real_rows)
        assert r["max_rate"] == 4.75
        assert r["max_rate_record_count"] == 11
        assert r["max_rate_first_date"] == "2010-11-03"


class TestMaxHoldStreak:
    @pytest.fixture
    def real_rows(self, real_dataset):
        return corpora.load_rba(real_dataset / "RBA Rates" / "RBA-rates.jsonl")

    @pytest.mark.needs_dataset
    def test_longest_gap_between_non_zero_changes(self, real_rows):
        """The brief's full-marks medium example: 1036 days, 2016-08-03 to
        2019-06-05, held at 1.5 then changed to 1.25."""
        r = rba.max_hold_streak(rows=real_rows)
        assert r["days"] == 1036
        assert r["start_date"] == "2016-08-03"
        assert r["end_date"] == "2019-06-05"
        assert r["rate_during_hold"] == 1.5
        assert r["rate_after"] == 1.25


class TestCycleSummary:
    """★ Required by the question bank, absent from the execution guide's metric
    list. MHQ035 and MHQ084 need it."""

    @pytest.fixture
    def real_rows(self, real_dataset):
        return corpora.load_rba(real_dataset / "RBA Rates" / "RBA-rates.jsonl")

    @pytest.mark.needs_dataset
    def test_2022_2023_tightening_cycle_matches_the_brief(self, real_rows):
        """The brief's full-marks hard example: 13 hikes from 4 May 2022 to
        8 Nov 2023, cumulative +4.25pp, 0.1 before the first hike, 4.35 final."""
        r = rba.cycle_summary("2022-05-04", "2023-11-08", rows=real_rows)
        assert r["hikes"] == 13
        assert r["cumulative_change_pp"] == pytest.approx(4.25, abs=0.001)
        assert r["rate_before_first"] == 0.1
        assert r["rate_final"] == 4.35

    @pytest.mark.needs_dataset
    def test_2011_2013_easing_cycle_matches_mhq035(self, real_rows):
        """MHQ035's reference: eight cuts (2 in 2011, 4 in 2012, 2 in 2013),
        -2.25pp, 4.75 before the first cut to 2.50 at the end."""
        r = rba.cycle_summary("2011-01-01", "2013-12-31", rows=real_rows)
        assert r["cuts"] == 8
        assert r["cumulative_change_pp"] == pytest.approx(-2.25, abs=0.001)
        assert r["rate_before_first"] == 4.75
        assert r["rate_final"] == 2.50

    def test_an_empty_range_returns_no_data_not_an_exception(self, fixture_dataset):
        rows = corpora.load_rba(fixture_dataset / "RBA Rates" / "RBA-rates.jsonl")
        r = rba.cycle_summary("1990-01-01", "1990-12-31", rows=rows)
        assert r.get("decisions") == 0
