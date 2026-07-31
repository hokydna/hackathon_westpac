"""AFR index tests. Session A, harness §10 step 5.

The search rule is the one `Setup_Instructions.md` calls non-negotiable for
reproducibility: case-insensitive, `\\bword\\b` anchored, over
HEADLINE + SUBHEAD + INTRO + TEXT **combined**, counted **once per record**.

Measured on the real corpus 2026-07-31 — every reference value below was verified
before the assertion was written:

    219,538 records, 85 files, 373,012 distinct tokens
    index build 25.1s, queries 0.03ms for three terms
    peak RSS 905 MB
    unemployment 5,997 | qbe 1,546 | nab 7,372   (identical to a regex scan)
    MHQ061  peak year 2020 = 1,452, peak month 202005 = 218
    MHQ076  QBE in 2021 = 369
    coverage 2015-2021

The real-corpus index is session-scoped: 25s once, not 25s per test.
"""

from __future__ import annotations

import re

import pytest

from src.tools import afr, corpora

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def real_index(request):
    """The full 219,538-record index. Built once for the whole session."""
    from tests.conftest import REAL_DATASET

    if not (REAL_DATASET / "AFR").is_dir():
        pytest.skip("full `data set/` not present -- it is untracked by design")
    return corpora.build_afr_index(REAL_DATASET / "AFR")


@pytest.fixture
def fixture_index(fixture_dataset):
    return corpora.build_afr_index(fixture_dataset / "AFR")


class TestTokenizer:
    """The equivalence the whole index rests on."""

    def test_apostrophe_is_not_a_token_character(self):
        """`[a-z0-9']+` gives 6,903 for nab where 7,372 is correct, because `\\b`
        treats an apostrophe as a boundary. So `nab's` must yield `nab`."""
        assert "nab" in corpora.tokenize("NAB's results")
        assert "nab's" not in corpora.tokenize("NAB's results")

    def test_is_case_insensitive(self):
        assert corpora.tokenize("QBE Qbe qbe") == {"qbe"}

    def test_digits_are_tokens(self):
        assert "2021" in corpora.tokenize("in 2021 the rate")

    def test_punctuation_splits_tokens(self):
        assert corpora.tokenize("rate-sensitive, shares.") == {"rate", "sensitive", "shares"}

    def test_index_equals_regex_on_the_fixture_corpus(self, fixture_dataset):
        """Generalises the three spot-checks: for every term tried, the index
        count must equal a `\\bword\\b` scan over the combined fields."""
        import json

        path = fixture_dataset / "AFR" / "AFR_fixture.jsonl"
        blobs = []
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            blobs.append(" ".join(str(r.get(k) or "") for k in corpora.AFR_FIELDS))

        idx = corpora.build_afr_index(fixture_dataset / "AFR")
        for term in ("nab", "qbe", "unemployment", "bank", "rate", "the"):
            regex = sum(1 for b in blobs if re.search(rf"\b{term}\b", b, re.I))
            assert afr.count(term, index=idx)["matching_records"] == regex, term


class TestCountOncePerRecord:
    def test_a_term_repeated_in_one_record_counts_once(self, fixture_dataset):
        """"Counted once per record" is the rule. Term frequency is irrelevant."""
        idx = corpora.build_afr_index(fixture_dataset / "AFR")
        ids = idx.record_ids("the")
        assert len(ids) == len(set(ids))

    def test_unknown_term_returns_zero_not_an_exception(self, fixture_index):
        r = afr.count("zzzzznotaword", index=fixture_index)
        assert r["matching_records"] == 0

    def test_term_is_normalised_before_lookup(self, fixture_index):
        assert (
            afr.count("  NAB  ", index=fixture_index)["matching_records"]
            == afr.count("nab", index=fixture_index)["matching_records"]
        )


class TestReferenceCounts:
    """The three verified constants. Do not re-derive — use as fixtures."""

    @pytest.mark.needs_dataset
    @pytest.mark.parametrize(
        "term,expected", [("unemployment", 5997), ("qbe", 1546), ("nab", 7372)]
    )
    def test_reference_term_counts(self, real_index, term, expected):
        assert afr.count(term, index=real_index)["matching_records"] == expected

    @pytest.mark.needs_dataset
    def test_corpus_shape(self, real_index):
        assert real_index.total == 219538


class TestMHQ061:
    """Two graded components in one question, so both ship together."""

    @pytest.mark.needs_dataset
    def test_peak_year_and_peak_month(self, real_index):
        r = afr.count_by_month("unemployment", index=real_index)
        assert r["peak_year"] == "2020"
        assert r["peak_year_count"] == 1452
        assert r["peak_month"] == "202005"
        assert r["peak_month_count"] == 218

    def test_empty_result_returns_a_note(self, fixture_index):
        r = afr.count_by_month("zzzznope", index=fixture_index)
        assert r["matching_records"] == 0
        assert "note" in r


class TestMHQ076:
    @pytest.mark.needs_dataset
    def test_qbe_2021_count(self, real_index):
        """One 10-point all-or-nothing component; the year filter is load-bearing."""
        assert afr.count("QBE", year=2021, index=real_index)["matching_records"] == 369

    @pytest.mark.needs_dataset
    def test_year_filter_narrows_the_result(self, real_index):
        whole = afr.count("qbe", index=real_index)["matching_records"]
        year = afr.count("qbe", year=2021, index=real_index)["matching_records"]
        assert 0 < year < whole


class TestShare:
    @pytest.mark.needs_dataset
    def test_share_of_corpus_is_a_percentage(self, real_index):
        r = afr.share("unemployment", index=real_index)
        assert r["matching_records"] == 5997
        assert r["scope_records"] == 219538
        assert r["share_pct"] == pytest.approx(100 * 5997 / 219538, abs=1e-4)

    def test_empty_scope_returns_a_note(self, fixture_index):
        assert "note" in afr.share("nab", year=1999, index=fixture_index)


class TestRetrieveByHeadline:
    """★ The entry point for all three sentiment questions."""

    def test_finds_a_record_by_partial_headline(self, fixture_index, fixture_dataset):
        import json

        first = json.loads(
            (fixture_dataset / "AFR" / "AFR_fixture.jsonl").read_text().splitlines()[0]
        )
        headline = first["HEADLINE"]
        r = afr.retrieve_by_headline(headline, index=fixture_index)
        assert r["matches"]
        assert r["matches"][0]["headline"] == headline

    def test_ranks_by_token_overlap(self, fixture_index):
        r = afr.retrieve_by_headline("bank shares rate", limit=3, index=fixture_index)
        scores = [m["matched_tokens"] for m in r["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_a_note_not_an_exception(self, fixture_index):
        r = afr.retrieve_by_headline("zzzznope qqqqnope", index=fixture_index)
        assert r["matches"] == []
        assert "note" in r

    def test_empty_query_is_handled(self, fixture_index):
        assert afr.retrieve_by_headline("   ", index=fixture_index)["matches"] == []


class TestCoverage:
    @pytest.mark.needs_dataset
    def test_afr_stops_in_2021(self, real_index):
        """The basis for MHQ090's justified refusal: RBA runs past the 2022-23
        hikes but AFR ends in 2021, so a post-2021 join is unsupported."""
        r = afr.coverage(index=real_index)
        assert r["first_year"] == "2015"
        assert r["last_year"] == "2021"

    @pytest.mark.needs_dataset
    def test_some_records_are_undated(self, real_index):
        """92 records carry no usable PUBLICATIONDATE, so year aggregation must
        skip them rather than assume a year."""
        dated = sum(1 for y in real_index.years if y)
        assert dated < real_index.total
        assert real_index.total - dated == 92


class TestDateHandling:
    def test_publicationdate_is_sliced_not_parsed(self, fixture_index):
        """It is a YYYYMMDD *string*. Date-parsing it is both slower and wrong on
        the undated records."""
        for y in fixture_index.years:
            assert y == "" or (len(y) == 4 and y.isdigit())
        for m in fixture_index.months:
            assert m == "" or (len(m) == 6 and m.isdigit())
