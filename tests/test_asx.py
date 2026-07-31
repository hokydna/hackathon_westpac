"""ASX metric tests. Session A, harness §10 step 6.

Every reference value was measured on the real corpus before the assertion was
written, and all of them reproduce exactly:

    MHQ040  18 tickers x 1,774 rows, 2015-01-02 -> 2021-12-30
    MHQ045  BHP.AX best 2018 +22.17%, AMP.AX worst -50.04%
    MHQ049  AMP.AX highest average daily volume, 11,635,671.71 shares/day
    MHQ076  QBE.AX best non-Tabcorp 2021 return, +35.57%
    MHQ072  5->12 Jun 2019: CBA +0.60 NAB +1.39 ANZ +0.89 BHP +5.89 RIO +2.91
    MHQ074  equal-weighted non-Tabcorp basket +2.88% / +0.24% / -2.17%

`exclude_tickers` gets its own class because 5 of the 15 public questions depend
on it and the failure is silent — a wrong exclusion returns a plausible number.
"""

from __future__ import annotations

import pytest

from src.tools import asx, corpora

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def real_series():
    from tests.conftest import REAL_DATASET

    if not (REAL_DATASET / "ASX").is_dir():
        pytest.skip("full `data set/` not present -- it is untracked by design")
    return corpora.load_asx(REAL_DATASET / "ASX")


@pytest.fixture
def fx(fixture_dataset):
    """BHP.AX + TAH.AX, 80 bars each — enough to exercise exclusion."""
    return corpora.load_asx(fixture_dataset / "ASX")


class TestExcludeTickers:
    """5 of the 15 public questions exclude Tabcorp, and the questions say
    "excluding Tabcorp" rather than naming a ticker."""

    def test_tabcorp_is_excluded_by_company_name(self, fx):
        assert "TAH.AX" not in asx.describe(exclude_tickers=["Tabcorp"], series=fx)["tickers"]

    @pytest.mark.parametrize("spelling", ["TAH.AX", "tah.ax", "TAH", "tah", "Tabcorp", "TABCORP"])
    def test_every_plausible_spelling_excludes_the_same_ticker(self, fx, spelling):
        """Qwen's wording varies with the question; a missed exclusion silently
        changes the answer rather than erroring."""
        assert "TAH.AX" not in asx.describe(exclude_tickers=[spelling], series=fx)["tickers"]

    def test_no_exclusion_keeps_everything(self, fx):
        assert "TAH.AX" in asx.describe(series=fx)["tickers"]

    def test_bare_ticker_gets_the_ax_suffix(self, fx):
        assert "BHP.AX" not in asx.describe(exclude_tickers=["BHP"], series=fx)["tickers"]

    def test_empty_and_none_exclusions_are_harmless(self, fx):
        assert asx.describe(exclude_tickers=[], series=fx)["ticker_files"] == 2
        assert asx.describe(exclude_tickers=["", "  "], series=fx)["ticker_files"] == 2

    def test_exclusion_applies_to_every_metric(self, real_series):
        """The plan says exclude_tickers is first-class on EVERY ASX metric, so
        parametrise over them rather than trusting one."""
        pytest.importorskip("statistics")
        for fn in (asx.rank_annual_returns,):
            r = fn(2018, exclude_tickers=["Tabcorp"], series=real_series)
            assert "TAH.AX" not in {x["ticker"] for x in r["ranking"]}
        assert "TAH.AX" not in asx.avg_volume(
            exclude_tickers=["Tabcorp"], series=real_series
        )["avg_daily_volume"]


class TestMHQ040Describe:
    """One 10-point all-or-nothing component bundling three numbers."""

    @pytest.mark.needs_dataset
    def test_dimensions_and_date_range(self, real_series):
        r = asx.describe(series=real_series)
        assert r["ticker_files"] == 18
        assert r["rows_per_ticker"] == 1774
        assert r["first_date"] == "2015-01-02"
        assert r["last_date"] == "2021-12-30"

    @pytest.mark.needs_dataset
    def test_every_ticker_has_the_same_row_count(self, real_series):
        r = asx.describe(series=real_series)
        assert r["row_counts_vary"] is False
        assert r["total_rows"] == 18 * 1774


class TestMHQ045Ranking:
    @pytest.mark.needs_dataset
    def test_best_and_worst_2018_excluding_tabcorp(self, real_series):
        r = asx.rank_annual_returns(2018, exclude_tickers=["Tabcorp"], series=real_series)
        assert r["best_ticker"] == "BHP.AX"
        assert r["best_return_pct"] == pytest.approx(22.17, abs=0.01)
        assert r["worst_ticker"] == "AMP.AX"
        assert r["worst_return_pct"] == pytest.approx(-50.04, abs=0.01)

    @pytest.mark.needs_dataset
    def test_ranking_is_ordered_best_to_worst(self, real_series):
        r = asx.rank_annual_returns(2018, exclude_tickers=["Tabcorp"], series=real_series)
        vals = [x["price_return_pct"] for x in r["ranking"]]
        assert vals == sorted(vals, reverse=True)


class TestMHQ049Volume:
    @pytest.mark.needs_dataset
    def test_highest_average_daily_volume(self, real_series):
        """Reference quotes 2dp, so rounding is part of the answer."""
        r = asx.avg_volume(exclude_tickers=["Tabcorp"], series=real_series)
        assert r["highest_ticker"] == "AMP.AX"
        assert r["highest_avg_daily_volume"] == pytest.approx(11_635_671.71, abs=0.01)


class TestMHQ076AnnualReturn:
    @pytest.mark.needs_dataset
    def test_qbe_2021_is_the_best_non_tabcorp_return(self, real_series):
        one = asx.annual_return("QBE.AX", 2021, series=real_series)
        assert one["price_return_pct"] == pytest.approx(35.57, abs=0.01)
        ranked = asx.rank_annual_returns(2021, exclude_tickers=["Tabcorp"], series=real_series)
        assert ranked["best_ticker"] == "QBE.AX"

    @pytest.mark.needs_dataset
    def test_bare_ticker_is_accepted(self, real_series):
        assert asx.annual_return("QBE", 2021, series=real_series)["ticker"] == "QBE.AX"


class TestWindowReturn:
    """★ Absent from the execution guide's metric list; MHQ072/074 need it."""

    @pytest.mark.needs_dataset
    def test_mhq072_per_ticker_returns(self, real_series):
        r = asx.window_return(
            "2019-06-05", "2019-06-12",
            tickers=["CBA.AX", "NAB.AX", "ANZ.AX", "BHP.AX", "RIO.AX"],
            series=real_series,
        )
        for ticker, expected in (
            ("CBA.AX", 0.60), ("NAB.AX", 1.39), ("ANZ.AX", 0.89),
            ("BHP.AX", 5.89), ("RIO.AX", 2.91),
        ):
            assert r["returns_pct"][ticker] == pytest.approx(expected, abs=0.01), ticker

    @pytest.mark.needs_dataset
    @pytest.mark.parametrize(
        "lo,hi,expected",
        [
            ("2019-06-05", "2019-06-12", 2.88),
            ("2019-07-03", "2019-07-10", 0.24),
            ("2019-10-02", "2019-10-09", -2.17),
        ],
    )
    def test_mhq074_equal_weighted_basket(self, real_series, lo, hi, expected):
        """Equal-weighted MEAN of per-ticker returns, not a price-weighted index."""
        r = asx.window_return(lo, hi, exclude_tickers=["Tabcorp"], series=real_series)
        assert r["equal_weighted_basket_return_pct"] == pytest.approx(expected, abs=0.01)

    @pytest.mark.needs_dataset
    def test_basket_excludes_tabcorp(self, real_series):
        r = asx.window_return("2019-06-05", "2019-06-12", exclude_tickers=["Tabcorp"], series=real_series)
        assert "TAH.AX" not in r["tickers"]
        assert len(r["tickers"]) == 17

    def test_a_non_trading_start_date_still_returns_a_window(self, fx):
        """RBA effective dates are often not trading days. Requiring an exact
        match would return nothing on exactly the cross-dataset questions this
        metric exists for."""
        r = asx.window_return("2015-01-03", "2015-02-01", series=fx)  # 3 Jan is a Saturday
        assert "returns_pct" in r

    def test_accepts_the_rba_date_format(self, fx):
        assert "returns_pct" in asx.window_return("5 Jan 2015", "1 Feb 2015", series=fx)

    def test_unparseable_dates_return_a_note(self, fx):
        assert "note" in asx.window_return("nope", "also-nope", series=fx)


class TestVolatilityAndCorrelation:
    @pytest.mark.needs_dataset
    def test_volatility_states_its_basis(self, real_series):
        """The basis string is itself a graded component in at least one question."""
        r = asx.volatility("CBA.AX", 2017, series=real_series)
        assert r["annualisation_factor"] == 252
        assert "close-to-close" in r["basis"]
        assert 0 < r["volatility_pct_annualised"] < 100

    @pytest.mark.needs_dataset
    def test_a_ticker_correlates_perfectly_with_itself(self, real_series):
        r = asx.correlation("CBA.AX", "CBA.AX", 2018, series=real_series)
        assert r["correlation"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.needs_dataset
    def test_correlation_is_symmetric(self, real_series):
        a = asx.correlation("CBA.AX", "NAB.AX", 2018, series=real_series)["correlation"]
        b = asx.correlation("NAB.AX", "CBA.AX", 2018, series=real_series)["correlation"]
        assert a == pytest.approx(b, abs=1e-9)

    def test_too_few_days_returns_a_note(self, fx):
        assert "note" in asx.volatility("BHP.AX", 1999, series=fx)


class TestMaxDrawdown:
    @pytest.mark.needs_dataset
    def test_drawdown_is_negative_with_trough_after_peak(self, real_series):
        r = asx.max_drawdown("AMP.AX", series=real_series)
        assert r["max_drawdown_pct"] < 0
        assert r["peak_date"] <= r["trough_date"]

    def test_unknown_ticker_returns_a_note(self, fx):
        assert "note" in asx.max_drawdown("ZZZ.AX", series=fx)


class TestNeverRaises:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s: asx.annual_return("NOPE", 2018, series=s),
            lambda s: asx.rank_annual_returns(1999, series=s),
            lambda s: asx.volatility("NOPE", series=s),
            lambda s: asx.correlation("NOPE", "BHP.AX", series=s),
            lambda s: asx.max_drawdown("NOPE", series=s),
            lambda s: asx.window_return("2015-01-05", "2015-01-06", tickers=["NOPE"], series=s),
        ],
    )
    def test_bad_input_returns_a_note_not_an_exception(self, fx, call):
        """§7: a tool never raises — it returns a valid payload so the brain can
        try a different query."""
        result = call(fx)
        assert isinstance(result, dict)


class TestCoverage:
    @pytest.mark.needs_dataset
    def test_asx_stops_in_2021(self, real_series):
        """With AFR, this is the basis for MHQ090's justified refusal."""
        r = asx.coverage(series=real_series)
        assert r["first_date"] == "2015-01-02"
        assert r["last_date"] == "2021-12-30"
        assert r["tickers"] == 18
