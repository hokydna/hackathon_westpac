"""Blocking correctness gate.

The 15 public questions are **fixtures, never training rows**. If the metric layer cannot
reproduce the organizer's own reference answers, the training data is wrong and nothing
downstream matters — so this fails loudly and the generator refuses to emit.

Twelve of the fifteen are fully deterministic and checked here. The three sentiment
questions (MHQ058 / MHQ067 / MHQ080) are checked only on their deterministic components —
the as-of RBA rate and, for MHQ080, the basket window return. Their sentiment and
direction clauses are judgements with no ground truth in the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import metrics as M
from .corpora import TABCORP, load_afr, load_asx, load_rba


@dataclass
class Check:
    qid: str
    what: str
    expected: object
    actual: object
    ok: bool


def _close(a: float, b: float, tol: float) -> bool:
    return a is not None and abs(a - b) <= tol


def run_gate() -> list[Check]:
    rba, asx, afr = load_rba(), load_asx(), load_afr()
    checks: list[Check] = []

    def add(qid: str, what: str, expected, actual, ok: bool) -> None:
        checks.append(Check(qid, what, expected, actual, ok))

    # -- MHQ001 -- 41 of 175 records changed; 20 increases, 21 decreases ----------------
    ev = M.rba_change_counts(rba)
    add("MHQ001", "record_count", 175, ev["record_count"], ev["record_count"] == 175)
    add("MHQ001", "change_count", 41, ev["change_count"], ev["change_count"] == 41)
    add("MHQ001", "increase_count", 20, ev["increase_count"], ev["increase_count"] == 20)
    add("MHQ001", "decrease_count", 21, ev["decrease_count"], ev["decrease_count"] == 21)

    # -- MHQ035 -- 2011-2013 easing: 8 cuts (2/4/2), -2.25 pp, 4.75% -> 2.50% ----------
    ev = M.rba_cycle_summary(rba, date(2011, 1, 1), date(2013, 12, 31), "easing")
    add("MHQ035", "move_count", 8, ev["move_count"], ev["move_count"] == 8)
    add("MHQ035", "moves_per_year", {"2011": 2, "2012": 4, "2013": 2},
        ev["moves_per_year"], ev["moves_per_year"] == {"2011": 2, "2012": 4, "2013": 2})
    add("MHQ035", "cumulative_change_pp", -2.25, ev["cumulative_change_pp"],
        _close(ev["cumulative_change_pp"], -2.25, 1e-9))
    add("MHQ035", "start_rate_pct", 4.75, ev["start_rate_pct"],
        _close(ev["start_rate_pct"], 4.75, 1e-9))
    add("MHQ035", "end_rate_pct", 2.50, ev["end_rate_pct"],
        _close(ev["end_rate_pct"], 2.50, 1e-9))

    # -- MHQ040 -- 18 files, 1,774 rows each, 2 Jan 2015 -> 30 Dec 2021 ----------------
    ev = M.asx_describe(asx)
    add("MHQ040", "ticker_file_count", 18, ev["ticker_file_count"], ev["ticker_file_count"] == 18)
    add("MHQ040", "rows_per_file", 1774, ev["rows_per_file"], ev["rows_per_file"] == 1774)
    add("MHQ040", "first_trade_date", "2015-01-02", ev["first_trade_date"],
        ev["first_trade_date"] == "2015-01-02")
    add("MHQ040", "last_trade_date", "2021-12-30", ev["last_trade_date"],
        ev["last_trade_date"] == "2021-12-30")

    # -- MHQ045 -- 2018: BHP best +22.17%, AMP worst -50.04% ---------------------------
    ev = M.asx_rank_annual_returns(asx, 2018, exclude=(TABCORP,))
    add("MHQ045", "best_ticker", "BHP.AX", ev["best_ticker"], ev["best_ticker"] == "BHP.AX")
    add("MHQ045", "best_price_return_pct", 22.17, round(ev["best_price_return_pct"], 2),
        _close(ev["best_price_return_pct"], 22.17, 0.02))
    add("MHQ045", "worst_ticker", "AMP.AX", ev["worst_ticker"], ev["worst_ticker"] == "AMP.AX")
    add("MHQ045", "worst_price_return_pct", -50.04, round(ev["worst_price_return_pct"], 2),
        _close(ev["worst_price_return_pct"], -50.04, 0.02))

    # -- MHQ049 -- AMP highest average daily volume, 11,635,671.71 ---------------------
    ev = M.asx_rank_avg_volume(asx, exclude=(TABCORP,))
    add("MHQ049", "highest_ticker", "AMP.AX", ev["highest_ticker"],
        ev["highest_ticker"] == "AMP.AX")
    add("MHQ049", "highest_avg_daily_volume", 11635671.71,
        round(ev["highest_avg_daily_volume"], 2),
        _close(ev["highest_avg_daily_volume"], 11635671.71, 1.0))

    # -- MHQ055 -- three worst drawdowns with peak/trough dates ------------------------
    ev = M.asx_rank_max_drawdown(asx, exclude=(TABCORP,), top=3)
    want = [
        ("AMP.AX", -82.45, "2015-03-20", "2021-12-17"),
        ("AGL.AX", -76.24, "2017-04-10", "2021-11-16"),
        ("QAN.AX", -71.08, "2019-12-19", "2020-03-19"),
    ]
    for i, (tk, dd, pk, tr) in enumerate(want):
        row = ev["ranked_max_drawdowns"][i]
        add("MHQ055", f"rank{i+1}",
            f"{tk} {dd} {pk}->{tr}",
            f"{row['ticker']} {round(row['max_drawdown_pct'],2)} {row['peak_date']}->{row['trough_date']}",
            row["ticker"] == tk
            and _close(row["max_drawdown_pct"], dd, 0.02)
            and row["peak_date"] == pk
            and row["trough_date"] == tr)

    # -- MHQ061 -- unemployment peaks: 2020 / 1,452 and May 2020 / 218 -----------------
    ev = M.afr_peak_year_and_month(afr, "unemployment")
    add("MHQ061", "peak_year", 2020, ev["peak_year"], ev["peak_year"] == 2020)
    add("MHQ061", "peak_year_count", 1452, ev["peak_year_count"], ev["peak_year_count"] == 1452)
    add("MHQ061", "peak_month", "202005", ev["peak_month"], ev["peak_month"] == "202005")
    add("MHQ061", "peak_month_count", 218, ev["peak_month_count"], ev["peak_month_count"] == 218)

    # -- MHQ058 / MHQ067 / MHQ080 -- deterministic component only: as-of RBA rate ------
    for qid, when in (
        ("MHQ058", date(2021, 2, 23)),
        ("MHQ067", date(2021, 11, 25)),
        ("MHQ080", date(2020, 11, 28)),
    ):
        ev = M.rba_lookup_rate(rba, when)
        add(qid, "cash_rate_target_pct (as-of)", 0.10, ev["cash_rate_target_pct"],
            _close(ev["cash_rate_target_pct"], 0.10, 1e-9))
    # the applicable decision dates the derivations name explicitly
    for qid, when, eff in (
        ("MHQ058", date(2021, 2, 23), "2021-02-03"),
        ("MHQ067", date(2021, 11, 25), "2021-11-03"),
        ("MHQ080", date(2020, 11, 28), "2020-11-04"),
    ):
        ev = M.rba_lookup_rate(rba, when)
        add(qid, "effective_date (as-of)", eff, ev["effective_date"],
            ev["effective_date"] == eff)

    # -- MHQ072 -- 5 Jun 2019 cut to 1.25%; 5->12 Jun basket +2.88% and 5 constituents -
    ev = M.rba_lookup_rate(rba, date(2019, 6, 5))
    add("MHQ072", "cash_rate_target_pct", 1.25, ev["cash_rate_target_pct"],
        _close(ev["cash_rate_target_pct"], 1.25, 1e-9))
    ev = M.asx_basket_window_return(asx, date(2019, 6, 5), date(2019, 6, 12))
    add("MHQ072", "basket_size", 17, ev["basket_size"], ev["basket_size"] == 17)
    add("MHQ072", "basket_return_pct", 2.88, round(ev["basket_return_pct"], 2),
        _close(ev["basket_return_pct"], 2.88, 0.02))
    for tk, want_pct in (("CBA.AX", 0.60), ("NAB.AX", 1.39), ("ANZ.AX", 0.89),
                         ("BHP.AX", 5.89), ("RIO.AX", 2.91)):
        got = ev["constituent_returns_pct"][tk]
        add("MHQ072", f"{tk} return", want_pct, round(got, 2), _close(got, want_pct, 0.02))

    # -- MHQ074 -- three 2019 cuts, one-week basket returns ----------------------------
    for start, end, want_pct, want_rate in (
        (date(2019, 6, 5), date(2019, 6, 12), 2.88, 1.25),
        (date(2019, 7, 3), date(2019, 7, 10), 0.24, 1.00),
        (date(2019, 10, 2), date(2019, 10, 9), -2.17, 0.75),
    ):
        ev = M.asx_basket_window_return(asx, start, end)
        add("MHQ074", f"basket {start.isoformat()}", want_pct, round(ev["basket_return_pct"], 2),
            _close(ev["basket_return_pct"], want_pct, 0.02))
        rate = M.rba_lookup_rate(rba, start)["cash_rate_target_pct"]
        add("MHQ074", f"rate {start.isoformat()}", want_rate, rate, _close(rate, want_rate, 1e-9))

    # -- MHQ076 -- QBE 2021: 369 AFR records, best non-Tabcorp return +35.57% ----------
    ev = M.afr_term_count(afr, "QBE", year=2021)
    add("MHQ076", "match_count", 369, ev["match_count"], ev["match_count"] == 369)
    ev = M.asx_rank_annual_returns(asx, 2021, exclude=(TABCORP,))
    add("MHQ076", "best_ticker", "QBE.AX", ev["best_ticker"], ev["best_ticker"] == "QBE.AX")
    add("MHQ076", "best_price_return_pct", 35.57, round(ev["best_price_return_pct"], 2),
        _close(ev["best_price_return_pct"], 35.57, 0.02))

    # -- MHQ084 -- 2019: 3 cuts, -0.75 pp, 0.75%; 3,181 AFR; +20.11% average -----------
    ev = M.rba_year_summary(rba, 2019)
    add("MHQ084", "cut_count", 3, ev["cut_count"], ev["cut_count"] == 3)
    add("MHQ084", "cumulative_change_pp", -0.75, ev["cumulative_change_pp"],
        _close(ev["cumulative_change_pp"], -0.75, 1e-9))
    add("MHQ084", "year_end_rate", 0.75, ev["year_end_cash_rate_target_pct"],
        _close(ev["year_end_cash_rate_target_pct"], 0.75, 1e-9))
    ev = M.afr_pattern_count(afr, "rba_rate_pattern", year=2019)
    add("MHQ084", "afr match_count", 3181, ev["match_count"], ev["match_count"] == 3181)
    ev = M.asx_rank_annual_returns(asx, 2019, exclude=(TABCORP,))
    add("MHQ084", "average_price_return_pct", 20.11, round(ev["average_price_return_pct"], 2),
        _close(ev["average_price_return_pct"], 20.11, 0.02))

    # -- MHQ080 -- 30 Nov -> 7 Dec 2020 basket +2.37% ----------------------------------
    ev = M.asx_basket_window_return(asx, date(2020, 11, 30), date(2020, 12, 7))
    add("MHQ080", "basket_return_pct", 2.37, round(ev["basket_return_pct"], 2),
        _close(ev["basket_return_pct"], 2.37, 0.02))

    # -- MHQ090 -- 2022-2023 tightening is RBA-only; ASX and AFR end in 2021 -----------
    ev = M.coverage_check(("RBA", "ASX", "AFR"), date(2022, 5, 1), date(2023, 11, 30))
    add("MHQ090", "unsupported_datasets", ["ASX", "AFR"], ev["unsupported_datasets"],
        ev["unsupported_datasets"] == ["ASX", "AFR"])
    hikes = [r for r in rba.in_range(date(2022, 5, 1), date(2023, 11, 30)) if r.change_pp > 0]
    add("MHQ090", "hike_count 2022-2023", 13, len(hikes), len(hikes) == 13)

    # -- AFR corpus artefact: 92 records carry an empty PUBLICATIONDATE ----------------
    ev = M.afr_describe(afr)
    add("AFR-ARTEFACT", "record_count", 219538, ev["record_count"], ev["record_count"] == 219538)
    add("AFR-ARTEFACT", "undated_record_count", 92, ev["undated_record_count"],
        ev["undated_record_count"] == 92)

    return checks


def assert_gate(verbose: bool = True) -> list[Check]:
    checks = run_gate()
    failed = [c for c in checks if not c.ok]
    if verbose:
        by_q: dict[str, list[Check]] = {}
        for c in checks:
            by_q.setdefault(c.qid, []).append(c)
        for qid, cs in by_q.items():
            bad = [c for c in cs if not c.ok]
            mark = "FAIL" if bad else "ok  "
            print(f"  [{mark}] {qid}: {len(cs) - len(bad)}/{len(cs)} checks")
            for c in bad:
                print(f"         {c.what}: expected {c.expected!r}, got {c.actual!r}")
    if failed:
        raise SystemExit(
            f"CORRECTNESS GATE FAILED: {len(failed)}/{len(checks)} checks. "
            "The metric layer does not reproduce the organizer's reference answers; "
            "refusing to generate training data."
        )
    if verbose:
        print(f"  gate passed: {len(checks)}/{len(checks)} checks against 15 public questions")
    return checks


if __name__ == "__main__":
    assert_gate()
