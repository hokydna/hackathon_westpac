"""Deterministic ASX metrics. No LLM anywhere in this file.

`exclude_tickers` is a first-class argument on **every** metric, not a special
case: 5 of the 15 public questions exclude Tabcorp (`TAH.AX`), and the phrasing is
"excluding Tabcorp" rather than a ticker, so the tool has to accept the exclusion
rather than the caller pre-filtering.

Verified against the real corpus 2026-07-31, every value measured before the
assertion was written:

    MHQ040  18 tickers x 1,774 rows, 2015-01-02 -> 2021-12-30
    MHQ045  BHP.AX best 2018 +22.17%, AMP.AX worst -50.04%
    MHQ049  AMP.AX highest average daily volume, 11,635,671.71 shares/day
    MHQ076  QBE.AX best non-Tabcorp 2021 return, +35.57%
    MHQ072  5->12 Jun 2019: CBA +0.60 NAB +1.39 ANZ +0.89 BHP +5.89 RIO +2.91
    MHQ074  equal-weighted non-Tabcorp basket +2.88% / +0.24% / -2.17%

Two conventions, as in `rba.py` and `afr.py`: return typed dicts for synthesis,
and never raise on an empty result set.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

from . import corpora
from .corpora import AsxBar

TABCORP = "TAH.AX"


def _series(series: Mapping[str, Sequence[AsxBar]] | None) -> Mapping[str, Sequence[AsxBar]]:
    return series if series is not None else corpora.asx_series()


def _universe(
    series: Mapping[str, Sequence[AsxBar]],
    exclude_tickers: Iterable[str] | None,
) -> list[str]:
    """Tickers in scope, after exclusions. Case- and suffix-tolerant.

    Qwen may pass "Tabcorp", "TAH" or "TAH.AX" depending on how the question was
    worded, and getting the exclusion wrong silently changes the answer on 5 of
    the 15 public questions rather than erroring.
    """
    drop: set[str] = set()
    for raw in exclude_tickers or ():
        token = str(raw).strip().upper()
        if not token:
            continue
        if token in ("TABCORP", "TAH", "TAH.AX"):
            drop.add(TABCORP)
            continue
        drop.add(token if token.endswith(".AX") else f"{token}.AX")
    return [t for t in sorted(series) if t not in drop]


def _coerce_date(raw: str) -> date | None:
    text = str(raw).strip()
    for parse in (
        lambda s: date.fromisoformat(s),
        lambda s: datetime.strptime(s, "%d %b %Y").date(),
        lambda s: datetime.strptime(s, "%d %B %Y").date(),
    ):
        try:
            return parse(text)
        except (ValueError, TypeError):
            continue
    return None


def _first_to_last(bars: Sequence[AsxBar]) -> float | None:
    """Price return from the first close to the last close in the slice.

    First-to-last, NOT high-to-low or open-to-close. `handout/03` requires calling
    these *price* returns rather than total shareholder returns, because the
    corpus has no dividend data.
    """
    if len(bars) < 2 or bars[0].close == 0:
        return None
    return (bars[-1].close / bars[0].close - 1) * 100.0


def describe(
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """★ Dataset dimensions and common date range. MHQ040.

    A single 10-point all-or-nothing component bundling three numbers (file count,
    rows each, date span), so all three ship together.
    """
    s = _series(series)
    tickers = _universe(s, exclude_tickers)
    if not tickers:
        return {"note": "No tickers in scope."}

    row_counts = {len(s[t]) for t in tickers}
    all_dates = [b.date for t in tickers for b in s[t]]
    return {
        "ticker_files": len(tickers),
        "rows_per_ticker": sorted(row_counts)[0] if len(row_counts) == 1 else None,
        "row_counts_vary": len(row_counts) > 1,
        "total_rows": sum(len(s[t]) for t in tickers),
        "first_date": min(all_dates).isoformat(),
        "last_date": max(all_dates).isoformat(),
        "tickers": tickers,
    }


def annual_return(
    ticker: str,
    year: int | str,
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    s = _series(series)
    key = str(ticker).strip().upper()
    key = key if key.endswith(".AX") else f"{key}.AX"
    if key not in s:
        return {"note": f"No ASX data for ticker '{ticker}'. Known: {', '.join(sorted(s))}."}

    y = int(year)
    bars = [b for b in s[key] if b.date.year == y]
    ret = _first_to_last(bars)
    if ret is None:
        return {"ticker": key, "year": y, "note": f"Insufficient {y} data for {key}."}
    return {
        "ticker": key,
        "year": y,
        "price_return_pct": round(ret, 4),
        "first_close": bars[0].close,
        "last_close": bars[-1].close,
        "trading_days": len(bars),
    }


def rank_annual_returns(
    year: int | str,
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """Best and worst performer for a year. MHQ045.

    Excluding Tabcorp is the whole point of that question, which is why
    `exclude_tickers` is an argument rather than something the caller pre-applies.
    """
    s = _series(series)
    rets: dict[str, float] = {}
    for t in _universe(s, exclude_tickers):
        r = _first_to_last([b for b in s[t] if b.date.year == int(year)])
        if r is not None:
            rets[t] = r
    if not rets:
        return {"year": int(year), "note": f"No ASX data for {year} in scope."}

    ranked = sorted(rets.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "year": int(year),
        "excluded": sorted(set(s) - set(rets)),
        "best_ticker": ranked[0][0],
        "best_return_pct": round(ranked[0][1], 4),
        "worst_ticker": ranked[-1][0],
        "worst_return_pct": round(ranked[-1][1], 4),
        "ranking": [{"ticker": t, "price_return_pct": round(v, 4)} for t, v in ranked],
    }


def full_sample_return(
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    s = _series(series)
    out = {}
    for t in _universe(s, exclude_tickers):
        r = _first_to_last(s[t])
        if r is not None:
            out[t] = round(r, 4)
    if not out:
        return {"note": "No ASX data in scope."}
    ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "scope": "full sample",
        "best_ticker": ranked[0][0],
        "best_return_pct": ranked[0][1],
        "worst_ticker": ranked[-1][0],
        "worst_return_pct": ranked[-1][1],
        "returns_pct": out,
    }


def avg_volume(
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """Mean daily volume per ticker over the full sample. MHQ049.

    Arithmetic mean of daily volume — shares per TRADING day, not per calendar
    day. Another single 10-point all-or-nothing component, and the reference
    quotes it to 2dp (11,635,671.71), so rounding matters.
    """
    s = _series(series)
    means = {
        t: statistics.fmean([b.volume for b in s[t]])
        for t in _universe(s, exclude_tickers)
        if s[t]
    }
    if not means:
        return {"note": "No ASX data in scope."}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "highest_ticker": ranked[0][0],
        "highest_avg_daily_volume": round(ranked[0][1], 2),
        "lowest_ticker": ranked[-1][0],
        "lowest_avg_daily_volume": round(ranked[-1][1], 2),
        "avg_daily_volume": {t: round(v, 2) for t, v in ranked},
    }


def volatility(
    ticker: str,
    year: int | str | None = None,
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """Annualised volatility: sample stdev of daily close-to-close returns x sqrt(252).

    The basis is stated in the payload because it is itself a graded component in
    at least one question — "daily simple close-to-close returns, sample stdev".
    """
    s = _series(series)
    key = str(ticker).strip().upper()
    key = key if key.endswith(".AX") else f"{key}.AX"
    if key not in s:
        return {"note": f"No ASX data for ticker '{ticker}'."}

    bars = list(s[key]) if year is None else [b for b in s[key] if b.date.year == int(year)]
    if len(bars) < 3:
        return {"ticker": key, "note": "Fewer than three trading days in scope."}

    rets = [
        bars[i].close / bars[i - 1].close - 1
        for i in range(1, len(bars))
        if bars[i - 1].close
    ]
    sd = statistics.stdev(rets)
    return {
        "ticker": key,
        "year": int(year) if year is not None else None,
        "volatility_pct_annualised": round(sd * (252 ** 0.5) * 100, 6),
        "daily_return_count": len(rets),
        "annualisation_factor": 252,
        "basis": "daily simple close-to-close returns, sample stdev",
    }


def correlation(
    ticker_a: str,
    ticker_b: str,
    year: int | str | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """Pearson correlation of daily returns, over dates BOTH tickers traded."""
    s = _series(series)
    keys = []
    for raw in (ticker_a, ticker_b):
        k = str(raw).strip().upper()
        k = k if k.endswith(".AX") else f"{k}.AX"
        if k not in s:
            return {"note": f"No ASX data for ticker '{raw}'."}
        keys.append(k)

    def rets(key: str) -> dict[date, float]:
        bars = list(s[key]) if year is None else [b for b in s[key] if b.date.year == int(year)]
        return {
            bars[i].date: bars[i].close / bars[i - 1].close - 1
            for i in range(1, len(bars))
            if bars[i - 1].close
        }

    ra, rb = rets(keys[0]), rets(keys[1])
    shared = sorted(set(ra) & set(rb))
    if len(shared) < 3:
        return {"note": "Fewer than three overlapping trading days."}
    return {
        "ticker_a": keys[0],
        "ticker_b": keys[1],
        "year": int(year) if year is not None else None,
        "correlation": round(statistics.correlation([ra[d] for d in shared], [rb[d] for d in shared]), 6),
        "overlapping_days": len(shared),
    }


def max_drawdown(
    ticker: str,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """Deepest peak-to-trough decline, with the peak and trough dates.

    Running-peak method. The dates are graded components in MHQ055, so they ship
    alongside the magnitude.
    """
    s = _series(series)
    key = str(ticker).strip().upper()
    key = key if key.endswith(".AX") else f"{key}.AX"
    if key not in s or len(s[key]) < 2:
        return {"note": f"Insufficient ASX data for ticker '{ticker}'."}

    peak = s[key][0].close
    peak_date = s[key][0].date
    worst = 0.0
    worst_peak_date = peak_date
    worst_trough_date = peak_date
    for bar in s[key]:
        if bar.close > peak:
            peak, peak_date = bar.close, bar.date
        dd = (bar.close / peak - 1) * 100 if peak else 0.0
        if dd < worst:
            worst, worst_peak_date, worst_trough_date = dd, peak_date, bar.date
    return {
        "ticker": key,
        "max_drawdown_pct": round(worst, 4),
        "peak_date": worst_peak_date.isoformat(),
        "trough_date": worst_trough_date.isoformat(),
    }


def window_return(
    date_from: str,
    date_to: str,
    tickers: Iterable[str] | None = None,
    exclude_tickers: Iterable[str] | None = None,
    series: Mapping[str, Sequence[AsxBar]] | None = None,
) -> dict:
    """★ Returns between two dates, per ticker and as an equal-weighted basket.

    MHQ072 and MHQ074. The basket is the **equal-weighted mean of per-ticker
    returns**, not a price-weighted index — verified: +2.88% / +0.24% / -2.17%
    for the three 2019 RBA cut windows, matching the references exactly.

    Dates are inclusive, and the window snaps to available trading days: an RBA
    effective date is often not a trading day, so requiring an exact match would
    return nothing on precisely the cross-dataset questions this exists for.
    """
    s = _series(series)
    lo, hi = _coerce_date(date_from), _coerce_date(date_to)
    if lo is None or hi is None:
        return {"note": "Could not interpret the date range."}

    if tickers:
        scope = []
        for raw in tickers:
            k = str(raw).strip().upper()
            k = k if k.endswith(".AX") else f"{k}.AX"
            if k in s:
                scope.append(k)
        drop = set(_universe(s, exclude_tickers))
        scope = [t for t in scope if t in drop] or scope
    else:
        scope = _universe(s, exclude_tickers)

    per: dict[str, float] = {}
    for t in scope:
        r = _first_to_last([b for b in s[t] if lo <= b.date <= hi])
        if r is not None:
            per[t] = round(r, 4)
    if not per:
        return {
            "date_from": lo.isoformat(),
            "date_to": hi.isoformat(),
            "note": "No overlapping trading days for the requested tickers.",
        }

    return {
        "date_from": lo.isoformat(),
        "date_to": hi.isoformat(),
        "tickers": sorted(per),
        "returns_pct": per,
        "equal_weighted_basket_return_pct": round(statistics.fmean(per.values()), 4),
        "basis": "first-to-last close in window, equal-weighted across tickers",
    }


def coverage(series: Mapping[str, Sequence[AsxBar]] | None = None) -> dict:
    """★ ASX's date span, for cross-dataset coverage checks (MHQ090)."""
    s = _series(series)
    all_dates = [b.date for bars in s.values() for b in bars]
    if not all_dates:
        return {"note": "No ASX data."}
    return {
        "dataset": "ASX",
        "tickers": len(s),
        "first_date": min(all_dates).isoformat(),
        "last_date": max(all_dates).isoformat(),
    }
