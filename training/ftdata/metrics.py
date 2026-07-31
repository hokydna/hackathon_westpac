"""Deterministic metrics over the three corpora.

Every gold answer in the training set is produced here, never by a language model. Each
function returns a plain dict of **canonical evidence fields** (``evidence_v1``) so the
same names appear in training data and in the runtime's tool results.

Metric coverage is derived from the real 15-question public bank, not from the handoff
pack's shorter list. Items the handoff pack omits but the bank requires are marked ★ in
``training/README.md``: ``cycle_summary``, ``describe``, ``window_return``,
``retrieve_by_headline``, ``max_drawdown``, ``correlation``, ``volatility``, ``coverage``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from .corpora import (
    TABCORP,
    AfrIndex,
    AsxCorpus,
    RbaCorpus,
    RbaRecord,
    coverage as _coverage,
    display_date,
    load_afr,
    load_asx,
    load_rba,
)

# --------------------------------------------------------------------------------------
# RBA
# --------------------------------------------------------------------------------------


def rba_describe(rba: RbaCorpus) -> dict:
    return {
        "record_count": len(rba.records),
        "first_effective_date": rba.records[0].iso,
        "last_effective_date": rba.records[-1].iso,
        "first_cash_rate_target_pct": rba.records[0].cash_rate_target_pct,
        "last_cash_rate_target_pct": rba.records[-1].cash_rate_target_pct,
    }


def rba_change_counts(
    rba: RbaCorpus, start: date | None = None, end: date | None = None
) -> dict:
    rows = rba.in_range(start, end)
    changes = [r for r in rows if r.change_pp != 0.0]
    increases = [r for r in changes if r.change_pp > 0]
    decreases = [r for r in changes if r.change_pp < 0]
    ev = {
        "record_count": len(rows),
        "change_count": len(changes),
        "increase_count": len(increases),
        "decrease_count": len(decreases),
        "hold_count": len(rows) - len(changes),
    }
    if start is not None:
        ev["range_start_date"] = start.isoformat()
    if end is not None:
        ev["range_end_date"] = end.isoformat()
    return ev


def rba_extreme(rba: RbaCorpus, mode: str = "minimum") -> dict:
    """Lowest/highest cash-rate target, its first effective date, and how many records
    sit at that level."""
    targets = [r.cash_rate_target_pct for r in rba.records]
    level = min(targets) if mode == "minimum" else max(targets)
    at_level = [r for r in rba.records if r.cash_rate_target_pct == level]
    key = "minimum_rate_pct" if mode == "minimum" else "maximum_rate_pct"
    return {
        key: level,
        "first_effective_date": at_level[0].iso,
        "last_effective_date": at_level[-1].iso,
        "record_count": len(at_level),
    }


def rba_lookup_rate(rba: RbaCorpus, when: date) -> dict:
    """Rate *in force* on a date — as-of, never nearest-match."""
    rec = rba.as_of(when)
    if rec is None:
        return {
            "query_date": when.isoformat(),
            "cash_rate_target_pct": None,
            "effective_date": None,
        }
    return {
        "query_date": when.isoformat(),
        "cash_rate_target_pct": rec.cash_rate_target_pct,
        "effective_date": rec.iso,
        "change_pp": rec.change_pp,
    }


def rba_cycle_summary(
    rba: RbaCorpus, start: date, end: date, direction: str = "easing"
) -> dict:
    """Cumulative movement across a tightening/easing window (MHQ035 / MHQ084 shape).

    ``start_rate_pct`` is the target *before* the first move in the window, which is what
    the reference answers quote ("from 4.75% before the first cut").
    """
    sign = -1 if direction == "easing" else 1
    rows = rba.in_range(start, end)
    moves = [r for r in rows if r.change_pp != 0.0 and (r.change_pp > 0) == (sign > 0)]
    per_year: dict[int, int] = {}
    for r in moves:
        per_year[r.effective_date.year] = per_year.get(r.effective_date.year, 0) + 1

    prior = rba.as_of(moves[0].effective_date - timedelta(days=1)) if moves else None
    end_rec = rows[-1] if rows else None
    return {
        "direction": direction,
        "range_start_date": start.isoformat(),
        "range_end_date": end.isoformat(),
        "move_count": len(moves),
        "moves_per_year": {str(k): v for k, v in sorted(per_year.items())},
        "cumulative_change_pp": round(sum(r.change_pp for r in moves), 2),
        "start_rate_pct": prior.cash_rate_target_pct if prior else None,
        "end_rate_pct": end_rec.cash_rate_target_pct if end_rec else None,
        "first_move_date": moves[0].iso if moves else None,
        "last_move_date": moves[-1].iso if moves else None,
    }


def rba_year_summary(rba: RbaCorpus, year: int) -> dict:
    rows = rba.in_year(year)
    cuts = [r for r in rows if r.change_pp < 0]
    hikes = [r for r in rows if r.change_pp > 0]
    return {
        "year": year,
        "record_count": len(rows),
        "cut_count": len(cuts),
        "hike_count": len(hikes),
        "cumulative_change_pp": round(sum(r.change_pp for r in rows), 2),
        "year_end_cash_rate_target_pct": rows[-1].cash_rate_target_pct if rows else None,
        "change_dates": [r.iso for r in rows if r.change_pp != 0.0],
    }


def rba_longest_hold(rba: RbaCorpus) -> dict:
    """Longest run of consecutive records at one target, by record count and by days."""
    best: tuple[int, int, RbaRecord, RbaRecord] | None = None
    i = 0
    recs = rba.records
    while i < len(recs):
        j = i
        while j + 1 < len(recs) and recs[j + 1].cash_rate_target_pct == recs[i].cash_rate_target_pct:
            j += 1
        span = (recs[j].effective_date - recs[i].effective_date).days
        cand = (j - i + 1, span, recs[i], recs[j])
        if best is None or cand[0] > best[0]:
            best = cand
        i = j + 1
    assert best is not None
    n, span, first, last = best
    return {
        "hold_record_count": n,
        "hold_span_days": span,
        "cash_rate_target_pct": first.cash_rate_target_pct,
        "hold_start_date": first.iso,
        "hold_end_date": last.iso,
    }


def rba_longest_gap(rba: RbaCorpus) -> dict:
    """Longest gap in days between consecutive decision records."""
    best = max(
        (
            ((b.effective_date - a.effective_date).days, a, b)
            for a, b in zip(rba.records, rba.records[1:])
        ),
        key=lambda t: t[0],
    )
    days, a, b = best
    return {
        "gap_days": days,
        "gap_start_date": a.iso,
        "gap_end_date": b.iso,
    }


# --------------------------------------------------------------------------------------
# ASX
# --------------------------------------------------------------------------------------


def asx_describe(asx: AsxCorpus) -> dict:
    rows = sorted(set(asx.rows_per_file.values()))
    return {
        "ticker_file_count": asx.file_count,
        "rows_per_file": rows[0] if len(rows) == 1 else rows,
        "total_rows": sum(asx.rows_per_file.values()),
        "first_trade_date": asx.date_min.isoformat(),
        "last_trade_date": asx.date_max.isoformat(),
        "tickers": asx.tickers(),
    }


def asx_annual_return(asx: AsxCorpus, ticker: str, year: int) -> dict:
    series = asx.get(ticker)
    bars = series.in_year(year)
    if not bars:
        return {"ticker": series.ticker, "year": year, "price_return_pct": None}
    first, last = bars[0], bars[-1]
    return {
        "ticker": series.ticker,
        "year": year,
        "start_trade_date": first.iso,
        "end_trade_date": last.iso,
        "start_close": first.close,
        "end_close": last.close,
        "price_return_pct": (last.close / first.close - 1.0) * 100.0,
        "session_count": len(bars),
    }


def asx_full_sample_return(asx: AsxCorpus, ticker: str) -> dict:
    series = asx.get(ticker)
    first, last = series.bars[0], series.bars[-1]
    return {
        "ticker": series.ticker,
        "start_trade_date": first.iso,
        "end_trade_date": last.iso,
        "start_close": first.close,
        "end_close": last.close,
        "price_return_pct": (last.close / first.close - 1.0) * 100.0,
        "session_count": len(series.bars),
    }


def asx_window_return(asx: AsxCorpus, ticker: str, start: date, end: date) -> dict:
    """Exact-date close-to-close return. ``None`` when either end is not a trading day —
    the runtime must then either resolve the date or state the limitation."""
    series = asx.get(ticker)
    a, b = series.on(start), series.on(end)
    ev: dict = {
        "ticker": series.ticker,
        "start_trade_date": start.isoformat(),
        "end_trade_date": end.isoformat(),
    }
    if a is None or b is None:
        ev["price_return_pct"] = None
        ev["missing_trade_dates"] = [
            x.isoformat() for x, bar in ((start, a), (end, b)) if bar is None
        ]
        return ev
    ev.update(
        start_close=a.close,
        end_close=b.close,
        price_return_pct=(b.close / a.close - 1.0) * 100.0,
        absolute_change=b.close - a.close,
    )
    return ev


def asx_basket_window_return(
    asx: AsxCorpus,
    start: date,
    end: date,
    exclude: Sequence[str] = (TABCORP,),
) -> dict:
    """Equal-weighted basket: the simple average of constituent close-to-close returns.

    This is the MHQ072/MHQ074 definition — average of returns, not the return of an
    averaged price index. The two differ, and only one matches the reference answers.
    """
    tickers = asx.tickers(exclude=exclude)
    per: dict[str, float] = {}
    missing: list[str] = []
    for t in tickers:
        r = asx_window_return(asx, t, start, end)
        if r.get("price_return_pct") is None:
            missing.append(t)
        else:
            per[t] = r["price_return_pct"]
    return {
        "start_trade_date": start.isoformat(),
        "end_trade_date": end.isoformat(),
        "basket_size": len(per),
        "excluded_tickers": list(exclude),
        "basket_return_pct": (sum(per.values()) / len(per)) if per else None,
        "constituent_returns_pct": per,
        "missing_tickers": missing,
    }


def asx_rank_annual_returns(
    asx: AsxCorpus, year: int, exclude: Sequence[str] = (TABCORP,)
) -> dict:
    rows = []
    for t in asx.tickers(exclude=exclude):
        r = asx_annual_return(asx, t, year)
        if r.get("price_return_pct") is not None:
            rows.append((t, r["price_return_pct"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    return {
        "year": year,
        "excluded_tickers": list(exclude),
        "ranked_returns_pct": [{"rank": i + 1, "ticker": t, "price_return_pct": v}
                               for i, (t, v) in enumerate(rows)],
        "best_ticker": rows[0][0],
        "best_price_return_pct": rows[0][1],
        "worst_ticker": rows[-1][0],
        "worst_price_return_pct": rows[-1][1],
        "basket_size": len(rows),
        "average_price_return_pct": sum(v for _, v in rows) / len(rows),
    }


def asx_avg_volume(asx: AsxCorpus, ticker: str) -> dict:
    series = asx.get(ticker)
    vols = [b.volume for b in series.bars]
    return {
        "ticker": series.ticker,
        "avg_daily_volume": sum(vols) / len(vols),
        "session_count": len(vols),
        "total_volume": sum(vols),
    }


def asx_rank_avg_volume(asx: AsxCorpus, exclude: Sequence[str] = (TABCORP,)) -> dict:
    rows = [
        (t, asx_avg_volume(asx, t)["avg_daily_volume"]) for t in asx.tickers(exclude=exclude)
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    return {
        "excluded_tickers": list(exclude),
        "ranked_avg_daily_volume": [
            {"rank": i + 1, "ticker": t, "avg_daily_volume": v} for i, (t, v) in enumerate(rows)
        ],
        "highest_ticker": rows[0][0],
        "highest_avg_daily_volume": rows[0][1],
        "lowest_ticker": rows[-1][0],
        "lowest_avg_daily_volume": rows[-1][1],
    }


def _daily_returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def asx_volatility(
    asx: AsxCorpus, ticker: str, year: int | None = None, trading_days: int = 252
) -> dict:
    """Annualised standard deviation of daily simple close-to-close returns.

    Sample standard deviation (n-1), scaled by ``sqrt(252)``. The convention is recorded
    in the evidence so the answer can never imply a different one.
    """
    series = asx.get(ticker)
    bars = series.in_year(year) if year is not None else series.bars
    rets = _daily_returns([b.close for b in bars])
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return {
        "ticker": series.ticker,
        "year": year,
        "volatility_pct_annualised": math.sqrt(var) * math.sqrt(trading_days) * 100.0,
        "daily_return_count": n,
        "annualisation_factor": trading_days,
        "basis": "daily simple close-to-close returns, sample stdev",
    }


def asx_correlation(
    asx: AsxCorpus, ticker_a: str, ticker_b: str, year: int | None = None
) -> dict:
    """Pearson correlation of daily simple returns over the shared trading dates."""
    sa, sb = asx.get(ticker_a), asx.get(ticker_b)
    bars_a = sa.in_year(year) if year is not None else sa.bars
    bars_b = sb.in_year(year) if year is not None else sb.bars
    map_b = {b.trade_date: b.close for b in bars_b}
    dates = [b.trade_date for b in bars_a if b.trade_date in map_b]
    closes_a = [next(b.close for b in bars_a if b.trade_date == d) for d in dates] \
        if len(dates) != len(bars_a) else [b.close for b in bars_a]
    closes_b = [map_b[d] for d in dates]
    ra, rb = _daily_returns(closes_a), _daily_returns(closes_b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return {
        "ticker_a": sa.ticker,
        "ticker_b": sb.ticker,
        "year": year,
        "correlation": cov / (va * vb),
        "paired_return_count": n,
        "basis": "Pearson correlation of daily simple close-to-close returns",
    }


def asx_max_drawdown(asx: AsxCorpus, ticker: str, year: int | None = None) -> dict:
    """Running-peak drawdown on closes; reports the trough and the peak it fell from."""
    series = asx.get(ticker)
    bars = series.in_year(year) if year is not None else series.bars
    peak = bars[0].close
    peak_date = bars[0].trade_date
    worst = 0.0
    worst_peak_date = bars[0].trade_date
    worst_trough_date = bars[0].trade_date
    for b in bars:
        if b.close > peak:
            peak, peak_date = b.close, b.trade_date
        dd = b.close / peak - 1.0
        if dd < worst:
            worst, worst_peak_date, worst_trough_date = dd, peak_date, b.trade_date
    return {
        "ticker": series.ticker,
        "year": year,
        "max_drawdown_pct": worst * 100.0,
        "peak_date": worst_peak_date.isoformat(),
        "trough_date": worst_trough_date.isoformat(),
    }


def asx_rank_max_drawdown(
    asx: AsxCorpus, exclude: Sequence[str] = (TABCORP,), top: int = 3
) -> dict:
    rows = [asx_max_drawdown(asx, t) for t in asx.tickers(exclude=exclude)]
    rows.sort(key=lambda r: r["max_drawdown_pct"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return {
        "excluded_tickers": list(exclude),
        "ranked_max_drawdowns": rows[:top],
        "worst_ticker": rows[0]["ticker"],
        "worst_max_drawdown_pct": rows[0]["max_drawdown_pct"],
    }


def asx_close_on(asx: AsxCorpus, ticker: str, when: date) -> dict:
    """Close on an exact date, with explicit trading-date resolution when closed."""
    series = asx.get(ticker)
    exact = series.on(when)
    if exact is not None:
        return {
            "ticker": series.ticker,
            "query_date": when.isoformat(),
            "trade_date": exact.iso,
            "close": exact.close,
            "date_resolved": False,
        }
    prior = series.on_or_before(when)
    return {
        "ticker": series.ticker,
        "query_date": when.isoformat(),
        "trade_date": prior.iso if prior else None,
        "close": prior.close if prior else None,
        "date_resolved": True,
        "resolution_rule": "last trading day on or before the requested date",
    }


# --------------------------------------------------------------------------------------
# AFR
# --------------------------------------------------------------------------------------


def afr_describe(afr: AfrIndex) -> dict:
    return {
        "file_count": afr.file_count,
        "record_count": afr.total_articles,
        "undated_record_count": afr.undated_records,
        "first_publication_date": f"{afr.date_min[:4]}-{afr.date_min[4:6]}-{afr.date_min[6:]}",
        "last_publication_date": f"{afr.date_max[:4]}-{afr.date_max[4:6]}-{afr.date_max[6:]}",
    }


def afr_term_count(
    afr: AfrIndex, term: str, year: int | None = None, month: int | None = None
) -> dict:
    ev = {
        "search_term": term,
        "search_rule": r"case-insensitive \bterm\b across HEADLINE+SUBHEAD+INTRO+TEXT, once per record",
        "match_count": afr.term_count(term, year, month),
    }
    if year is not None:
        ev["year"] = year
    if month is not None:
        ev["month"] = month
    return ev


def afr_pattern_count(
    afr: AfrIndex, name: str, year: int | None = None, month: int | None = None
) -> dict:
    from .corpora import AFR_REGEX_PATTERNS

    ev = {
        "search_pattern": AFR_REGEX_PATTERNS[name],
        "search_rule": "case-insensitive regex across HEADLINE+SUBHEAD+INTRO+TEXT, once per record",
        "match_count": afr.pattern_count(name, year, month),
    }
    if year is not None:
        ev["year"] = year
    if month is not None:
        ev["month"] = month
    return ev


def afr_peak_year_and_month(afr: AfrIndex, term: str) -> dict:
    by_year = afr.term_by_year(term)
    by_month = afr.term_by_month(term)
    peak_year = max(by_year.items(), key=lambda kv: kv[1])
    peak_month = max(by_month.items(), key=lambda kv: kv[1])
    return {
        "search_term": term,
        "search_rule": r"case-insensitive \bterm\b across HEADLINE+SUBHEAD+INTRO+TEXT, once per record",
        "match_count": sum(by_year.values()),
        "peak_year": peak_year[0],
        "peak_year_count": peak_year[1],
        "peak_month": peak_month[0],
        "peak_month_count": peak_month[1],
    }


def afr_share(afr: AfrIndex, term: str, year: int | None = None) -> dict:
    matches = afr.term_count(term, year)
    total = afr.record_count(year)
    return {
        "search_term": term,
        "year": year,
        "match_count": matches,
        "total_records": total,
        "share_pct": matches / total * 100.0 if total else None,
    }


def afr_monthly_series(afr: AfrIndex, term: str, year: int) -> dict:
    by_month = {
        k: v for k, v in afr.term_by_month(term).items() if k.startswith(f"{year:04d}")
    }
    if not by_month:
        return {"search_term": term, "year": year, "match_count": 0, "monthly_counts": {}}
    peak = max(by_month.items(), key=lambda kv: kv[1])
    low = min(by_month.items(), key=lambda kv: kv[1])
    return {
        "search_term": term,
        "year": year,
        "match_count": sum(by_month.values()),
        "monthly_counts": by_month,
        "peak_month": peak[0],
        "peak_month_count": peak[1],
        "lowest_month": low[0],
        "lowest_month_count": low[1],
    }


def afr_retrieve_by_headline(afr: AfrIndex, headline: str, publication_date: str) -> dict:
    """Exact-headline retrieval — the entry point for every sentiment question."""
    rec = afr.headlines.get(f"{headline}|{publication_date}")
    if rec is None:
        return {
            "headline": headline,
            "publication_date": publication_date,
            "match_count": 0,
        }
    return {
        "headline": rec["headline"],
        "publication_date": f"{rec['publication_date'][:4]}-{rec['publication_date'][4:6]}-{rec['publication_date'][6:]}",
        "match_count": 1,
        "subhead": rec["subhead"],
        "article_excerpt": (rec["intro"] or rec["text"])[:600],
    }


# --------------------------------------------------------------------------------------
# coverage — feeds every legitimate refusal
# --------------------------------------------------------------------------------------


def coverage_check(datasets: Sequence[str], start: date, end: date) -> dict:
    """Compare a requested window against each corpus's real span.

    MHQ090's shape: the answer is a refusal, and the *reasoning* (which corpus ends when)
    carries most of the points, not the verdict.
    """
    cov = _coverage()
    spans = {}
    unsupported = []
    for name in datasets:
        c = cov[name]
        spans[name] = {"first_date": c.first, "last_date": c.last}
        if end.isoformat() > c.last or start.isoformat() < c.first:
            unsupported.append(name)
    return {
        "requested_start_date": start.isoformat(),
        "requested_end_date": end.isoformat(),
        "dataset_coverage": spans,
        "unsupported_datasets": unsupported,
        "fully_supported": not unsupported,
    }


# --------------------------------------------------------------------------------------
# convenience bundle
# --------------------------------------------------------------------------------------


@dataclass
class Corpora:
    rba: RbaCorpus
    asx: AsxCorpus
    afr: AfrIndex


def load_all(progress: bool = False) -> Corpora:
    return Corpora(rba=load_rba(), asx=load_asx(), afr=load_afr(progress=progress))
