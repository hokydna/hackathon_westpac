#!/usr/bin/env python3
"""Generate the QLoRA training corpus from the three participant-package datasets.

Every gold answer here is computed by ``ftdata.metrics`` — never by a language model.
The adapter is being taught a *discipline* (state every requested component exactly, in
the grader's own shape, without hedging or invention), not new knowledge, so the targets
have to be arithmetically perfect and stylistically uniform.

Pipeline::

    ftdata.gate  ->  families  ->  split by entity key  ->  chat JSONL

Run::

    python3 prepare_data.py                 # ~4,000 examples, seed 42
    python3 prepare_data.py --target 800    # quick pass for smoke tests

Three properties are enforced, not hoped for:

* **The blocking gate runs first.** If the metric layer cannot reproduce the organizer's
  own reference answers for the 15 public questions, generation refuses to start.
* **Splits are grouped by entity key, never by row.** A ticker-year that appears in
  ``train`` can never appear in ``heldout``; the assertion is logged. Row-level shuffling
  would leak and inflate the base-vs-fine-tuned comparison, which is exactly what the
  rubric means by "must not contain hidden evaluation data".
* **Every ``required_fact`` appears verbatim in ``expected_answer``**, and compound facts
  stay one sentence. Four of the fifteen public questions are a single all-or-nothing
  10-point component bundling three or four numbers — 26.7% of public points behind four
  YES/NO gates. Splitting such a fact in two scores zero on a perfectly-computed answer.

The 15 public questions are **fixtures, never training rows**.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from typing import Callable, Iterator

from ftdata import fmt
from ftdata import metrics as M
from ftdata import paths
from ftdata.corpora import ASX_NAMES, TABCORP, TICKER_TO_NAME
from ftdata.example import Example, build
from ftdata.gate import assert_gate
from ftdata.prompts import (
    SENTIMENT_CHAR_CAP,
    render_sentiment_messages,
    render_synthesis_messages,
)

SEED = 42

# Evidence serialisation drift insurance (§6.3 rule 3). The runtime emits "json"; the
# other two styles appear in a minority of rows so the adapter does not bind to one
# serialiser's whitespace.
EVIDENCE_STYLES = ("json",) * 8 + ("compact_json", "kv_lines")


# ======================================================================================
# helpers
# ======================================================================================


def _name(ticker: str) -> str:
    """``BHP.AX`` -> ``BHP``. The bank quotes bare tickers in answers."""
    return ticker


def _company(ticker: str) -> str:
    return TICKER_TO_NAME.get(ticker, ticker.replace(".AX", ""))


class Ctx:
    """Corpora plus a seeded RNG, threaded through every family."""

    def __init__(self, corpora: M.Corpora, rng: random.Random) -> None:
        self.rba = corpora.rba
        self.asx = corpora.asx
        self.afr = corpora.afr
        self.rng = rng

    def pick(self, seq):
        return self.rng.choice(list(seq))

    def style(self) -> str:
        return self.rng.choice(EVIDENCE_STYLES)


#: A family yields (question, facts, evidence, components, datasets, difficulty,
#: split_key) tuples. Registered families are sampled round-robin up to the mix quota.
Family = Callable[[Ctx], Iterator[Example]]

_REGISTRY: dict[str, list[tuple[str, Family]]] = {}


def family(slice_: str, name: str):
    def deco(fn: Family) -> Family:
        _REGISTRY.setdefault(slice_, []).append((name, fn))
        return fn

    return deco


# ======================================================================================
# RBA families
# ======================================================================================


@family("rba_single", "rba_change_counts")
def _rba_change_counts(c: Ctx) -> Iterator[Example]:
    """MHQ001's shape: one compound all-or-nothing component."""
    windows: list[tuple[date | None, date | None]] = [(None, None)]
    years = sorted({r.effective_date.year for r in c.rba.records})
    seen_w: set[tuple[int, int]] = set()
    for _ in range(400):
        a, b = sorted(c.rng.sample(years, 2))
        if (a, b) in seen_w:
            continue
        seen_w.add((a, b))
        windows.append((date(a, 1, 1), date(b, 12, 31)))

    templates = [
        "Across the RBA cash rate decision records{span}, how many changed the rate, and what was the split between increases and decreases?",
        "Using the RBA dataset{span}, count the decision records that moved the cash rate and break that down into increases and decreases.",
        "How many of the RBA decision records{span} actually changed the cash rate target, and how many of those were rises versus cuts?",
        "For the RBA rate decisions{span}, report the number of records, how many changed the rate, and the increase/decrease split.",
    ]
    for start, end in windows:
        ev = M.rba_change_counts(c.rba, start, end)
        if ev["change_count"] == 0:
            continue
        span = "" if start is None else f" between {fmt.d(start)} and {fmt.d(end)}"
        fact = (
            f"{fmt.count(ev['change_count'])} of the {fmt.count(ev['record_count'])} "
            f"decision records changed the rate: {fmt.count(ev['increase_count'])} increases "
            f"and {fmt.count(ev['decrease_count'])} decreases."
        )
        key = f"rba_change_counts:{start}:{end}"
        yield build(
            family="rba_change_counts",
            slice_="rba_single",
            question=c.pick(templates).format(span=span),
            facts=[fact],
            evidence=ev,
            components=["change_count", "increase_count", "decrease_count"],
            datasets=["RBA"],
            difficulty="medium",
            split_key=key,
            evidence_style=c.style(),
            notes="compound single-component fact, MHQ001 shape",
        )


@family("rba_single", "rba_cycle_summary")
def _rba_cycle(c: Ctx) -> Iterator[Example]:
    years = sorted({r.effective_date.year for r in c.rba.records})
    templates = [
        "Summarise the RBA {direction} cycle between {start} and {end}: how many moves, the cumulative change, and the start and end cash rate targets.",
        "Between {start} and {end}, how many {direction} moves did the RBA make, what was the total movement in percentage points, and where did the cash rate start and finish?",
        "Describe the {direction} phase from {start} to {end} using the RBA data — move count, cumulative percentage-point change, and the rate before and after.",
    ]
    for _ in range(400):
        a, b = sorted(c.rng.sample(years, 2))
        if b - a < 1:
            continue
        direction = c.pick(["easing", "tightening"])
        start, end = date(a, 1, 1), date(b, 12, 31)
        ev = M.rba_cycle_summary(c.rba, start, end, direction)
        if ev["move_count"] < 2 or ev["start_rate_pct"] is None:
            continue
        word = "cuts" if direction == "easing" else "increases"
        per_year = ", ".join(f"{v} in {k}" for k, v in ev["moves_per_year"].items())
        fact = (
            f"The RBA made {ev['move_count']} {word} between {fmt.d(start)} and {fmt.d(end)} "
            f"({per_year}), a cumulative {fmt.pp(ev['cumulative_change_pp'])}, taking the cash "
            f"rate target from {fmt.rate_pct(ev['start_rate_pct'])} to "
            f"{fmt.rate_pct(ev['end_rate_pct'])}."
        )
        yield build(
            family="rba_cycle_summary",
            slice_="rba_single",
            question=c.pick(templates).format(
                direction=direction, start=fmt.d(start), end=fmt.d(end)
            ),
            facts=[fact],
            evidence=ev,
            components=[
                "move_count",
                "cumulative_change_pp",
                "start_rate_pct",
                "end_rate_pct",
            ],
            datasets=["RBA"],
            difficulty="hard",
            split_key=f"rba_cycle:{direction}:{a}:{b}",
            evidence_style=c.style(),
            notes="compound single-component fact; percentage POINTS, never percent",
        )


@family("rba_single", "rba_lookup_rate")
def _rba_lookup(c: Ctx) -> Iterator[Example]:
    """As-of lookup. The wrong implementation (nearest match) can return a *future*
    decision, which is the single easiest way to be confidently wrong here."""
    lo = c.rba.records[0].effective_date
    hi = c.rba.records[-1].effective_date
    templates = [
        "What was the RBA cash rate target in force on {when}?",
        "On {when}, which cash rate target was the RBA applying, and when did it take effect?",
        "Give the cash rate in effect on {when} and the effective date of the decision that set it.",
        "As at {when}, what was the RBA's cash rate target?",
    ]
    span = (hi - lo).days
    for _ in range(500):
        when = lo + timedelta(days=c.rng.randrange(span))
        ev = M.rba_lookup_rate(c.rba, when)
        if ev["cash_rate_target_pct"] is None:
            continue
        fact = (
            f"The RBA cash rate target in force on {fmt.d(when)} was "
            f"{fmt.rate_pct(ev['cash_rate_target_pct'])}, set effective "
            f"{fmt.d(date.fromisoformat(ev['effective_date']))}."
        )
        yield build(
            family="rba_lookup_rate",
            slice_="rba_single",
            question=c.pick(templates).format(when=fmt.d(when)),
            facts=[fact],
            evidence=ev,
            components=["cash_rate_target_pct", "effective_date"],
            datasets=["RBA"],
            difficulty="easy",
            split_key=f"rba_lookup:{ev['effective_date']}",
            evidence_style=c.style(),
            notes="as-of semantics, never nearest-match",
        )


@family("rba_single", "rba_extremes")
def _rba_extremes(c: Ctx) -> Iterator[Example]:
    templates = {
        "minimum": [
            "What is the lowest cash rate target in the RBA dataset, when did it first take effect, and how many records sit at that level?",
            "Identify the record-low RBA cash rate target, its first effective date, and the number of decision records at that rate.",
        ],
        "maximum": [
            "What is the highest cash rate target in the RBA dataset, when did it first apply, and how many records are at that level?",
            "Report the peak RBA cash rate target in the data, the date it took effect, and how many records sit there.",
        ],
    }
    for mode in ("minimum", "maximum"):
        ev = M.rba_extreme(c.rba, mode)
        level = ev["minimum_rate_pct" if mode == "minimum" else "maximum_rate_pct"]
        word = "lowest" if mode == "minimum" else "highest"
        fact = (
            f"The {word} cash rate target in the dataset is {fmt.rate_pct(level)}, first "
            f"effective {fmt.d(date.fromisoformat(ev['first_effective_date']))}, and "
            f"{fmt.count(ev['record_count'])} decision records sit at that level."
        )
        yield build(
            family="rba_extremes",
            slice_="rba_single",
            question=c.pick(templates[mode]),
            facts=[fact],
            evidence=ev,
            components=["rate_pct", "first_effective_date", "record_count"],
            datasets=["RBA"],
            difficulty="easy",
            split_key=f"rba_extreme:{mode}",
            evidence_style=c.style(),
        )


@family("rba_single", "rba_year_summary")
def _rba_year(c: Ctx) -> Iterator[Example]:
    templates = [
        "What did the RBA do in {year}? Give the number of decision records, cuts, hikes, the cumulative change, and the year-end cash rate target.",
        "Summarise RBA policy in {year}: how many records, how many cuts and hikes, the net movement, and where the cash rate finished.",
        "For calendar {year}, report the RBA's decision count, cut and hike counts, net percentage-point change, and closing cash rate target.",
    ]
    for year in sorted({r.effective_date.year for r in c.rba.records}):
        ev = M.rba_year_summary(c.rba, year)
        if ev["record_count"] == 0:
            continue
        facts = [
            f"In {year} the RBA dataset carries {fmt.count(ev['record_count'])} decision "
            f"records, comprising {ev['cut_count']} cuts and {ev['hike_count']} increases, a "
            f"net {fmt.pp(ev['cumulative_change_pp'])}.",
            f"The cash rate target finished {year} at "
            f"{fmt.rate_pct(ev['year_end_cash_rate_target_pct'])}.",
        ]
        yield build(
            family="rba_year_summary",
            slice_="rba_single",
            question=c.pick(templates).format(year=year),
            facts=facts,
            evidence=ev,
            components=["record_count", "cut_count", "hike_count", "year_end_rate"],
            datasets=["RBA"],
            difficulty="medium",
            split_key=f"rba_year:{year}",
            evidence_style=c.style(),
        )


@family("rba_single", "rba_longest_hold")
def _rba_hold(c: Ctx) -> Iterator[Example]:
    ev = M.rba_longest_hold(c.rba)
    templates = [
        "What is the longest stretch the RBA held the cash rate unchanged in this dataset — at what level, over how many records, and between which dates?",
        "Identify the longest unchanged run of the cash rate target: the rate, the record count, the span in days, and the start and end dates.",
    ]
    fact = (
        f"The longest unchanged stretch held the cash rate target at "
        f"{fmt.rate_pct(ev['cash_rate_target_pct'])} across "
        f"{fmt.count(ev['hold_record_count'])} consecutive decision records, from "
        f"{fmt.d(date.fromisoformat(ev['hold_start_date']))} to "
        f"{fmt.d(date.fromisoformat(ev['hold_end_date']))}, a span of "
        f"{fmt.count(ev['hold_span_days'])} days."
    )
    for t in templates:
        yield build(
            family="rba_longest_hold",
            slice_="rba_single",
            question=t,
            facts=[fact],
            evidence=ev,
            components=["cash_rate_target_pct", "hold_record_count", "hold_span_days"],
            datasets=["RBA"],
            difficulty="medium",
            split_key="rba_longest_hold",
            evidence_style=c.style(),
        )


@family("rba_single", "rba_describe")
def _rba_describe(c: Ctx) -> Iterator[Example]:
    ev = M.rba_describe(c.rba)
    templates = [
        "Describe the RBA cash rate dataset: how many records, what date range, and the first and last cash rate targets.",
        "What does the RBA rates file contain — record count, coverage dates, and opening and closing cash rate targets?",
    ]
    facts = [
        f"The RBA dataset contains {fmt.count(ev['record_count'])} decision records covering "
        f"{fmt.d(date.fromisoformat(ev['first_effective_date']))} through "
        f"{fmt.d(date.fromisoformat(ev['last_effective_date']))}.",
        f"The first cash rate target is {fmt.rate_pct(ev['first_cash_rate_target_pct'])} and "
        f"the last is {fmt.rate_pct(ev['last_cash_rate_target_pct'])}.",
    ]
    for t in templates:
        yield build(
            family="rba_describe",
            slice_="rba_single",
            question=t,
            facts=facts,
            evidence=ev,
            components=["record_count", "date_range", "first_rate", "last_rate"],
            datasets=["RBA"],
            difficulty="easy",
            split_key="rba_describe",
            evidence_style=c.style(),
        )


# ======================================================================================
# ASX families
# ======================================================================================

ASX_YEARS = tuple(range(2015, 2022))


@family("asx_single", "asx_annual_return")
def _asx_annual(c: Ctx) -> Iterator[Example]:
    templates = [
        "What was {ticker}'s price return in {year}?",
        "Calculate the {year} price return for {ticker}, including the closing prices used.",
        "How did {ticker} perform over {year} on a close-to-close basis?",
        "Report {ticker}'s {year} return along with the first and last closes of the year.",
    ]
    for ticker in c.asx.tickers():
        for year in ASX_YEARS:
            ev = M.asx_annual_return(c.asx, ticker, year)
            if ev.get("price_return_pct") is None:
                continue
            facts = [
                f"{ticker} returned {fmt.signed_pct(ev['price_return_pct'])} over {year} on a "
                f"price basis, from a close of {fmt.price(ev['start_close'])} on "
                f"{fmt.d(date.fromisoformat(ev['start_trade_date']))} to "
                f"{fmt.price(ev['end_close'])} on "
                f"{fmt.d(date.fromisoformat(ev['end_trade_date']))}."
            ]
            yield build(
                family="asx_annual_return",
                slice_="asx_single",
                question=c.pick(templates).format(ticker=ticker, year=year),
                facts=facts,
                evidence=ev,
                components=["price_return_pct", "start_close", "end_close"],
                datasets=["ASX"],
                difficulty="easy",
                split_key=f"asx_annual:{ticker}:{year}",
                evidence_style=c.style(),
                notes="price return, never total shareholder return",
            )


@family("asx_single", "asx_rank_annual")
def _asx_rank(c: Ctx) -> Iterator[Example]:
    templates = [
        "Excluding {excl}, which ASX stock had the best {year} price return and which had the worst?",
        "For {year}, rank the ASX names excluding {excl} — report the best and worst performers with their returns.",
        "Which non-{excl} ticker led and which lagged in {year}, and by how much?",
    ]
    for year in ASX_YEARS:
        for excl in ([TABCORP], []):
            ev = M.asx_rank_annual_returns(c.asx, year, exclude=excl)
            label = _company(TABCORP) if excl else "none"
            if not excl:
                q_excl = "no tickers"
            else:
                q_excl = f"{label} ({TABCORP})"
            fact = (
                f"{ev['best_ticker']} had the best {year} price return at "
                f"{fmt.signed_pct(ev['best_price_return_pct'])}, and {ev['worst_ticker']} the "
                f"worst at {fmt.signed_pct(ev['worst_price_return_pct'])}."
            )
            yield build(
                family="asx_rank_annual",
                slice_="asx_single",
                question=c.pick(templates).format(year=year, excl=q_excl),
                facts=[fact],
                evidence=ev,
                components=["best_ticker", "best_return", "worst_ticker", "worst_return"],
                datasets=["ASX"],
                difficulty="medium",
                split_key=f"asx_rank_annual:{year}:{'excl' if excl else 'all'}",
                evidence_style=c.style(),
                notes="MHQ045 shape; exclude_tickers is first-class",
            )


@family("asx_single", "asx_full_sample")
def _asx_full(c: Ctx) -> Iterator[Example]:
    templates = [
        "What was {ticker}'s total price return across the whole sample, and over what dates?",
        "Over the full {ticker} history in this dataset, what is the close-to-close return?",
        "Give {ticker}'s full-sample price return with the start and end closes and dates.",
    ]
    for ticker in c.asx.tickers():
        ev = M.asx_full_sample_return(c.asx, ticker)
        facts = [
            f"{ticker} returned {fmt.signed_pct(ev['price_return_pct'])} over the full sample, "
            f"from {fmt.price(ev['start_close'])} on "
            f"{fmt.d(date.fromisoformat(ev['start_trade_date']))} to "
            f"{fmt.price(ev['end_close'])} on "
            f"{fmt.d(date.fromisoformat(ev['end_trade_date']))} across "
            f"{fmt.count(ev['session_count'])} sessions."
        ]
        yield build(
            family="asx_full_sample",
            slice_="asx_single",
            question=c.pick(templates).format(ticker=ticker),
            facts=facts,
            evidence=ev,
            components=["price_return_pct", "start_close", "end_close", "session_count"],
            datasets=["ASX"],
            difficulty="easy",
            split_key=f"asx_full:{ticker}",
            evidence_style=c.style(),
        )


@family("asx_single", "asx_volatility")
def _asx_vol(c: Ctx) -> Iterator[Example]:
    templates = [
        "What was {ticker}'s annualised volatility in {year}?",
        "Calculate {ticker}'s {year} annualised volatility and state the convention used.",
        "How volatile was {ticker} in {year} on an annualised basis?",
    ]
    for ticker in c.asx.tickers():
        for year in ASX_YEARS:
            ev = M.asx_volatility(c.asx, ticker, year)
            facts = [
                f"{ticker}'s annualised volatility in {year} was "
                f"{fmt.pct(ev['volatility_pct_annualised'])}, computed as the sample standard "
                f"deviation of {fmt.count(ev['daily_return_count'])} daily close-to-close "
                f"returns scaled by the square root of {ev['annualisation_factor']}."
            ]
            yield build(
                family="asx_volatility",
                slice_="asx_single",
                question=c.pick(templates).format(ticker=ticker, year=year),
                facts=facts,
                evidence=ev,
                components=["volatility_pct_annualised", "basis"],
                datasets=["ASX"],
                difficulty="medium",
                split_key=f"asx_vol:{ticker}:{year}",
                evidence_style=c.style(),
            )


@family("asx_single", "asx_correlation")
def _asx_corr(c: Ctx) -> Iterator[Example]:
    templates = [
        "What was the correlation of daily returns between {a} and {b} in {year}?",
        "How closely did {a} and {b} move together in {year}?",
        "Report the {year} daily-return correlation for {a} and {b}, and the number of paired observations.",
    ]
    tickers = c.asx.tickers()
    seen: set[tuple[str, str, int]] = set()
    for _ in range(240):
        a, b = c.rng.sample(tickers, 2)
        a, b = sorted((a, b))
        year = c.pick(ASX_YEARS)
        if (a, b, year) in seen:
            continue
        seen.add((a, b, year))
        ev = M.asx_correlation(c.asx, a, b, year)
        facts = [
            f"The correlation of daily price returns between {a} and {b} in {year} was "
            f"{fmt.corr(ev['correlation'])}, across "
            f"{fmt.count(ev['paired_return_count'])} paired observations."
        ]
        yield build(
            family="asx_correlation",
            slice_="asx_single",
            question=c.pick(templates).format(a=a, b=b, year=year),
            facts=facts,
            evidence=ev,
            components=["correlation", "paired_return_count"],
            datasets=["ASX"],
            difficulty="medium",
            split_key=f"asx_corr:{a}:{b}:{year}",
            evidence_style=c.style(),
            tolerance="corr_0.001",
        )


@family("asx_single", "asx_max_drawdown")
def _asx_dd(c: Ctx) -> Iterator[Example]:
    templates = [
        "What was {ticker}'s maximum drawdown over the sample, and between which dates?",
        "Report the worst peak-to-trough decline for {ticker}, with the peak and trough dates.",
        "How deep was {ticker}'s largest drawdown, and when did the peak and trough occur?",
    ]
    for ticker in c.asx.tickers():
        ev = M.asx_max_drawdown(c.asx, ticker)
        facts = [
            f"{ticker}'s maximum drawdown was "
            f"{fmt.signed_pct(ev['max_drawdown_pct'])}, from a closing peak on "
            f"{fmt.d(date.fromisoformat(ev['peak_date']))} to the trough on "
            f"{fmt.d(date.fromisoformat(ev['trough_date']))}."
        ]
        yield build(
            family="asx_max_drawdown",
            slice_="asx_single",
            question=c.pick(templates).format(ticker=ticker),
            facts=facts,
            evidence=ev,
            components=["max_drawdown_pct", "peak_date", "trough_date"],
            datasets=["ASX"],
            difficulty="medium",
            split_key=f"asx_dd:{ticker}",
            evidence_style=c.style(),
        )


@family("asx_single", "asx_avg_volume")
def _asx_vol_rank(c: Ctx) -> Iterator[Example]:
    """MHQ049 shape — a compound all-or-nothing component with a ±1-share tolerance."""
    templates = [
        "Excluding {excl}, which ASX ticker traded the highest average daily volume, and what was it?",
        "Which non-{excl} name has the largest average daily volume across the sample, and how many shares per trading day?",
        "Report the highest average daily volume among the ASX tickers, excluding {excl}.",
    ]
    for excl in ([TABCORP], []):
        ev = M.asx_rank_avg_volume(c.asx, exclude=excl)
        q_excl = f"{_company(TABCORP)} ({TABCORP})" if excl else "no tickers"
        fact = (
            f"{ev['highest_ticker']} has the highest average daily volume at "
            f"{fmt.volume(ev['highest_avg_daily_volume'])} shares per trading day."
        )
        for t in templates:
            yield build(
                family="asx_avg_volume",
                slice_="asx_single",
                question=t.format(excl=q_excl),
                facts=[fact],
                evidence=ev,
                components=["highest_ticker", "highest_avg_daily_volume"],
                datasets=["ASX"],
                difficulty="medium",
                split_key=f"asx_avgvol:{'excl' if excl else 'all'}",
                evidence_style=c.style(),
                tolerance="volume_1_share",
                notes="MHQ049 compound component",
            )
    for ticker in c.asx.tickers():
        ev = M.asx_avg_volume(c.asx, ticker)
        yield build(
            family="asx_avg_volume_single",
            slice_="asx_single",
            question=f"What is {ticker}'s average daily traded volume across the sample?",
            facts=[
                f"{ticker} averaged {fmt.volume(ev['avg_daily_volume'])} shares per trading "
                f"day across {fmt.count(ev['session_count'])} sessions."
            ],
            evidence=ev,
            components=["avg_daily_volume", "session_count"],
            datasets=["ASX"],
            difficulty="easy",
            split_key=f"asx_avgvol_single:{ticker}",
            evidence_style=c.style(),
            tolerance="volume_1_share",
        )


@family("asx_single", "asx_describe")
def _asx_describe(c: Ctx) -> Iterator[Example]:
    """MHQ040 shape — one compound component bundling file count, row count and range."""
    ev = M.asx_describe(c.asx)
    templates = [
        "Describe the ASX dataset: how many ticker files, how many rows each, and what date range?",
        "What does the ASX price data cover — number of files, rows per file, and the first and last trading dates?",
        "Summarise the shape of the ASX dataset in this package.",
    ]
    rows = ev["rows_per_file"]
    rows_txt = fmt.count(rows) if isinstance(rows, int) else "varying numbers of"
    fact = (
        f"There are {fmt.count(ev['ticker_file_count'])} ticker files, each containing "
        f"{rows_txt} rows, covering "
        f"{fmt.d(date.fromisoformat(ev['first_trade_date']))} through "
        f"{fmt.d(date.fromisoformat(ev['last_trade_date']))}."
    )
    for t in templates:
        yield build(
            family="asx_describe",
            slice_="asx_single",
            question=t,
            facts=[fact],
            evidence={k: v for k, v in ev.items() if k != "tickers"},
            components=["ticker_file_count", "rows_per_file", "date_range"],
            datasets=["ASX"],
            difficulty="easy",
            split_key="asx_describe",
            evidence_style=c.style(),
            notes="MHQ040 compound component",
        )


@family("asx_single", "asx_close_on")
def _asx_close(c: Ctx) -> Iterator[Example]:
    """Half of these land on non-trading days, so the answer must state the resolution
    rule instead of silently quoting a different date."""
    templates = [
        "What did {ticker} close at on {when}?",
        "Give {ticker}'s closing price on {when}.",
        "Report the {ticker} close for {when}, and say if the date had to be resolved.",
    ]
    for _ in range(260):
        ticker = c.pick(c.asx.tickers())
        year = c.pick(ASX_YEARS)
        when = date(year, c.rng.randrange(1, 13), c.rng.randrange(1, 29))
        ev = M.asx_close_on(c.asx, ticker, when)
        if ev["close"] is None:
            continue
        if ev["date_resolved"]:
            facts = [
                f"{ticker} did not trade on {fmt.d(when)}; the last close on or before that "
                f"date was {fmt.price(ev['close'])} on "
                f"{fmt.d(date.fromisoformat(ev['trade_date']))}."
            ]
        else:
            facts = [f"{ticker} closed at {fmt.price(ev['close'])} on {fmt.d(when)}."]
        yield build(
            family="asx_close_on",
            slice_="asx_single",
            question=c.pick(templates).format(ticker=ticker, when=fmt.d(when)),
            facts=facts,
            evidence=ev,
            components=["close", "trade_date"],
            datasets=["ASX"],
            difficulty="easy",
            split_key=f"asx_close:{ticker}:{when.isoformat()}",
            evidence_style=c.style(),
            tolerance="price_0.0001",
        )


# ======================================================================================
# AFR families
# ======================================================================================


@family("afr_single", "afr_term_count")
def _afr_count(c: Ctx) -> Iterator[Example]:
    from ftdata.corpora import AFR_VOCAB

    templates = [
        "How many AFR articles mention {term} as a whole word{span}?",
        "Count the AFR records containing the whole word {term}{span}.",
        "Using whole-word matching, how many AFR articles reference {term}{span}?",
        "How many times does {term} appear as a standalone word across the AFR corpus{span}?",
    ]
    years = c.afr.years
    for _ in range(420):
        term = c.pick(AFR_VOCAB)
        scope = c.rng.random()
        if scope < 0.4:
            year, span, key = None, "", f"afr_count:{term}:all"
        else:
            year = c.pick(years)
            span, key = f" in {year}", f"afr_count:{term}:{year}"
        ev = M.afr_term_count(c.afr, term, year)
        if ev["match_count"] == 0:
            continue
        facts = [
            f"There are {fmt.count(ev['match_count'])} AFR records matching whole-word "
            f"{term.upper() if len(term) <= 4 else term}{span}."
        ]
        yield build(
            family="afr_term_count",
            slice_="afr_single",
            question=c.pick(templates).format(
                term=term.upper() if len(term) <= 4 else term, span=span
            ),
            facts=facts,
            evidence=ev,
            components=["match_count"],
            datasets=["AFR"],
            difficulty="easy",
            split_key=key,
            evidence_style=c.style(),
        )


@family("afr_single", "afr_peak")
def _afr_peak(c: Ctx) -> Iterator[Example]:
    """MHQ061 shape — peak year and peak month together."""
    from ftdata.corpora import AFR_VOCAB

    templates = [
        "For AFR coverage of {term}, which year and which month had the most articles, and how many?",
        "When did AFR coverage of {term} peak — give the peak year and peak month with counts.",
        "Identify the busiest year and busiest month for {term} in the AFR data, with article counts.",
    ]
    for term in c.rng.sample(list(AFR_VOCAB), min(90, len(AFR_VOCAB))):
        ev = M.afr_peak_year_and_month(c.afr, term)
        if ev["match_count"] < 50:
            continue
        display = term.upper() if len(term) <= 4 else term
        fact = (
            f"AFR coverage of {display} peaked in {ev['peak_year']} with "
            f"{fmt.count(ev['peak_year_count'])} records, and the peak month was "
            f"{fmt.month_label_long(ev['peak_month'])} with "
            f"{fmt.count(ev['peak_month_count'])} records."
        )
        yield build(
            family="afr_peak",
            slice_="afr_single",
            question=c.pick(templates).format(term=display),
            facts=[fact],
            evidence={k: v for k, v in ev.items()},
            components=["peak_year", "peak_year_count", "peak_month", "peak_month_count"],
            datasets=["AFR"],
            difficulty="medium",
            split_key=f"afr_peak:{term}",
            evidence_style=c.style(),
            notes="MHQ061 compound component",
        )


@family("afr_single", "afr_share")
def _afr_share(c: Ctx) -> Iterator[Example]:
    from ftdata.corpora import AFR_VOCAB

    templates = [
        "What share of AFR articles in {year} mention {term}?",
        "In {year}, what percentage of AFR records reference {term}, and out of how many articles?",
        "Express {term} coverage in {year} as a share of all AFR records that year.",
    ]
    for _ in range(200):
        term = c.pick(AFR_VOCAB)
        year = c.pick(c.afr.years)
        ev = M.afr_share(c.afr, term, year)
        if not ev["match_count"] or ev["share_pct"] is None:
            continue
        display = term.upper() if len(term) <= 4 else term
        facts = [
            f"{fmt.count(ev['match_count'])} of the {fmt.count(ev['total_records'])} AFR "
            f"records published in {year} mention {display}, a share of "
            f"{fmt.pct(ev['share_pct'])}."
        ]
        yield build(
            family="afr_share",
            slice_="afr_single",
            question=c.pick(templates).format(term=display, year=year),
            facts=facts,
            evidence=ev,
            components=["match_count", "total_records", "share_pct"],
            datasets=["AFR"],
            difficulty="medium",
            split_key=f"afr_share:{term}:{year}",
            evidence_style=c.style(),
        )


@family("afr_single", "afr_monthly")
def _afr_monthly(c: Ctx) -> Iterator[Example]:
    from ftdata.corpora import AFR_VOCAB

    templates = [
        "Across {year}, which month had the most AFR articles about {term} and which had the fewest?",
        "Give the monthly high and low for {term} coverage in {year}, with counts and the annual total.",
    ]
    for _ in range(180):
        term = c.pick(AFR_VOCAB)
        year = c.pick(c.afr.years)
        ev = M.afr_monthly_series(c.afr, term, year)
        if ev["match_count"] < 24 or "peak_month" not in ev:
            continue
        display = term.upper() if len(term) <= 4 else term
        facts = [
            f"{display} coverage in {year} totalled {fmt.count(ev['match_count'])} AFR "
            f"records, peaking in {fmt.month_label_long(ev['peak_month'])} with "
            f"{fmt.count(ev['peak_month_count'])} and bottoming in "
            f"{fmt.month_label_long(ev['lowest_month'])} with "
            f"{fmt.count(ev['lowest_month_count'])}."
        ]
        yield build(
            family="afr_monthly",
            slice_="afr_single",
            question=c.pick(templates).format(term=display, year=year),
            facts=facts,
            evidence={k: v for k, v in ev.items() if k != "monthly_counts"},
            components=["match_count", "peak_month", "lowest_month"],
            datasets=["AFR"],
            difficulty="medium",
            split_key=f"afr_monthly:{term}:{year}",
            evidence_style=c.style(),
        )


@family("afr_single", "afr_pattern")
def _afr_pattern(c: Ctx) -> Iterator[Example]:
    from ftdata.corpora import AFR_REGEX_PATTERNS

    labels = {
        "rba_rate_pattern": "interest rates, the cash rate, rate cuts or hikes, or the RBA",
        "iron_ore": "iron ore",
        "royal_commission": "the royal commission",
        "reserve_bank": "the Reserve Bank",
        "profit_downgrade": "profit downgrades",
        "capital_raising": "capital raisings",
    }
    for name in AFR_REGEX_PATTERNS:
        for year in [None] + list(c.afr.years):
            ev = M.afr_pattern_count(c.afr, name, year)
            if ev["match_count"] == 0:
                continue
            span = "" if year is None else f" in {year}"
            facts = [
                f"{fmt.count(ev['match_count'])} AFR records mention "
                f"{labels[name]}{span}."
            ]
            yield build(
                family="afr_pattern",
                slice_="afr_single",
                question=f"How many AFR articles reference {labels[name]}{span}?",
                facts=facts,
                evidence=ev,
                components=["match_count"],
                datasets=["AFR"],
                difficulty="easy",
                split_key=f"afr_pattern:{name}:{year}",
                evidence_style=c.style(),
            )


@family("afr_single", "afr_describe")
def _afr_describe(c: Ctx) -> Iterator[Example]:
    ev = M.afr_describe(c.afr)
    templates = [
        "Describe the AFR dataset: how many files, how many articles, and what publication date range?",
        "What does the AFR corpus in this package contain — file count, article count, and coverage dates?",
    ]
    facts = [
        f"The AFR corpus spans {fmt.count(ev['file_count'])} files containing "
        f"{fmt.count(ev['record_count'])} articles, published between "
        f"{fmt.d(date.fromisoformat(ev['first_publication_date']))} and "
        f"{fmt.d(date.fromisoformat(ev['last_publication_date']))}."
    ]
    for t in templates:
        yield build(
            family="afr_describe",
            slice_="afr_single",
            question=t,
            facts=facts,
            evidence=ev,
            components=["file_count", "record_count", "date_range"],
            datasets=["AFR"],
            difficulty="easy",
            split_key="afr_describe",
            evidence_style=c.style(),
        )


# ======================================================================================
# cross-dataset families
# ======================================================================================


@family("cross", "cross_rba_window")
def _cross_rba_window(c: Ctx) -> Iterator[Example]:
    """RBA decision date -> ASX reaction. MHQ072/MHQ074 shape."""
    templates = [
        "After the RBA decision on {when}, how did {ticker} move over the following {n} trading sessions?",
        "Take the RBA decision effective {when}: what was {ticker}'s close-to-close return over the next {n} sessions?",
        "Measure {ticker}'s {n}-session return following the {when} RBA decision, and state the cash rate that decision set.",
    ]
    changes = [r for r in c.rba.changes() if 2015 <= r.effective_date.year <= 2021]
    seen_w: set[tuple[str, str, int]] = set()
    for _ in range(1200):
        rec = c.pick(changes)
        ticker = c.pick(c.asx.tickers())
        n = c.pick([5, 10, 20, 30])
        if (rec.iso, ticker, n) in seen_w:
            continue
        seen_w.add((rec.iso, ticker, n))
        series = c.asx.get(ticker)
        a = series.on_or_after(rec.effective_date)
        if a is None:
            continue
        idx = series.dates.index(a.trade_date)
        if idx + n >= len(series.bars):
            continue
        b = series.bars[idx + n]
        ev_asx = M.asx_window_return(c.asx, ticker, a.trade_date, b.trade_date)
        ev_rba = M.rba_lookup_rate(c.rba, rec.effective_date)
        if ev_asx.get("price_return_pct") is None:
            continue
        ev = {"rba_decision": ev_rba, "asx_window": ev_asx, "session_count": n}
        facts = [
            f"The RBA decision effective {rec.display} set the cash rate target at "
            f"{fmt.rate_pct(rec.cash_rate_target_pct)}, a move of "
            f"{fmt.pp(rec.change_pp)}.",
            f"Over the {n} trading sessions from {fmt.d(a.trade_date)} to "
            f"{fmt.d(b.trade_date)}, {ticker} returned "
            f"{fmt.signed_pct(ev_asx['price_return_pct'])} on a price basis, from "
            f"{fmt.price(a.close)} to {fmt.price(b.close)}.",
        ]
        yield build(
            family="cross_rba_window",
            slice_="cross",
            question=c.pick(templates).format(
                when=rec.display, ticker=ticker, n=n
            ),
            facts=facts,
            evidence=ev,
            components=["cash_rate_target_pct", "price_return_pct"],
            datasets=["RBA", "ASX"],
            difficulty="hard",
            split_key=f"cross_rba_window:{ticker}:{rec.iso}:{n}",
            evidence_style=c.style(),
        )


@family("cross", "cross_basket_window")
def _cross_basket(c: Ctx) -> Iterator[Example]:
    templates = [
        "Following the RBA decision on {when}, what was the equal-weighted return of the ASX names excluding {excl} over the next {n} sessions?",
        "Compute the equal-weighted basket return (excluding {excl}) for the {n} sessions after the {when} RBA decision.",
    ]
    changes = [r for r in c.rba.changes() if 2015 <= r.effective_date.year <= 2021]
    for _ in range(120):
        rec = c.pick(changes)
        n = c.pick([5, 10, 20])
        ref = c.asx.get("BHP.AX")
        a = ref.on_or_after(rec.effective_date)
        if a is None:
            continue
        idx = ref.dates.index(a.trade_date)
        if idx + n >= len(ref.bars):
            continue
        b = ref.bars[idx + n]
        ev = M.asx_basket_window_return(c.asx, a.trade_date, b.trade_date)
        if ev["basket_return_pct"] is None:
            continue
        ev_rba = M.rba_lookup_rate(c.rba, rec.effective_date)
        merged = {"rba_decision": ev_rba, "asx_basket": ev}
        facts = [
            f"With the cash rate target at "
            f"{fmt.rate_pct(rec.cash_rate_target_pct)} from {rec.display}, the "
            f"equal-weighted basket of {ev['basket_size']} ASX names excluding "
            f"{_company(TABCORP)} ({TABCORP}) returned "
            f"{fmt.signed_pct(ev['basket_return_pct'])} over the {n} sessions from "
            f"{fmt.d(a.trade_date)} to {fmt.d(b.trade_date)}."
        ]
        yield build(
            family="cross_basket_window",
            slice_="cross",
            question=c.pick(templates).format(
                when=rec.display, excl=f"{_company(TABCORP)} ({TABCORP})", n=n
            ),
            facts=facts,
            evidence={
                "rba_decision": ev_rba,
                "asx_basket": {
                    k: v for k, v in ev.items() if k != "constituent_returns_pct"
                },
            },
            components=["basket_return_pct", "basket_size"],
            datasets=["RBA", "ASX"],
            difficulty="hard",
            split_key=f"cross_basket:{rec.iso}:{n}",
            evidence_style=c.style(),
            notes="equal-weighted = average of constituent returns, not return of an index",
        )


@family("cross", "cross_afr_asx")
def _cross_afr_asx(c: Ctx) -> Iterator[Example]:
    """MHQ076 shape — AFR count and best non-Tabcorp return in one compound fact."""
    ticker_terms = {
        "AGL.AX": "agl", "AMP.AX": "amp", "ANZ.AX": "anz", "BHP.AX": "bhp",
        "CBA.AX": "cba", "IAG.AX": "iag", "NAB.AX": "nab", "QAN.AX": "qan",
        "QBE.AX": "qbe", "RIO.AX": "rio", "SUN.AX": "sun", "TCL.AX": "tcl",
        "GPT.AX": "gpt", "SGP.AX": "sgp",
    }
    templates = [
        "How many AFR records mention whole-word {term} in {year}, and which non-{excl} ticker had the best {year} return?",
        "For {year}: count AFR articles referencing {term}, and identify the best-performing ASX name excluding {excl}.",
        "Combine the datasets for {year} — AFR mentions of {term}, plus the top non-{excl} price return.",
    ]
    for ticker, term in ticker_terms.items():
        for year in ASX_YEARS:
            ev_afr = M.afr_term_count(c.afr, term, year)
            ev_asx = M.asx_rank_annual_returns(c.asx, year, exclude=[TABCORP])
            if ev_afr["match_count"] == 0:
                continue
            display = term.upper()
            fact = (
                f"There are {fmt.count(ev_afr['match_count'])} AFR records matching "
                f"whole-word {display} in {year}, and {ev_asx['best_ticker']} had the best "
                f"non-{_company(TABCORP)} {year} return at "
                f"{fmt.signed_pct(ev_asx['best_price_return_pct'])}."
            )
            yield build(
                family="cross_afr_asx",
                slice_="cross",
                question=c.pick(templates).format(
                    term=display, year=year, excl=f"{_company(TABCORP)} ({TABCORP})"
                ),
                facts=[fact],
                evidence={
                    "afr": ev_afr,
                    "asx": {
                        k: v for k, v in ev_asx.items() if k != "ranked_returns_pct"
                    },
                },
                components=["match_count", "best_ticker", "best_price_return_pct"],
                datasets=["AFR", "ASX"],
                difficulty="hard",
                split_key=f"cross_afr_asx:{term}:{year}",
                evidence_style=c.style(),
                notes="MHQ076 compound component",
            )


@family("cross", "cross_triple")
def _cross_triple(c: Ctx) -> Iterator[Example]:
    """All three corpora at once: policy setting, market outcome, media coverage."""
    templates = [
        "For {year}: what did the RBA do, which ASX name led excluding {excl}, and how many AFR articles discussed interest rates?",
        "Give a {year} wrap across all three datasets — RBA policy, best non-{excl} ASX return, and AFR rates coverage.",
    ]
    for year in ASX_YEARS:
        ev_rba = M.rba_year_summary(c.rba, year)
        ev_asx = M.asx_rank_annual_returns(c.asx, year, exclude=[TABCORP])
        ev_afr = M.afr_pattern_count(c.afr, "rba_rate_pattern", year)
        facts = [
            f"In {year} the RBA made {ev_rba['cut_count']} cuts and "
            f"{ev_rba['hike_count']} increases, a net "
            f"{fmt.pp(ev_rba['cumulative_change_pp'])}, ending the year at "
            f"{fmt.rate_pct(ev_rba['year_end_cash_rate_target_pct'])}.",
            f"{ev_asx['best_ticker']} led the ASX names excluding "
            f"{_company(TABCORP)} ({TABCORP}) with a {year} price return of "
            f"{fmt.signed_pct(ev_asx['best_price_return_pct'])}.",
            f"{fmt.count(ev_afr['match_count'])} AFR records published in {year} discussed "
            f"interest rates, the cash rate, rate moves or the RBA.",
        ]
        yield build(
            family="cross_triple",
            slice_="cross",
            question=c.pick(templates).format(
                year=year, excl=f"{_company(TABCORP)} ({TABCORP})"
            ),
            facts=facts,
            evidence={
                "rba": ev_rba,
                "asx": {k: v for k, v in ev_asx.items() if k != "ranked_returns_pct"},
                "afr": ev_afr,
            },
            components=["rba_year_moves", "best_ticker", "afr_match_count"],
            datasets=["RBA", "ASX", "AFR"],
            difficulty="hard",
            split_key=f"cross_triple:{year}",
            evidence_style=c.style(),
        )


# ======================================================================================
# sentiment slice — the adapter's second role
# ======================================================================================

_POS = ("rally", "rallied", "surge", "surged", "gains", "gained", "jumped", "soared",
        "upgrade", "record high", "beat", "profit rose", "rebound", "optimism", "boost")
_NEG = ("selloff", "sell-off", "slump", "slumped", "plunge", "plunged", "fell", "losses",
        "downgrade", "warning", "profit fell", "writedown", "fears", "recession", "cut jobs")


def _classify(text: str) -> tuple[str, str]:
    low = text.lower()
    pos = sum(low.count(w) for w in _POS)
    neg = sum(low.count(w) for w in _NEG)
    if pos > neg * 1.5 and pos >= 2:
        return "Positive", "higher"
    if neg > pos * 1.5 and neg >= 2:
        return "Negative", "lower"
    return "Mixed", "flat"


@family("sentiment", "afr_sentiment")
def _sentiment(c: Ctx) -> Iterator[Example]:
    """Role 2: the ``domain_sentiment`` tool Qwen calls mid-loop.

    The label is derived from a deterministic lexicon rather than a model, so the target
    is reproducible and contains no hidden LLM-generated ground truth. Answers are held
    under the 200-character tool clamp *by construction* — a target that would be clamped
    teaches the model to lose its own direction clause.
    """
    for rec in c.afr.sentiment_pool:
        headline = rec.get("headline") or ""
        pub = str(rec.get("publication_date") or "")
        excerpt = (rec.get("intro") or rec.get("text") or "")[:600]
        if len(pub) != 8 or not headline or len(excerpt) < 120:
            continue
        when = date(int(pub[:4]), int(pub[4:6]), int(pub[6:]))
        rate_ev = M.rba_lookup_rate(c.rba, when)
        if rate_ev["cash_rate_target_pct"] is None:
            continue
        label, direction = _classify(f"{headline}\n{excerpt}")
        rate_txt = fmt.rate_pct(rate_ev["cash_rate_target_pct"])
        answer = (
            f"{label}. With the cash rate at {rate_txt}, the reporting points to "
            f"{direction} near-term prices for the names discussed."
        )
        if len(answer) > SENTIMENT_CHAR_CAP:
            answer = f"{label}. Cash rate {rate_txt}; near-term direction {direction}."
        ev = {
            "headline": headline,
            "publication_date": fmt.ymd_iso(pub),
            "article_excerpt": excerpt,
            "cash_rate_target_pct": rate_ev["cash_rate_target_pct"],
            "rate_effective_date": rate_ev["effective_date"],
        }
        ex = build(
            family="afr_sentiment",
            slice_="sentiment",
            question=f"Classify the sentiment of the AFR article '{headline}' "
                     f"({fmt.ymd_display(pub)}) toward Australian equities, and give the "
                     f"likely near-term direction.",
            facts=[answer],
            evidence=ev,
            components=["sentiment_label", "likely_direction"],
            datasets=["AFR", "RBA"],
            difficulty="medium",
            split_key=f"sentiment:{pub}:{headline[:48]}",
            evidence_style="json",
            notes="Role 2 (domain_sentiment tool); no numeric forecast, <=200 chars",
            max_score=4.0,
        )
        yield ex


# ======================================================================================
# insufficient slice — the justified refusal (MHQ090 shape)
# ======================================================================================


@family("insufficient", "coverage_refusal")
def _insufficient(c: Ctx) -> Iterator[Example]:
    """MHQ090's correct answer is a *refusal with reasoning*. Most of the points sit in
    naming which corpus ends when, not in the verdict."""
    templates = [
        "How did the ASX names respond to the RBA tightening cycle between {start} and {end}?",
        "Analyse AFR coverage and ASX returns across the {start} to {end} period.",
        "What was the equity-market reaction to RBA policy between {start} and {end}?",
        "Compare ASX performance with AFR sentiment from {start} to {end}.",
        "Which ASX sectors led between {start} and {end}, and what did the AFR say about them?",
        "Report the correlation between AFR rate coverage and ASX returns from {start} to {end}.",
        "Summarise Australian equity performance and media sentiment over {start} to {end}.",
        "Did the ASX rally or sell off between {start} and {end}, and how did the AFR frame it?",
    ]
    # Windows that fall wholly or partly outside at least one corpus. The post-2021
    # windows are the important ones: RBA runs to Jun 2026 while ASX and AFR both stop
    # in 2021, which is exactly MHQ090's trap — the question looks answerable because one
    # dataset does cover it.
    windows: list[tuple[date, date]] = []
    seen_w: set[tuple[str, str]] = set()
    for _ in range(500):
        mode = c.rng.random()
        if mode < 0.6:                                  # after ASX/AFR end
            ys, ye = c.rng.randrange(2022, 2026), c.rng.randrange(2022, 2027)
        elif mode < 0.85:                               # straddles the 2021 cliff
            ys, ye = c.rng.randrange(2018, 2022), c.rng.randrange(2022, 2027)
        else:                                           # before ASX/AFR start
            ys, ye = c.rng.randrange(2010, 2013), c.rng.randrange(2013, 2016)
        if ye < ys:
            continue
        start = date(ys, c.rng.randrange(1, 13), 1)
        end = date(ye, c.rng.randrange(1, 13), 28)
        if end <= start:
            continue
        k = (start.isoformat(), end.isoformat())
        if k in seen_w:
            continue
        seen_w.add(k)
        windows.append((start, end))

    subsets = (["RBA", "ASX", "AFR"], ["ASX", "AFR"], ["RBA", "ASX"], ["RBA", "AFR"])
    for start, end in windows:
        datasets = c.pick(subsets)
        ev = M.coverage_check(datasets, start, end)
        if ev["fully_supported"]:
            continue
        unsupported = ev["unsupported_datasets"]
        spans = ev["dataset_coverage"]
        supported = [d for d in datasets if d not in unsupported]
        detail = "; ".join(
            f"{n} covers {fmt.d(date.fromisoformat(spans[n]['first_date']))} to "
            f"{fmt.d(date.fromisoformat(spans[n]['last_date']))}"
            for n in unsupported
        )
        facts = [
            f"This cannot be determined from the supplied datasets for "
            f"{fmt.d(start)} to {fmt.d(end)}, because {detail}.",
            (
                f"Only the {fmt.ticker_list(supported)} data extends across the requested "
                f"window; the {fmt.ticker_list(unsupported)} data does not, so no "
                f"comparison is possible for that period."
                if supported
                else
                f"None of the {fmt.ticker_list(unsupported)} data covers the requested "
                f"window, so no comparison is possible for that period."
            ),
        ]
        yield build(
            family="coverage_refusal",
            slice_="insufficient",
            question=c.pick(templates).format(start=fmt.d(start), end=fmt.d(end)),
            facts=facts,
            evidence=ev,
            components=["insufficient_coverage", "which_datasets"],
            datasets=datasets,
            difficulty="hard",
            split_key=f"insufficient:{start.isoformat()}:{end.isoformat()}",
            evidence_style=c.style(),
            limitations=[
                {
                    "component": n,
                    "message": f"{n} coverage ends "
                               f"{fmt.d(date.fromisoformat(spans[n]['last_date']))}.",
                }
                for n in unsupported
            ],
            notes="MHQ090 shape: refusal WITH reasoning",
        )


# ======================================================================================
# robust slice — tool errors, empty results, over-supplied evidence
# ======================================================================================


@family("robust", "robust_over_supplied")
def _robust_over(c: Ctx) -> Iterator[Example]:
    """Evidence carries far more than the question asks for. The brief is explicit: "the
    grader requires only information requested by the prompt." Over-answering buries the
    component that is actually being scored."""
    templates = [
        "What was {ticker}'s price return in {year}? Answer only that.",
        "Report just {ticker}'s {year} return — nothing else.",
        "I need only one number: {ticker}'s {year} price return.",
    ]
    for _ in range(200):
        ticker = c.pick(c.asx.tickers())
        year = c.pick(ASX_YEARS)
        ev_ret = M.asx_annual_return(c.asx, ticker, year)
        if ev_ret.get("price_return_pct") is None:
            continue
        # deliberately over-supply
        ev = {
            "annual_return": ev_ret,
            "volatility": M.asx_volatility(c.asx, ticker, year),
            "max_drawdown": M.asx_max_drawdown(c.asx, ticker, year),
            "avg_volume": M.asx_avg_volume(c.asx, ticker),
        }
        facts = [
            f"{ticker} returned {fmt.signed_pct(ev_ret['price_return_pct'])} over {year} "
            f"on a price basis."
        ]
        yield build(
            family="robust_over_supplied",
            slice_="robust",
            question=c.pick(templates).format(ticker=ticker, year=year),
            facts=facts,
            evidence=ev,
            components=["price_return_pct"],
            datasets=["ASX"],
            difficulty="medium",
            split_key=f"robust_over:{ticker}:{year}",
            evidence_style=c.style(),
            notes="answer only what was asked, despite extra evidence",
        )


@family("robust", "robust_tool_error")
def _robust_error(c: Ctx) -> Iterator[Example]:
    """A tool failed or returned nothing. The answer must say exactly what is missing —
    not hedge, and not guess a plausible number."""
    cases = [
        (
            "What did {ticker} close at on {when}?",
            {"error": "no_data", "ticker": "{ticker}", "query_date": "{iso}",
             "message": "requested date is outside the ASX sample"},
            "The {ticker} close on {when} cannot be determined: that date lies outside "
            "the ASX sample, which ends 30 Dec 2021.",
        ),
        (
            "How many AFR articles mentioned {term} in {far_year}?",
            {"error": "out_of_range", "search_term": "{term}", "year": "{far_year}",
             "message": "AFR coverage ends 2021"},
            "The count of AFR articles mentioning {term} in {far_year} cannot be "
            "determined: AFR coverage ends in 2021.",
        ),
        (
            "What was the RBA cash rate on {early}?",
            {"error": "before_coverage", "query_date": "{early_iso}",
             "message": "RBA series begins 3 Feb 2010"},
            "The RBA cash rate target on {early} cannot be determined: the series begins "
            "3 Feb 2010.",
        ),
    ]
    for _ in range(120):
        q_t, ev_t, a_t = c.pick(cases)
        ticker = c.pick(c.asx.tickers())
        far_year = c.pick([2022, 2023, 2024, 2025])
        when = date(far_year, c.rng.randrange(1, 13), c.rng.randrange(1, 28))
        early = date(c.rng.randrange(1990, 2010), c.rng.randrange(1, 13), 15)
        term = c.pick(["unemployment", "inflation", "dividend", "housing"])
        subs = {
            "ticker": ticker, "when": fmt.d(when), "iso": when.isoformat(),
            "far_year": far_year, "term": term,
            "early": fmt.d(early), "early_iso": early.isoformat(),
        }
        ev = {k: (v.format(**subs) if isinstance(v, str) else v) for k, v in ev_t.items()}
        facts = [a_t.format(**subs)]
        yield build(
            family="robust_tool_error",
            slice_="robust",
            question=q_t.format(**subs),
            facts=facts,
            evidence=ev,
            components=["cannot_determine", "reason"],
            datasets=["ASX"],
            difficulty="medium",
            split_key=f"robust_error:{ev_t['error']}:{subs['iso'] if 'iso' in ev_t.get('query_date','') else far_year}:{ticker}",
            evidence_style=c.style(),
            limitations=[{"component": "coverage", "message": ev["message"]}],
            notes="state what is missing; never guess a plausible number",
        )


# ======================================================================================
# assembly
# ======================================================================================

MIX = {
    "rba_single": 0.15,
    "asx_single": 0.18,
    "afr_single": 0.12,
    "cross": 0.25,
    "sentiment": 0.12,
    "insufficient": 0.10,
    "robust": 0.08,
}


def generate(ctx: Ctx, target: int) -> list[Example]:
    """Sample each slice up to its quota, round-robin across the slice's families so no
    single family dominates when one is far more prolific than another."""
    out: list[Example] = []
    report: dict[str, dict[str, int]] = {}
    for slice_, share in MIX.items():
        quota = max(1, int(round(target * share)))
        pools: list[list[Example]] = []
        for name, fn in _REGISTRY.get(slice_, []):
            got = list(fn(ctx))
            ctx.rng.shuffle(got)
            pools.append(got)
            report.setdefault(slice_, {})[name] = len(got)
        picked: list[Example] = []
        i = 0
        while len(picked) < quota and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                picked.append(pool.pop())
            else:
                pools = [p for p in pools if p]
                if not pools:
                    break
                continue
            i += 1
        if len(picked) < quota:
            print(
                f"  ! slice {slice_}: only {len(picked)} of {quota} available "
                f"(families exhausted)"
            )
        out.extend(picked)

    print("\n  family yields (available before quota):")
    for slice_, fams in report.items():
        for name, n in sorted(fams.items(), key=lambda kv: -kv[1]):
            print(f"    {slice_:<13} {name:<26} {n:>6}")

    ctx.rng.shuffle(out)
    for i, ex in enumerate(out):
        ex.example_id = f"gen_{ex.slice}_{i:06d}"
    return out


def split(examples: list[Example], rng: random.Random) -> dict[str, list[Example]]:
    """80/10/10 grouped by ``split_key``.

    Assignment is on the *key*, so every row sharing an entity key lands in the same
    split. Row-level shuffling would put BHP.AX-2018 in both train and heldout and
    silently inflate the comparison.
    """
    keys = sorted({ex.split_key for ex in examples})
    rng.shuffle(keys)
    n = len(keys)
    n_val = max(1, int(n * 0.10))
    n_held = max(1, int(n * 0.10))
    assign = {}
    for k in keys[:n_held]:
        assign[k] = "heldout"
    for k in keys[n_held:n_held + n_val]:
        assign[k] = "val"
    for k in keys[n_held + n_val:]:
        assign[k] = "train"

    out: dict[str, list[Example]] = {"train": [], "val": [], "heldout": []}
    for ex in examples:
        out[assign[ex.split_key]].append(ex)

    # -- the leakage assertion, logged because the rubric asks for the evidence ---------
    key_sets = {s: {ex.split_key for ex in rows} for s, rows in out.items()}
    for a, b in (("train", "val"), ("train", "heldout"), ("val", "heldout")):
        overlap = key_sets[a] & key_sets[b]
        assert not overlap, f"SPLIT LEAKAGE {a}/{b}: {sorted(overlap)[:5]}"
    print(
        f"\n  split keys: {n} total -> train {len(key_sets['train'])}, "
        f"val {len(key_sets['val'])}, heldout {len(key_sets['heldout'])}"
    )
    print("  leakage assertion passed: no split_key appears in two splits")
    return out


def to_chat_record(ex: Example) -> dict:
    """Chat-format training row. ``meta`` is carried but never trained on — the evaluator
    reads it to score points-weighted component recall."""
    if ex.slice == "sentiment":
        messages = render_sentiment_messages(
            headline=ex.verified_evidence["headline"],
            publication_date=ex.verified_evidence["publication_date"],
            excerpt=ex.verified_evidence["article_excerpt"],
            cash_rate=fmt.rate_pct(ex.verified_evidence["cash_rate_target_pct"]),
        )
    else:
        messages = render_synthesis_messages(
            ex.question,
            ex.requested_components,
            ex.verified_evidence,
            ex.limitations,
            evidence_style=ex.evidence_style,
        )
    messages = messages + [{"role": "assistant", "content": ex.expected_answer}]
    return {
        "messages": messages,
        "meta": {
            "id": ex.example_id,
            "slice": ex.slice,
            "family": ex.question_family,
            "datasets": ex.datasets,
            "question": ex.question,
            "required_facts": ex.required_facts,
            "points": ex.points,
            "tolerance": ex.tolerance,
            "split_key": ex.split_key,
            "difficulty": ex.difficulty,
            "evidence": ex.verified_evidence,
        },
    }


def verify_invariants(examples: list[Example]) -> None:
    """Every required_fact must appear verbatim in the answer, and no answer may hedge."""
    import re

    hedge = re.compile(r"\b(approximately|roughly|about|around)\s+[\d$+-]|~\d")
    bad_fact = bad_hedge = 0
    for ex in examples:
        for f in ex.required_facts:
            if f not in ex.expected_answer:
                bad_fact += 1
        if hedge.search(ex.expected_answer):
            bad_hedge += 1
    if bad_fact or bad_hedge:
        raise SystemExit(
            f"INVARIANT FAILURE: {bad_fact} facts not verbatim in answer, "
            f"{bad_hedge} answers hedge before a number"
        )
    print(f"  invariants passed on {len(examples)} examples: "
          f"facts verbatim, zero hedging")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=4000, help="approximate example count")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-gate", action="store_true",
                    help="debugging only; never use for a real run")
    args = ap.parse_args()

    paths.ensure_dirs()

    print("== blocking correctness gate (§6.5) ==")
    if args.skip_gate:
        print("  SKIPPED — this output is not valid training data")
    else:
        assert_gate()

    print("\n== loading corpora ==")
    corpora = M.load_all(progress=False)
    print(f"  RBA {len(corpora.rba.records)} records | "
          f"ASX {corpora.asx.file_count} files | "
          f"AFR {corpora.afr.total_articles} articles "
          f"({len(corpora.afr.sentiment_pool)} in sentiment pool)")

    rng = random.Random(args.seed)
    ctx = Ctx(corpora, rng)

    print(f"\n== generating (~{args.target}, seed {args.seed}) ==")
    examples = generate(ctx, args.target)

    print(f"\n== invariants ==")
    verify_invariants(examples)

    print("\n== splitting ==")
    splits = split(examples, random.Random(args.seed))

    print("\n== writing ==")
    counts: dict[str, dict[str, int]] = {}
    for name, rows in splits.items():
        fp = paths.PREPARED_DIR / f"{name}.jsonl"
        with fp.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(to_chat_record(ex), ensure_ascii=False) + "\n")
        by_slice: dict[str, int] = {}
        for ex in rows:
            by_slice[ex.slice] = by_slice.get(ex.slice, 0) + 1
        counts[name] = by_slice
        print(f"  {fp.relative_to(paths.REPO_ROOT)}: {len(rows)} rows")

    stats = {
        "seed": args.seed,
        "target": args.target,
        "total": len(examples),
        "counts_by_split_and_slice": counts,
        "mix_requested": MIX,
        "families": sorted({ex.question_family for ex in examples}),
    }
    (paths.PREPARED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\n  slice mix (all splits):")
    total = len(examples)
    agg: dict[str, int] = {}
    for ex in examples:
        agg[ex.slice] = agg.get(ex.slice, 0) + 1
    for s, n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<14} {n:>5}  {n / total * 100:5.1f}%  (target "
              f"{MIX.get(s, 0) * 100:.0f}%)")
    print(f"\n  wrote {total} examples to {paths.PREPARED_DIR}")


if __name__ == "__main__":
    main()
