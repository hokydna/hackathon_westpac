"""Deterministic loaders for the three supplied corpora.

Everything the generator knows about the data lives here. Three gotchas are handled
once, in this file, and nowhere else:

* ``RBA-rates.csv`` carries a UTF-8 BOM and stores every value as a string, including
  signed changes (``"+0.25"``, ``"0.00"``). Dates are ``"3 Feb 2010"``, not ISO.
* ASX filenames use company names (``Aurizon-ASX-2015-2021.jsonl``) while the ``ticker``
  field inside carries the real code (``AZJ.AX``). Always trust the field.
* ``PUBLICATIONDATE`` is a ``YYYYMMDD`` string. Slice it; never date-parse it.

The AFR search rule is fixed by ``Setup_Instructions.md``: case-insensitive, word-boundary
anchored, matched across ``HEADLINE + SUBHEAD + INTRO + TEXT`` combined, counted once per
record. ``token_counts`` implements that rule through an ``[a-z0-9]+`` tokenisation, which
is exactly equivalent (see ``AfrIndex`` docstring).
"""

from __future__ import annotations

import csv
import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Iterable, Iterator, Sequence

from . import paths

# --------------------------------------------------------------------------------------
# RBA
# --------------------------------------------------------------------------------------

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_rba_date(raw: str) -> date:
    """``"3 Feb 2010"`` -> ``date(2010, 2, 3)``."""
    day, mon, year = raw.split()
    return date(int(year), _MONTHS[mon[:3]], int(day))


def display_date(d: date) -> str:
    """ISO date -> the RBA/question-bank display form ``"3 Feb 2010"``."""
    return f"{d.day} {d.strftime('%b')} {d.year}"


@dataclass(frozen=True)
class RbaRecord:
    effective_date: date
    display: str
    change_pp: float
    cash_rate_target_pct: float

    @property
    def iso(self) -> str:
        return self.effective_date.isoformat()


@dataclass
class RbaCorpus:
    records: list[RbaRecord]

    @property
    def dates(self) -> list[date]:
        return [r.effective_date for r in self.records]

    def changes(self, sign: int | None = None) -> list[RbaRecord]:
        """Non-zero decisions; ``sign`` +1 for increases, -1 for decreases."""
        out = [r for r in self.records if r.change_pp != 0.0]
        if sign is not None:
            out = [r for r in out if (r.change_pp > 0) == (sign > 0)]
        return out

    def in_range(self, start: date | None = None, end: date | None = None) -> list[RbaRecord]:
        return [
            r for r in self.records
            if (start is None or r.effective_date >= start)
            and (end is None or r.effective_date <= end)
        ]

    def in_year(self, year: int) -> list[RbaRecord]:
        return [r for r in self.records if r.effective_date.year == year]

    def as_of(self, when: date) -> RbaRecord | None:
        """Rate *in force* on ``when``: the latest decision on or before that date.

        As-of, never nearest-match. Nearest-match can return a future decision, which is
        the single most common wrong answer on rate-lookup questions.
        """
        idx = bisect_right(self.dates, when) - 1
        return self.records[idx] if idx >= 0 else None


@lru_cache(maxsize=1)
def load_rba() -> RbaCorpus:
    records: list[RbaRecord] = []
    with open(paths.RBA_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_date = row["Effective Date"].strip()
            if not raw_date:
                continue
            d = parse_rba_date(raw_date)
            records.append(
                RbaRecord(
                    effective_date=d,
                    display=raw_date,
                    change_pp=float(row["Change % points"]),
                    cash_rate_target_pct=float(row["Cash rate target%"]),
                )
            )
    records.sort(key=lambda r: r.effective_date)
    return RbaCorpus(records)


# --------------------------------------------------------------------------------------
# ASX
# --------------------------------------------------------------------------------------

TABCORP = "TAH.AX"

#: Company label used in filenames and natural-language questions -> real ticker.
ASX_NAMES = {
    "AGL": "AGL.AX",
    "AMP": "AMP.AX",
    "ANZ": "ANZ.AX",
    "Aurizon": "AZJ.AX",
    "BHP": "BHP.AX",
    "CBA": "CBA.AX",
    "Cromwell": "CMW.AX",
    "GPT": "GPT.AX",
    "IAG": "IAG.AX",
    "NAB": "NAB.AX",
    "Qantas": "QAN.AX",
    "QBE": "QBE.AX",
    "Rio": "RIO.AX",
    "Stockland": "SGP.AX",
    "Suncorp": "SUN.AX",
    "Tabcorp": "TAH.AX",
    "TPG": "TPG.AX",
    "Transurban": "TCL.AX",
}
TICKER_TO_NAME = {v: k for k, v in ASX_NAMES.items()}


@dataclass(frozen=True)
class Bar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def iso(self) -> str:
        return self.trade_date.isoformat()


@dataclass
class AsxSeries:
    ticker: str
    bars: list[Bar]

    @property
    def dates(self) -> list[date]:
        return [b.trade_date for b in self.bars]

    def on(self, when: date) -> Bar | None:
        """Bar for an exact trading date, or ``None`` if the market was closed."""
        idx = bisect_right(self.dates, when) - 1
        if idx < 0:
            return None
        bar = self.bars[idx]
        return bar if bar.trade_date == when else None

    def on_or_before(self, when: date) -> Bar | None:
        idx = bisect_right(self.dates, when) - 1
        return self.bars[idx] if idx >= 0 else None

    def on_or_after(self, when: date) -> Bar | None:
        from bisect import bisect_left

        idx = bisect_left(self.dates, when)
        return self.bars[idx] if idx < len(self.bars) else None

    def in_year(self, year: int) -> list[Bar]:
        return [b for b in self.bars if b.trade_date.year == year]

    def between(self, start: date, end: date) -> list[Bar]:
        return [b for b in self.bars if start <= b.trade_date <= end]


@dataclass
class AsxCorpus:
    series: dict[str, AsxSeries]
    file_count: int
    rows_per_file: dict[str, int]

    def tickers(self, exclude: Sequence[str] = ()) -> list[str]:
        excl = {_norm_ticker(t) for t in exclude}
        return sorted(t for t in self.series if t not in excl)

    def basket(self, exclude: Sequence[str] = (TABCORP,)) -> list[str]:
        return self.tickers(exclude=exclude)

    def get(self, ticker: str) -> AsxSeries:
        return self.series[_norm_ticker(ticker)]

    @property
    def date_min(self) -> date:
        return min(s.bars[0].trade_date for s in self.series.values())

    @property
    def date_max(self) -> date:
        return max(s.bars[-1].trade_date for s in self.series.values())


def _norm_ticker(t: str) -> str:
    """Accept ``BHP``, ``bhp``, ``BHP.AX`` or ``Qantas`` and return ``BHP.AX``."""
    t = t.strip()
    if t in ASX_NAMES:
        return ASX_NAMES[t]
    up = t.upper()
    if up in TICKER_TO_NAME:
        return up
    if up + ".AX" in TICKER_TO_NAME:
        return up + ".AX"
    title = t.title()
    if title in ASX_NAMES:
        return ASX_NAMES[title]
    raise KeyError(f"unknown ASX ticker or name: {t!r}")


@lru_cache(maxsize=1)
def load_asx() -> AsxCorpus:
    series: dict[str, list[Bar]] = {}
    rows_per_file: dict[str, int] = {}
    files = sorted(paths.ASX_DIR.glob("*-ASX-*.jsonl"))
    for fp in files:
        count = 0
        for line in fp.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            y, m, d = rec["date"].split("-")
            series.setdefault(rec["ticker"], []).append(
                Bar(
                    trade_date=date(int(y), int(m), int(d)),
                    open=float(rec["open"]),
                    high=float(rec["high"]),
                    low=float(rec["low"]),
                    close=float(rec["close"]),
                    volume=int(rec["volume"]),
                )
            )
            count += 1
        rows_per_file[fp.name] = count
    built = {}
    for ticker, bars in series.items():
        bars.sort(key=lambda b: b.trade_date)
        built[ticker] = AsxSeries(ticker, bars)
    return AsxCorpus(series=built, file_count=len(files), rows_per_file=rows_per_file)


# --------------------------------------------------------------------------------------
# AFR
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Multi-token / alternation searches that cannot be expressed as a single token.
#: Kept deliberately small — each one is a real regex pass over 780 MB.
AFR_REGEX_PATTERNS: dict[str, str] = {
    "rba_rate_pattern": r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b",
    "iron_ore": r"\biron ore\b",
    "royal_commission": r"\broyal commission\b",
    "reserve_bank": r"\breserve bank\b",
    "profit_downgrade": r"\bprofit downgrade\b",
    "capital_raising": r"\bcapital raising\b",
}

#: Single-token whole-word vocabulary. Counted once per record via tokenisation, which is
#: exactly equivalent to ``\bword\b`` over the combined field text.
AFR_VOCAB: tuple[str, ...] = (
    # tickers / issuer shorthand used in the question bank
    "agl", "amp", "anz", "azj", "bhp", "cba", "cmw", "gpt", "iag", "nab",
    "qan", "qbe", "rio", "sgp", "sun", "tah", "tpg", "tcl",
    # company names
    "qantas", "tabcorp", "transurban", "suncorp", "stockland", "cromwell",
    "aurizon", "westpac", "telstra", "woolworths", "wesfarmers", "macquarie",
    # policy and macro
    "rba", "asx", "unemployment", "inflation", "recession", "gdp", "budget",
    "deficit", "surplus", "stimulus", "wages", "productivity", "apra", "asic",
    "austrac", "treasury", "treasurer", "election", "tariffs", "brexit",
    # markets
    "dividend", "dividends", "franking", "buyback", "takeover", "merger",
    "ipo", "profit", "earnings", "downgrade", "upgrade", "volatility",
    "correction", "selloff", "rally", "bond", "bonds", "yields", "equities",
    "superannuation", "mortgage", "mortgages", "housing", "property",
    # sectors and commodities
    "banks", "banking", "insurance", "energy", "mining", "retail", "airlines",
    "oil", "gold", "copper", "coal", "lithium", "gas", "infrastructure",
    # events in-sample
    "coronavirus", "covid", "pandemic", "lockdown", "vaccine", "bushfires",
    "drought", "china", "trade", "hayne", "jobkeeper",
)


@dataclass
class AfrIndex:
    """One-pass summary of the AFR corpus.

    Only aggregates are retained — the corpus itself is never carried into training
    records (both models cap at ``max_model_len 4096``; tool results are capped at 1,200
    characters). ``token_counts[term][yyyymm]`` is the number of *records* in that month
    whose combined ``HEADLINE + SUBHEAD + INTRO + TEXT`` contains ``term`` as a whole
    word, counted once per record.
    """

    total_articles: int
    file_count: int
    records_by_month: dict[str, int]
    token_counts: dict[str, dict[str, int]]
    pattern_counts: dict[str, dict[str, int]]
    date_min: str
    date_max: str
    undated_records: int = 0
    headlines: dict[str, dict] = field(default_factory=dict)
    sentiment_pool: list[dict] = field(default_factory=list)

    # -- aggregation helpers -----------------------------------------------------------
    #: 92 records carry an empty ``PUBLICATIONDATE``. They are bucketed here rather than
    #: dropped, so whole-corpus counts match a full regex scan of the raw files while
    #: year/month aggregation excludes them automatically.
    UNDATED = "undated"

    @staticmethod
    def _sum(buckets: dict[str, int], year: int | None = None, month: int | None = None) -> int:
        if year is None:
            return sum(buckets.values())
        prefix = f"{year:04d}" if month is None else f"{year:04d}{month:02d}"
        return sum(v for k, v in buckets.items() if k.startswith(prefix))

    def term_count(self, term: str, year: int | None = None, month: int | None = None) -> int:
        return self._sum(self.token_counts[term.lower()], year, month)

    def pattern_count(self, name: str, year: int | None = None, month: int | None = None) -> int:
        return self._sum(self.pattern_counts[name], year, month)

    def record_count(self, year: int | None = None, month: int | None = None) -> int:
        return self._sum(self.records_by_month, year, month)

    def term_by_year(self, term: str) -> dict[int, int]:
        out: dict[int, int] = {}
        for ym, n in self.token_counts[term.lower()].items():
            if ym == self.UNDATED:
                continue
            out[int(ym[:4])] = out.get(int(ym[:4]), 0) + n
        return dict(sorted(out.items()))

    def term_by_month(self, term: str) -> dict[str, int]:
        return {
            k: v for k, v in sorted(self.token_counts[term.lower()].items())
            if k != self.UNDATED
        }

    @property
    def years(self) -> list[int]:
        return sorted({int(k[:4]) for k in self.records_by_month if k != self.UNDATED})

    def to_json(self) -> dict:
        return {
            "total_articles": self.total_articles,
            "file_count": self.file_count,
            "records_by_month": self.records_by_month,
            "token_counts": self.token_counts,
            "pattern_counts": self.pattern_counts,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "undated_records": self.undated_records,
            "headlines": self.headlines,
            "sentiment_pool": self.sentiment_pool,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "AfrIndex":
        return cls(**blob)


#: Headlines the public question bank retrieves by name. Captured verbatim during the
#: scan so the sentiment slice can be anchored on the organizer's own examples.
ANCHOR_HEADLINES: tuple[tuple[str, str], ...] = (
    ("Travel stocks take off on vaccine rollout", "20210223"),
    ("Why investors don't believe the RBA on interest rates", "20211125"),
    ("Energy stocks shine as vaccines fuel oil rally", "20201128"),
)

_SENTIMENT_HINT_RE = re.compile(
    r"\b(shares?|stocks?|asx|investors?|market|rally|selloff|profit|earnings|"
    r"dividend|rate cut|rate rise|cash rate)\b"
)


def _combined_text(rec: dict) -> str:
    return "\n".join(
        str(rec.get(k) or "") for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT")
    )


def build_afr_index(
    *,
    sentiment_pool_size: int = 900,
    sentiment_stride: int = 211,
    progress: bool = False,
) -> AfrIndex:
    """Single pass over all 85 AFR files.

    One ``re.findall`` per record for the whole-word vocabulary, plus one search per
    entry in :data:`AFR_REGEX_PATTERNS`. Everything downstream reads the cached result.
    """
    vocab = set(AFR_VOCAB)
    token_counts: dict[str, dict[str, int]] = {t: {} for t in vocab}
    compiled = {
        name: re.compile(pat, re.IGNORECASE) for name, pat in AFR_REGEX_PATTERNS.items()
    }
    pattern_counts: dict[str, dict[str, int]] = {name: {} for name in compiled}
    records_by_month: dict[str, int] = {}
    headlines: dict[str, dict] = {}
    anchors = {(h.lower(), d) for h, d in ANCHOR_HEADLINES}
    sentiment_pool: list[dict] = []

    total = 0
    undated = 0
    date_min = "99999999"
    date_max = "00000000"
    files = sorted(paths.AFR_DIR.glob("AFR_*.jsonl"))
    seen = 0
    for fp in files:
        if progress:
            print(f"  scanning {fp.name}", flush=True)
        for line in fp.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pub = str(rec.get("PUBLICATIONDATE") or "")
            total += 1
            seen += 1
            if len(pub) == 8:
                ym = pub[:6]
                date_min = min(date_min, pub)
                date_max = max(date_max, pub)
            else:
                ym = AfrIndex.UNDATED
                undated += 1
            records_by_month[ym] = records_by_month.get(ym, 0) + 1

            text = _combined_text(rec)
            lowered = text.lower()
            present = vocab.intersection(_TOKEN_RE.findall(lowered))
            for term in present:
                bucket = token_counts[term]
                bucket[ym] = bucket.get(ym, 0) + 1
            for name, rx in compiled.items():
                if rx.search(text):
                    bucket = pattern_counts[name]
                    bucket[ym] = bucket.get(ym, 0) + 1

            headline = str(rec.get("HEADLINE") or "")
            if (headline.lower(), pub) in anchors:
                headlines[f"{headline}|{pub}"] = _slim_article(rec)
            if (
                len(sentiment_pool) < sentiment_pool_size
                and seen % sentiment_stride == 0
                and len(headline) > 18
                and _SENTIMENT_HINT_RE.search(lowered)
            ):
                sentiment_pool.append(_slim_article(rec))

    return AfrIndex(
        total_articles=total,
        file_count=len(files),
        records_by_month=records_by_month,
        token_counts=token_counts,
        pattern_counts=pattern_counts,
        date_min=date_min,
        date_max=date_max,
        undated_records=undated,
        headlines=headlines,
        sentiment_pool=sentiment_pool,
    )


def _slim_article(rec: dict, text_chars: int = 1500) -> dict:
    """Keep only what the sentiment slice needs; never carry full article bodies."""
    return {
        "headline": str(rec.get("HEADLINE") or "").strip(),
        "subhead": str(rec.get("SUBHEAD") or "").strip(),
        "intro": str(rec.get("INTRO") or "").strip(),
        "text": str(rec.get("TEXT") or "").strip()[:text_chars],
        "publication_date": str(rec["PUBLICATIONDATE"]),
    }


@lru_cache(maxsize=1)
def load_afr(rebuild: bool = False, progress: bool = False) -> AfrIndex:
    paths.ensure_dirs()
    if paths.AFR_INDEX_CACHE.exists() and not rebuild:
        with paths.AFR_INDEX_CACHE.open(encoding="utf-8") as fh:
            return AfrIndex.from_json(json.load(fh))
    index = build_afr_index(progress=progress)
    with paths.AFR_INDEX_CACHE.open("w", encoding="utf-8") as fh:
        json.dump(index.to_json(), fh)
    return index


# --------------------------------------------------------------------------------------
# coverage — the metadata behind every legitimate refusal
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Coverage:
    dataset: str
    first: str
    last: str


def coverage() -> dict[str, Coverage]:
    rba = load_rba()
    asx = load_asx()
    afr = load_afr()

    def iso(yyyymmdd: str) -> str:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"

    return {
        "RBA": Coverage("RBA", rba.records[0].iso, rba.records[-1].iso),
        "ASX": Coverage("ASX", asx.date_min.isoformat(), asx.date_max.isoformat()),
        "AFR": Coverage("AFR", iso(afr.date_min), iso(afr.date_max)),
    }


def iter_public_questions() -> Iterator[dict]:
    with paths.PUBLIC_QUESTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
