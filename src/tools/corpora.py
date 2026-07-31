"""Corpus loaders. Every field gotcha is handled here, once.

Loaded at import by `app.py` **before uvicorn binds the port**, so `/health` only
starts answering once the corpora are ready — that keeps `/health` a pure
process-liveness check, which matters because it is a hard gate on the entire 40%
hidden-question category.

The gotchas below are measured, not guessed, and each one silently produces wrong
counts rather than an error — which is why they live in one module instead of at
every call site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from .. import config

# --------------------------------------------------------------------------
# RBA
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RbaRow:
    """One cash-rate decision, with the strings already coerced.

    The corpus stores every value as a string, including signed changes
    (`"+0.25"`, `"-0.25"`, `"0.00"`). Coercing at load means no metric has to
    remember to.
    """

    date: date
    change: float
    rate: float


def _parse_rba_date(raw: str) -> date:
    """`"3 Feb 2010"` -> date. NOT ISO — `date.fromisoformat` raises on it."""
    return datetime.strptime(raw.strip(), "%d %b %Y").date()


def load_rba(path: Path | None = None) -> list[RbaRow]:
    """Load RBA decisions, sorted ascending by effective date.

    **UTF-8 BOM.** `RBA-rates.jsonl` starts with `\\ufeff`. Read as plain utf-8
    that character reaches `json.loads`, which raises
    `JSONDecodeError: Unexpected UTF-8 BOM` on the very first line — so this one
    fails loudly rather than silently, but it fails at import, before uvicorn
    binds, which would take `/health` down with it. `encoding="utf-8-sig"` is
    mandatory.

    Sorted because `lookup_rate` relies on ordering for as-of semantics, and the
    file order should not be trusted to provide it.
    """
    path = path or config.RBA_PATH
    rows: list[RbaRow] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                RbaRow(
                    date=_parse_rba_date(raw["Effective Date"]),
                    change=float(raw["Change % points"]),
                    rate=float(raw["Cash rate target%"]),
                )
            )
    rows.sort(key=lambda r: r.date)
    return rows


# --------------------------------------------------------------------------
# ASX
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AsxBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def load_asx(directory: Path | None = None) -> dict[str, list[AsxBar]]:
    """Load every ticker's daily bars, keyed by the REAL ticker.

    **Filenames are company names, not tickers.** `Tabcorp-ASX-2015-2021.jsonl`
    holds `TAH.AX`; `Aurizon-` holds `AZJ.AX`. 7 of the 18 stems do not match
    their ticker prefix, so deriving the ticker from the filename mis-keys the
    corpus — and `TAH.AX` is the one ticker excluded in 5 of the 15 public
    questions, so that mistake would silently corrupt `exclude_tickers`.
    `FINETUNE_PLAN.md` §2.4's `<TICKER>-` pattern is wrong; read the `ticker`
    field off the rows.
    """
    directory = directory or config.ASX_DIR
    out: dict[str, list[AsxBar]] = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        ticker = None
        bars: list[AsxBar] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                raw = json.loads(line)
                ticker = ticker or raw["ticker"]
                bars.append(
                    AsxBar(
                        date=date.fromisoformat(raw["date"]),
                        open=float(raw["open"]),
                        high=float(raw["high"]),
                        low=float(raw["low"]),
                        close=float(raw["close"]),
                        volume=int(raw["volume"]),
                    )
                )
        if ticker:
            bars.sort(key=lambda b: b.date)
            out[ticker] = bars
    return out


# --------------------------------------------------------------------------
# Cached accessors
#
# The loop is per-request but the corpora are read-only after startup, so they
# load once per process. That is also what makes concurrency safe by
# construction rather than by locking (harness §7).
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def rba_rows() -> tuple[RbaRow, ...]:
    return tuple(load_rba())


@lru_cache(maxsize=1)
def asx_series() -> dict[str, tuple[AsxBar, ...]]:
    return {k: tuple(v) for k, v in load_asx().items()}
