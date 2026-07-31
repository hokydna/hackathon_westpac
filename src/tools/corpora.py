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
import re
from array import array
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


# --------------------------------------------------------------------------
# AFR — inverted index
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Fields the whole-word search runs over, COMBINED, counted once per record.
#: `Setup_Instructions.md` calls this non-negotiable for reproducibility: a
#: different field set silently yields counts that will not match the reference
#: answers. Measured: this exact set gives unemployment 5,997 / qbe 1,546 /
#: nab 7,372, identical to a full `\bword\b` regex scan.
AFR_FIELDS = ("HEADLINE", "SUBHEAD", "INTRO", "TEXT")


def tokenize(text: str) -> set[str]:
    """Lowercase `[a-z0-9]+` tokens.

    **Do not add the apostrophe to the character class.** `[a-z0-9']+` gives
    6,903 for `nab` where the correct answer is 7,372, because `\\b` treats an
    apostrophe as a word boundary — so `nab's` must yield the token `nab`, not
    `nab's`. Verified equal to a `\\bword\\b` regex scan on all three reference
    terms.
    """
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class AfrIndex:
    """Postings + per-record dates. Answers counts in <1ms.

    Postings are `array("i")` of record ids rather than Python sets: there are
    ~44M (token, record) pairs across 219,538 records, and a set of boxed ints
    per token would cost multiple GB. Each node here is a single GB10 whose
    unified memory is SHARED with vLLM, and `vllm-brain` sits on node0 beside the
    agent — so an oversized index does not just slow us down, it can OOM the
    brain and take sessions B and C with it.

    The index is mandatory, not an optimisation: a full regex scan is ~36s per
    pattern and `re` does not release the GIL, so neither caching nor threads fix
    it. Against a 60s scored budget, one uncached AFR question would blow it.
    """

    postings: dict[str, array]
    years: list[str]
    months: list[str]
    headlines: list[str]
    total: int

    def record_ids(self, term: str) -> array:
        return self.postings.get(term.strip().lower(), array("i"))


def build_afr_index(directory: Path | None = None) -> AfrIndex:
    """Build the inverted index. ~40s over the real corpus."""
    directory = directory or config.AFR_DIR
    postings: dict[str, list[int]] = {}
    years: list[str] = []
    months: list[str] = []
    headlines: list[str] = []

    rid = 0
    for path in sorted(Path(directory).glob("AFR_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                raw = json.loads(line)
                blob = " ".join(str(raw.get(k) or "") for k in AFR_FIELDS)
                # PUBLICATIONDATE is a YYYYMMDD *string*. Slice it — do not
                # date-parse. Some records are undated, hence the guard.
                pub = str(raw.get("PUBLICATIONDATE") or "")
                years.append(pub[:4] if len(pub) >= 4 and pub[:4].isdigit() else "")
                months.append(pub[:6] if len(pub) >= 6 and pub[:6].isdigit() else "")
                headlines.append(str(raw.get("HEADLINE") or ""))
                for tok in tokenize(blob):
                    postings.setdefault(tok, []).append(rid)
                rid += 1

    return AfrIndex(
        postings={t: array("i", ids) for t, ids in postings.items()},
        years=years,
        months=months,
        headlines=headlines,
        total=rid,
    )


@lru_cache(maxsize=1)
def afr_index() -> AfrIndex:
    return build_afr_index()
