"""Deterministic RBA metrics. No LLM anywhere in this file.

This is where hidden-question points come from, and every value is computable
exactly — so a wrong answer here is a bug, never a model failure.

Two conventions hold across every function:

* **Return a dict of typed values**, not a string. The adapter is trained on
  structured evidence (`{"changed": 41, ...}`), so the registry wraps these into
  the human-readable string the brain reads while `last_data` carries the typed
  payload through to synthesis.
* **Never raise on an empty result set** (harness §7). An out-of-range date or an
  empty window returns a valid "no data" payload with a `note`, so the brain can
  try a different query instead of the request failing.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

from . import corpora
from .corpora import RbaRow


def _rows(rows: Sequence[RbaRow] | None) -> Sequence[RbaRow]:
    return rows if rows is not None else corpora.rba_rows()


def _coerce_date(raw: str) -> date | None:
    """Accept ISO and the corpus's own `"3 Feb 2010"` form.

    Qwen may echo a date straight back out of a previous tool result, so both
    shapes have to work. Returns None rather than raising — an unparseable date
    is a "no data" answer, not a crash.
    """
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


def count_changes(rows: Sequence[RbaRow] | None = None) -> dict:
    """How many decisions changed the rate, split by direction.

    MHQ001. A single 10-point all-or-nothing component bundling three numbers, so
    two of three correct scores zero — which is why all four values ship together
    in one payload rather than being recomputed per component.
    """
    data = _rows(rows)
    changed = [r for r in data if r.change != 0.0]
    increases = sum(1 for r in changed if r.change > 0)
    return {
        "total_records": len(data),
        "changed": len(changed),
        "increases": increases,
        "decreases": len(changed) - increases,
    }


def count(rows: Sequence[RbaRow] | None = None) -> dict:
    return {"total_records": len(_rows(rows))}


def lookup_rate(as_of: str, rows: Sequence[RbaRow] | None = None) -> dict:
    """The rate in force ON or BEFORE `as_of`.

    **As-of, never nearest-match.** Nearest-match can return a decision from the
    FUTURE relative to the requested date, which answers a different question
    while looking plausible. `handout/03` calls this out explicitly.
    """
    data = _rows(rows)
    target = _coerce_date(as_of)
    if target is None:
        return {"rate": None, "note": f"Could not interpret '{as_of}' as a date."}

    prior = [r for r in data if r.date <= target]
    if not prior:
        first = data[0].date.isoformat() if data else "n/a"
        return {
            "rate": None,
            "note": (
                f"No cash-rate decision exists on or before {target.isoformat()}; "
                f"the dataset begins {first}."
            ),
        }

    row = prior[-1]
    return {
        "as_of": target.isoformat(),
        "effective_date": row.date.isoformat(),
        "rate": row.rate,
        "change_on_that_date": row.change,
    }


def extremes(rows: Sequence[RbaRow] | None = None) -> dict:
    """Highest and lowest target, each with its first effective date and count."""
    data = _rows(rows)
    if not data:
        return {"note": "No decisions in range."}

    lo = min(r.rate for r in data)
    hi = max(r.rate for r in data)
    return {
        "min_rate": lo,
        "min_rate_first_date": min(r.date for r in data if r.rate == lo).isoformat(),
        "min_rate_record_count": sum(1 for r in data if r.rate == lo),
        "max_rate": hi,
        "max_rate_first_date": min(r.date for r in data if r.rate == hi).isoformat(),
        "max_rate_record_count": sum(1 for r in data if r.rate == hi),
    }


def max_hold_streak(rows: Sequence[RbaRow] | None = None) -> dict:
    """Longest stretch between two consecutive non-zero changes.

    Measured between CHANGE dates, not between decision records: the hold is the
    gap during which the rate did not move, so the intervening 0.00 decisions are
    part of the streak rather than breaks in it.
    """
    data = _rows(rows)
    changes = [r for r in data if r.change != 0.0]
    if len(changes) < 2:
        return {"note": "Fewer than two rate changes in range."}

    best = max(zip(changes, changes[1:]), key=lambda pair: (pair[1].date - pair[0].date).days)
    start, end = best
    return {
        "days": (end.date - start.date).days,
        "start_date": start.date.isoformat(),
        "end_date": end.date.isoformat(),
        "rate_during_hold": start.rate,
        "rate_after": end.rate,
    }


def cycle_summary(
    date_from: str,
    date_to: str,
    rows: Sequence[RbaRow] | None = None,
) -> dict:
    """Cumulative movement across a tightening or easing cycle.

    ★ Required by the question bank (MHQ035, MHQ084) but absent from the
    execution guide's metric list, so it is easy to miss entirely.

    `rate_before_first` is the rate in force immediately before the first CHANGE
    in the window, not before the window — a cycle question asks what the rate was
    when the cycle began moving.
    """
    data = _rows(rows)
    lo, hi = _coerce_date(date_from), _coerce_date(date_to)
    if lo is None or hi is None:
        return {"decisions": 0, "note": "Could not interpret the date range."}

    window = [r for r in data if lo <= r.date <= hi]
    changes = [r for r in window if r.change != 0.0]
    if not changes:
        return {
            "decisions": len(window),
            "changes": 0,
            "note": (
                f"No rate changes between {lo.isoformat()} and {hi.isoformat()}"
                + (f"; {len(window)} decisions held the rate." if window else "; no decisions in range.")
            ),
        }

    before = [r for r in data if r.date < changes[0].date]
    hikes = sum(1 for r in changes if r.change > 0)
    return {
        "date_from": lo.isoformat(),
        "date_to": hi.isoformat(),
        "decisions": len(window),
        "changes": len(changes),
        "hikes": hikes,
        "cuts": len(changes) - hikes,
        "cumulative_change_pp": round(sum(r.change for r in changes), 4),
        "rate_before_first": before[-1].rate if before else None,
        "rate_final": window[-1].rate,
        "first_change_date": changes[0].date.isoformat(),
        "last_change_date": changes[-1].date.isoformat(),
    }
