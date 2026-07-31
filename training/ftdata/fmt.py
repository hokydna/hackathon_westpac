"""Display formatting for gold answers.

The grader is exact on dates, counts, labels, rankings and rates, and allows narrow
tolerances on calculated values (±0.02 pp on returns, ±0.001 on correlations, ±0.0001 on
quoted closes, ±1 share on average volume). Every number that reaches an ``expected_answer``
goes through one of these helpers so training targets and the evaluator agree on precision.
"""

from __future__ import annotations

from datetime import date

from .corpora import display_date


def signed_pct(value: float, dp: int = 2) -> str:
    """``+22.17%`` / ``-50.04%`` — the question bank's return format."""
    return f"{value:+.{dp}f}%"


def pct(value: float, dp: int = 2) -> str:
    """``0.10%`` — a level, not a movement."""
    return f"{value:.{dp}f}%"


def rate_pct(value: float) -> str:
    """RBA cash-rate target: two decimals, as the RBA publishes it."""
    return f"{value:.2f}%"


def pp(value: float, dp: int = 2) -> str:
    """Percentage points, signed. RBA movements are never called 'percent'."""
    return f"{value:+.{dp}f} percentage points"


def count(value: int) -> str:
    """Thousands separators, matching ``1,774`` and ``11,635,671.71`` in the bank."""
    return f"{value:,}"


def volume(value: float) -> str:
    return f"{value:,.2f}"


def price(value: float, dp: int = 4) -> str:
    """Quoted closes carry a ±0.0001 tolerance, so four decimals."""
    return f"{value:.{dp}f}"


def corr(value: float) -> str:
    return f"{value:.3f}"


def d(value: date | str) -> str:
    """``date(2015, 3, 20)`` -> ``20 Mar 2015``."""
    if isinstance(value, str):
        return value
    return display_date(value)


def iso(value: date | str) -> str:
    return value if isinstance(value, str) else value.isoformat()


def ymd_display(yyyymmdd: str) -> str:
    """``"20210223"`` -> ``23 Feb 2021``."""
    y, m, dd = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
    return display_date(date(y, m, dd))


def ymd_iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def month_label(yyyymm: str) -> str:
    """``"202005"`` -> ``May 2020``."""
    return f"{MONTH_NAMES[int(yyyymm[4:6])][:3]} {yyyymm[:4]}"


def month_label_long(yyyymm: str) -> str:
    return f"{MONTH_NAMES[int(yyyymm[4:6])]} {yyyymm[:4]}"


def ticker_list(tickers: list[str]) -> str:
    if len(tickers) == 1:
        return tickers[0]
    return ", ".join(tickers[:-1]) + f" and {tickers[-1]}"
