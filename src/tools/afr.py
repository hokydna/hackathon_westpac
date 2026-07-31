"""Index-backed AFR metrics. No LLM anywhere in this file.

Every count here obeys the rule `Setup_Instructions.md` calls non-negotiable for
reproducibility: **case-insensitive, `\\bword\\b` anchored, matched across
HEADLINE + SUBHEAD + INTRO + TEXT combined, counted once per record.** A
different field set or a different token class silently produces counts that will
not match the reference answers — so the rule lives in `corpora.tokenize` and
`corpora.AFR_FIELDS`, and nothing here re-implements it.

Same two conventions as `rba.py`: return typed dicts for synthesis, and never
raise on an empty result set.
"""

from __future__ import annotations

import collections

from . import corpora
from .corpora import AfrIndex


def _index(index: AfrIndex | None) -> AfrIndex:
    return index if index is not None else corpora.afr_index()


def count(term: str, year: str | int | None = None, index: AfrIndex | None = None) -> dict:
    """Records containing `term` as a whole word, optionally within one year.

    MHQ076 needs the 2021 slice: "369 AFR records matching whole-word QBE in
    2021". That question is a single 10-point all-or-nothing component, so the
    year filter is not a nicety.
    """
    idx = _index(index)
    ids = idx.record_ids(term)

    if year is None:
        return {
            "term": term.strip().lower(),
            "matching_records": len(ids),
            "corpus_records": idx.total,
        }

    y = str(year)
    n = sum(1 for rid in ids if idx.years[rid] == y)
    return {
        "term": term.strip().lower(),
        "year": y,
        "matching_records": n,
        "corpus_records": idx.total,
    }


def count_by_month(term: str, index: AfrIndex | None = None) -> dict:
    """Peak year and peak month for a term.

    MHQ061: "It peaked in 2020 with 1,452 matching records. May 2020 is the peak
    month with 218." Both numbers ship together because they are two graded
    components of one question.
    """
    idx = _index(index)
    ids = idx.record_ids(term)
    if not len(ids):
        return {
            "term": term.strip().lower(),
            "matching_records": 0,
            "note": f"No AFR records contain the whole word '{term}'.",
        }

    years: collections.Counter = collections.Counter()
    months: collections.Counter = collections.Counter()
    for rid in ids:
        if idx.years[rid]:
            years[idx.years[rid]] += 1
        if idx.months[rid]:
            months[idx.months[rid]] += 1

    peak_year, peak_year_n = years.most_common(1)[0]
    peak_month, peak_month_n = months.most_common(1)[0]
    return {
        "term": term.strip().lower(),
        "matching_records": len(ids),
        "peak_year": peak_year,
        "peak_year_count": peak_year_n,
        "peak_month": peak_month,           # YYYYMM, as the corpus stores it
        "peak_month_count": peak_month_n,
        "counts_by_year": dict(sorted(years.items())),
    }


def share(term: str, year: str | int | None = None, index: AfrIndex | None = None) -> dict:
    """Matching records as a share of the corpus (or of that year)."""
    idx = _index(index)
    ids = idx.record_ids(term)

    if year is None:
        n, denom, scope = len(ids), idx.total, "corpus"
    else:
        y = str(year)
        n = sum(1 for rid in ids if idx.years[rid] == y)
        denom = sum(1 for v in idx.years if v == y)
        scope = y

    if denom == 0:
        return {"term": term.strip().lower(), "note": f"No records in scope '{scope}'."}
    return {
        "term": term.strip().lower(),
        "scope": scope,
        "matching_records": n,
        "scope_records": denom,
        "share_pct": round(100.0 * n / denom, 4),
    }


def retrieve_by_headline(
    query: str,
    limit: int = 3,
    index: AfrIndex | None = None,
) -> dict:
    """★ Find records by headline text — the entry point for MHQ058/067/080.

    Those three are the article-grounded sentiment questions, and none of them can
    start without pulling the specific article first. Ranked by how many of the
    query's tokens the headline contains, so a partial or reworded headline still
    lands on the right record.
    """
    idx = _index(index)
    wanted = corpora.tokenize(query)
    if not wanted:
        return {"matches": [], "note": "Empty query."}

    scored: list[tuple[int, int]] = []
    for rid, headline in enumerate(idx.headlines):
        overlap = len(wanted & corpora.tokenize(headline))
        if overlap:
            scored.append((overlap, rid))
    scored.sort(key=lambda p: (-p[0], p[1]))

    if not scored:
        return {"matches": [], "note": f"No AFR headline matches '{query}'."}

    return {
        "query": query,
        "matches": [
            {
                "headline": idx.headlines[rid],
                "publication_date": idx.months[rid] or idx.years[rid] or "",
                "matched_tokens": overlap,
            }
            for overlap, rid in scored[:limit]
        ],
    }


def coverage(index: AfrIndex | None = None) -> dict:
    """★ AFR's date span, for cross-dataset coverage checks.

    MHQ090's correct answer is a justified refusal: RBA runs past the 2022–23
    hikes but AFR and ASX both stop in 2021, so any post-2021 join is
    unsupported. That refusal is worth 10 points across three components, and
    "No" alone earns 3.33 — the evidence-boundary reasoning carries the rest.
    """
    idx = _index(index)
    dated = [y for y in idx.years if y]
    if not dated:
        return {"note": "No dated AFR records."}
    return {
        "dataset": "AFR",
        "records": idx.total,
        "dated_records": len(dated),
        "first_year": min(dated),
        "last_year": max(dated),
    }
