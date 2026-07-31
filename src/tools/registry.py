"""Tool definitions -> `ALL_TOOLS` + `BRAIN_SCHEMAS`. Owned by session A alone.

Session B imports these and never edits this file — that ownership call is
kickoff §4, made because harness §10 step 8 originally listed it as "owners A+B",
which is exactly the shared-file collision Phase 0 exists to prevent.

Two things the tool surface has to get right, both scored:

**Tools return prose, not `k=v` dumps.** An earlier draft returned
`"changed=41; increases=20"`, and base Nemotron simply echoed that shape into the
answer — which the judge scored 0/10 on MHQ001 despite every number being
correct. The brain reads these strings, and so does synthesis, so each one is a
sentence a human would accept. The typed payload rides alongside in `last_data`
for the adapter, which was trained on structured evidence.

**`domain_sentiment` is the one tool that calls Nemotron** (kickoff §10, Role 2).
`Setup_Instructions.md` L95 requires article-grounded sentiment to route retrieved
AFR text plus the applicable RBA rate through `DOMAIN_FT_MODEL`. It returns a
CLASSIFICATION clamped to 200 chars, never a full answer, and it is denied when no
article has been retrieved — those constraints are what keep Role 2 from becoming
the prohibited "Nemotron as planner and tool caller" pattern.
"""

from __future__ import annotations

from typing import Any, Iterable

from .. import config, domain_client, prompts
from . import afr, asx, corpora, rba


def _fmt(value: Any) -> str:
    """Numbers with thousands separators; everything else as-is.

    Challenge_Brief.md accepts "commas, trailing zeros, and equivalent date
    formats", so grouping is safe and it makes 11635671.71 legible to the brain.
    """
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 1000 else f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


class Tool:
    """A callable tool with a JSON schema and a typed payload channel.

    Deliberately not a LangChain `@tool`: those derive their schema from the
    signature, and we need one `query_data` entry point over three datasets with
    a metric discriminator, which is the shape the brain was prompted against.
    Pydantic coercion still happens — in `guard.py`, via `args_schema`.
    """

    def __init__(self, name: str, description: str, parameters: dict, fn):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn
        self.args_schema = None  # guard passes args through; each fn validates
        self.last_data: dict | None = None

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def ainvoke(self, args: dict) -> str:
        text, data = self._fn(args or {})
        self.last_data = data
        return text


# --------------------------------------------------------------------------
# query_data — the deterministic surface over all three corpora
# --------------------------------------------------------------------------

_RBA_METRICS = {
    "count", "count_changes", "count_increases", "count_decreases",
    "extremes", "max_hold_streak", "lookup_rate", "cycle_summary",
}
_ASX_METRICS = {
    "describe", "annual_return", "rank_annual_returns", "full_sample_return",
    "volatility", "correlation", "max_drawdown", "avg_volume", "window_return",
}
_AFR_METRICS = {"count", "count_by_month", "share", "retrieve_by_headline"}


def _excl(args: dict) -> Iterable[str] | None:
    raw = args.get("exclude_tickers")
    if raw is None:
        return None
    return raw if isinstance(raw, (list, tuple)) else [raw]


def _query_data(args: dict) -> tuple[str, dict]:
    dataset = str(args.get("dataset", "")).strip().lower()
    metric = str(args.get("metric", "")).strip().lower()

    if dataset == "rba":
        if metric not in _RBA_METRICS:
            return (f"Unknown RBA metric '{metric}'. Available: "
                    f"{', '.join(sorted(_RBA_METRICS))}.", {})
        if metric in ("count_changes", "count_increases", "count_decreases"):
            d = rba.count_changes()
            return (
                f"{_fmt(d['changed'])} of {_fmt(d['total_records'])} RBA decision records "
                f"changed the cash-rate target: {_fmt(d['increases'])} increases and "
                f"{_fmt(d['decreases'])} decreases.", d)
        if metric == "count":
            d = rba.count()
            return f"The RBA dataset holds {_fmt(d['total_records'])} decision records.", d
        if metric == "extremes":
            d = rba.extremes()
            return (
                f"The highest cash-rate target was {_fmt(d['max_rate'])}, first effective "
                f"{d['max_rate_first_date']} and shown by {_fmt(d['max_rate_record_count'])} "
                f"records; the lowest was {_fmt(d['min_rate'])}, first effective "
                f"{d['min_rate_first_date']} and shown by "
                f"{_fmt(d['min_rate_record_count'])} records.", d)
        if metric == "max_hold_streak":
            d = rba.max_hold_streak()
            if "note" in d:
                return d["note"], d
            return (
                f"The longest stretch between two non-zero rate changes was "
                f"{_fmt(d['days'])} days, from {d['start_date']} to {d['end_date']}, held at "
                f"{_fmt(d['rate_during_hold'])} before changing to "
                f"{_fmt(d['rate_after'])}.", d)
        if metric == "lookup_rate":
            d = rba.lookup_rate(args.get("date") or args.get("date_from") or "")
            if d.get("rate") is None:
                return d.get("note", "No rate available."), d
            return (
                f"The cash-rate target in force on {d['as_of']} was {_fmt(d['rate'])}%, "
                f"set effective {d['effective_date']}.", d)
        d = rba.cycle_summary(args.get("date_from", ""), args.get("date_to", ""))
        if d.get("changes", 0) == 0:
            return d.get("note", "No rate changes in range."), d
        return (
            f"Between {d['date_from']} and {d['date_to']} there were {_fmt(d['changes'])} "
            f"rate changes ({_fmt(d['hikes'])} hikes, {_fmt(d['cuts'])} cuts) across "
            f"{_fmt(d['decisions'])} decisions, a cumulative "
            f"{d['cumulative_change_pp']:+.2f} percentage points, taking the target from "
            f"{_fmt(d['rate_before_first'])}% before the first change to "
            f"{_fmt(d['rate_final'])}%.", d)

    if dataset == "asx":
        if metric not in _ASX_METRICS:
            return (f"Unknown ASX metric '{metric}'. Available: "
                    f"{', '.join(sorted(_ASX_METRICS))}.", {})
        ex = _excl(args)
        if metric == "describe":
            d = asx.describe(exclude_tickers=ex)
            return (
                f"There are {_fmt(d['ticker_files'])} ticker files, each containing "
                f"{_fmt(d['rows_per_ticker'])} rows, covering {d['first_date']} through "
                f"{d['last_date']}.", d)
        if metric == "annual_return":
            d = asx.annual_return(args.get("ticker", ""), args.get("year", 0), exclude_tickers=ex)
            if "note" in d:
                return d["note"], d
            return (
                f"{d['ticker']}'s {d['year']} price return was "
                f"{d['price_return_pct']:+.2f}%, first-to-last close over "
                f"{_fmt(d['trading_days'])} trading days.", d)
        if metric == "rank_annual_returns":
            d = asx.rank_annual_returns(args.get("year", 0), exclude_tickers=ex)
            if "note" in d:
                return d["note"], d
            return (
                f"In {d['year']}, {d['best_ticker']} was best at "
                f"{d['best_return_pct']:+.2f}% and {d['worst_ticker']} was worst at "
                f"{d['worst_return_pct']:+.2f}%.", d)
        if metric == "full_sample_return":
            d = asx.full_sample_return(exclude_tickers=ex)
            return (
                f"Over the full sample {d['best_ticker']} returned "
                f"{d['best_return_pct']:+.2f}% and {d['worst_ticker']} returned "
                f"{d['worst_return_pct']:+.2f}%.", d)
        if metric == "avg_volume":
            d = asx.avg_volume(exclude_tickers=ex)
            return (
                f"{d['highest_ticker']} has the highest average daily volume at "
                f"{_fmt(d['highest_avg_daily_volume'])} shares per trading day.", d)
        if metric == "volatility":
            d = asx.volatility(args.get("ticker", ""), args.get("year"), exclude_tickers=ex)
            if "note" in d:
                return d["note"], d
            scope = f" in {d['year']}" if d.get("year") else ""
            return (
                f"{d['ticker']}'s annualised volatility{scope} was "
                f"{d['volatility_pct_annualised']:.2f}%, from "
                f"{_fmt(d['daily_return_count'])} {d['basis']} scaled by the square root "
                f"of {d['annualisation_factor']}.", d)
        if metric == "correlation":
            d = asx.correlation(args.get("ticker", ""), args.get("ticker_b", ""), args.get("year"))
            if "note" in d:
                return d["note"], d
            return (
                f"The correlation of daily returns between {d['ticker_a']} and "
                f"{d['ticker_b']} was {d['correlation']:.4f} over "
                f"{_fmt(d['overlapping_days'])} overlapping trading days.", d)
        if metric == "max_drawdown":
            d = asx.max_drawdown(args.get("ticker", ""))
            if "note" in d:
                return d["note"], d
            return (
                f"{d['ticker']}'s deepest peak-to-trough decline was "
                f"{d['max_drawdown_pct']:+.2f}%, from a peak on {d['peak_date']} to a "
                f"trough on {d['trough_date']}.", d)
        d = asx.window_return(
            args.get("date_from", ""), args.get("date_to", ""),
            tickers=args.get("tickers"), exclude_tickers=ex)
        if "note" in d:
            return d["note"], d
        per = "; ".join(f"{t} {v:+.2f}%" for t, v in sorted(d["returns_pct"].items()))
        return (
            f"From {d['date_from']} to {d['date_to']} the equal-weighted basket of "
            f"{len(d['tickers'])} tickers returned "
            f"{d['equal_weighted_basket_return_pct']:+.2f}%. Per ticker: {per}.", d)

    if dataset == "afr":
        if metric not in _AFR_METRICS:
            return (f"Unknown AFR metric '{metric}'. Available: "
                    f"{', '.join(sorted(_AFR_METRICS))}.", {})
        term = str(args.get("pattern") or args.get("term") or "")
        if metric == "count":
            d = afr.count(term, year=args.get("year"))
            scope = f" in {d['year']}" if d.get("year") else ""
            return (
                f"There are {_fmt(d['matching_records'])} AFR records matching whole-word "
                f"'{d['term']}'{scope}, out of {_fmt(d['corpus_records'])}.", d)
        if metric == "count_by_month":
            d = afr.count_by_month(term)
            if d.get("matching_records", 0) == 0:
                return d.get("note", "No matching records."), d
            ym = d["peak_month"]
            return (
                f"Whole-word '{d['term']}' appears in {_fmt(d['matching_records'])} AFR "
                f"records. It peaked in {d['peak_year']} with "
                f"{_fmt(d['peak_year_count'])} matching records, and the peak month was "
                f"{ym[:4]}-{ym[4:]} with {_fmt(d['peak_month_count'])}.", d)
        if metric == "share":
            d = afr.share(term, year=args.get("year"))
            if "note" in d:
                return d["note"], d
            return (
                f"Whole-word '{d['term']}' appears in {_fmt(d['matching_records'])} of "
                f"{_fmt(d['scope_records'])} records in {d['scope']}, a share of "
                f"{d['share_pct']:.4f}%.", d)
        d = afr.retrieve_by_headline(term or str(args.get("headline", "")))
        if not d.get("matches"):
            return d.get("note", "No headline match."), d
        top = d["matches"][0]
        return (
            f"Closest AFR headline: \"{top['headline']}\" published "
            f"{top['publication_date']}.", d)

    return (
        f"Unknown dataset '{dataset}'. Available: rba, asx, afr. "
        f"Use the coverage tool to compare their date ranges.", {})


def _coverage(args: dict) -> tuple[str, dict]:
    """★ Compare date ranges across datasets.

    MHQ090's correct answer is a justified refusal worth 10 points across three
    components, and "No" alone earns 3.33 — the evidence-boundary reasoning
    carries the rest, so this returns the boundary explicitly.
    """
    r = rba.count()
    rows = list(corpora.rba_rows())
    d = {
        "rba": {
            "records": r["total_records"],
            "first_date": rows[0].date.isoformat(),
            "last_date": rows[-1].date.isoformat(),
        },
        "asx": asx.coverage(),
        "afr": afr.coverage(),
    }
    return (
        f"Coverage differs by dataset: RBA runs {d['rba']['first_date']} to "
        f"{d['rba']['last_date']}; ASX runs {d['asx']['first_date']} to "
        f"{d['asx']['last_date']}; AFR spans {d['afr']['first_year']} to "
        f"{d['afr']['last_year']}. Any analysis requiring ASX or AFR evidence after "
        f"2021 is therefore unsupported by the approved datasets.", d)


def _domain_sentiment(args: dict) -> tuple[str, dict]:
    """Role 2 — the ONE tool that calls Nemotron (kickoff §10).

    Required by Setup_Instructions.md L95. Returns a classification clamped to
    `config.SENTIMENT_CHAR_CAP`, never a full answer, and never a numeric
    forecast. Degrades to a note rather than raising, so the deterministic
    rate-lookup component of a sentiment question still scores if Nemotron is
    unreachable.
    """
    headline = str(args.get("headline", "")).strip()
    text = str(args.get("article_text", "")).strip()
    if not (headline or text):
        return (
            "ERROR: domain_sentiment needs a retrieved AFR article. Call "
            "query_data with dataset='afr', metric='retrieve_by_headline' first.",
            {},
        )

    # Frozen renderer, so training and inference build the identical message list.
    msgs = prompts.render_sentiment_messages(
        headline=headline or "(not supplied)",
        publication_date=str(args.get("publication_date", "") or "unknown"),
        excerpt=text[: config.AFR_TEXT_CHAR_CAP],
        cash_rate=str(args.get("rba_rate", "") or "unknown"),
    )
    try:
        out = domain_client.complete(
            msgs[0]["content"], msgs[1]["content"], max_tokens=120
        )
    except domain_client.DomainUnavailable as exc:
        note = (
            "Sentiment classification unavailable: the domain model could not be "
            "reached. The deterministic parts of this question can still be answered."
        )
        return note, {"sentiment_error": str(exc)}

    clamped = out.strip()[: config.SENTIMENT_CHAR_CAP]
    return clamped, {"sentiment_classification": clamped, "rba_rate": args.get("rba_rate")}


QUERY_DATA_PARAMETERS = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string", "enum": ["rba", "asx", "afr"]},
        "metric": {
            "type": "string",
            "description": (
                "RBA: count, count_changes, extremes, max_hold_streak, lookup_rate, "
                "cycle_summary. ASX: describe, annual_return, rank_annual_returns, "
                "full_sample_return, volatility, correlation, max_drawdown, avg_volume, "
                "window_return. AFR: count, count_by_month, share, retrieve_by_headline."
            ),
        },
        "ticker": {"type": "string", "description": "e.g. BHP.AX"},
        "ticker_b": {"type": "string", "description": "second ticker, for correlation"},
        "tickers": {"type": "array", "items": {"type": "string"}},
        "exclude_tickers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. [\"TAH.AX\"] to exclude Tabcorp",
        },
        "year": {"type": "integer"},
        "date": {"type": "string", "description": "ISO date, for lookup_rate"},
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "pattern": {
            "type": "string",
            "description": "AFR search term, matched as a whole word, case-insensitively",
        },
    },
    "required": ["dataset", "metric"],
}

ALL_TOOLS: list[Tool] = [
    Tool(
        "query_data",
        "Query the approved local RBA, ASX and AFR datasets deterministically. "
        "Use this for every count, date, ranking or calculation — never estimate.",
        QUERY_DATA_PARAMETERS,
        _query_data,
    ),
    Tool(
        "coverage",
        "Compare the date ranges of the RBA, ASX and AFR datasets. Use this before "
        "answering any question whose period may fall outside a dataset's coverage.",
        {"type": "object", "properties": {}},
        _coverage,
    ),
    Tool(
        "domain_sentiment",
        "Classify a retrieved AFR article's sentiment and the likely direction for the "
        "relevant ASX shares, given the RBA cash rate in force on its publication date. "
        "Retrieve the article first with query_data(dataset='afr', "
        "metric='retrieve_by_headline'). Returns a short classification, never a "
        "numeric forecast.",
        {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "article_text": {"type": "string"},
                "publication_date": {"type": "string"},
                "rba_rate": {"type": "string", "description": "cash-rate target in force"},
            },
            "required": ["headline"],
        },
        _domain_sentiment,
    ),
]

BRAIN_SCHEMAS: list[dict] = [t.schema for t in ALL_TOOLS]

REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
