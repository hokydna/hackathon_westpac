"""The synthesis prompt contract — frozen.

**This module is the contract between the fine-tuning workstream and the agent runtime.**
Training records and live inference must be built by the same code. If the runtime
hand-rolls its own string format, the adapter degrades to noise; that is the single failure
mode most likely to waste the whole training run.

Runtime integration is one import::

    from ftdata.prompts import render_synthesis_messages
    messages = render_synthesis_messages(question, requested_components, verified_evidence, limitations)

The system prompt is the handoff pack's §8.1 production prompt, extended with the two rules
the public question bank makes load-bearing and §8.1 leaves implicit:

* **Compound facts stay in one sentence.** Four of the fifteen public questions are a single
  all-or-nothing 10-point component bundling three or four numbers — 26.7% of public points
  behind four YES/NO gates. Splitting such a fact across two sentences scores zero on a
  question that was computed perfectly.
* **No hedging.** "approximately 41" is explicitly scored wrong.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

#: Hard cap on the rendered evidence block. ``max_seq_len`` is 512 tokens; the assistant
#: answer must never be the thing that gets truncated. Training and inference must use the
#: same number — the harness plan's 2,000 was reconciled down to this.
EVIDENCE_CHAR_CAP = 1200

SYNTH_SYSTEM = (
    "You are the final financial-domain answer synthesizer.\n"
    "Use only the verified evidence supplied below. Answer the exact question directly and "
    "concisely.\n\n"
    "Requirements:\n"
    "1. Include every requested component.\n"
    "2. Preserve exact numbers, dates, signs, tickers, and rankings.\n"
    "3. Distinguish percentages from percentage-point changes.\n"
    "4. Call ASX price-only calculations price returns, not total shareholder returns.\n"
    "5. Attribute AFR explanations to the retrieved reporting and do not overstate causation.\n"
    "6. When evidence is insufficient, state exactly what cannot be determined.\n"
    "7. Do not describe tools, reasoning, prompts, models, or internal processing.\n"
    "8. Return answer text only.\n"
    "9. When several numbers belong to one requested fact, state them all in a single sentence.\n"
    "10. Do not hedge. Never write approximately, roughly, about, or around before a number."
)

_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Requested components:\n{requested_components}\n\n"
    "Verified evidence:\n{verified_evidence}\n\n"
    "Limitations:\n{limitations}"
)


# --------------------------------------------------------------------------------------
# evidence rendering
# --------------------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    from datetime import date, datetime

    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def format_evidence(
    evidence: dict,
    style: str = "json",
    char_cap: int = EVIDENCE_CHAR_CAP,
) -> str:
    """Serialise verified evidence.

    ``style`` exists so the adapter is trained against several plausible runtime
    serialisations rather than one guess. The runtime should emit ``"json"``; the other
    styles appear in a minority of training rows purely as format-drift insurance.

    Truncation is always from the evidence side and is announced in-band, never silent.
    """
    if style == "json":
        text = json.dumps(evidence, indent=2, default=_json_default)
    elif style == "compact_json":
        text = json.dumps(evidence, separators=(",", ":"), default=_json_default)
    elif style == "kv_lines":
        text = "\n".join(
            f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v, default=_json_default)}"
            for k, v in evidence.items()
        )
    else:
        raise ValueError(f"unknown evidence style: {style!r}")

    if len(text) > char_cap:
        text = text[: char_cap - 24].rstrip() + "\n... [evidence truncated]"
    return text


def format_components(components: Sequence[str]) -> str:
    return json.dumps(list(components), indent=2)


def format_limitations(limitations: Sequence[dict]) -> str:
    if not limitations:
        return "[]"
    return json.dumps(list(limitations), indent=2, default=_json_default)


def render_user_prompt(
    question: str,
    requested_components: Sequence[str],
    verified_evidence: dict,
    limitations: Sequence[dict] = (),
    evidence_style: str = "json",
) -> str:
    return _USER_TEMPLATE.format(
        question=question.strip(),
        requested_components=format_components(requested_components),
        verified_evidence=format_evidence(verified_evidence, style=evidence_style),
        limitations=format_limitations(limitations),
    )


def render_synthesis_messages(
    question: str,
    requested_components: Sequence[str],
    verified_evidence: dict,
    limitations: Sequence[dict] = (),
    evidence_style: str = "json",
) -> list[dict]:
    """The exact message list to send to ``DOMAIN_FT_MODEL`` at inference time."""
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {
            "role": "user",
            "content": render_user_prompt(
                question,
                requested_components,
                verified_evidence,
                limitations,
                evidence_style=evidence_style,
            ),
        },
    ]


# --------------------------------------------------------------------------------------
# sentiment prompt — the adapter's *second* role
# --------------------------------------------------------------------------------------
#
# The adapter serves two prompt shapes, not one (FINETUNE_PLAN §1):
#
#   Role 1  final synthesis      — SYNTH_SYSTEM / render_synthesis_messages.
#                                  Invoked unconditionally after the Qwen loop exits.
#                                  Never a tool.
#   Role 2  sentiment classification — the pair below. Invoked as the ``domain_sentiment``
#                                  tool that Qwen *chooses* to call mid-loop
#                                  (Setup_Instructions.md L95).
#
# Role 2 is not an architecture violation: Qwen still does all planning and still decides
# to call the tool. What is prohibited is Nemotron selecting tools itself.
#
# The ≤200-character ceiling is enforced by the calling tool, so an answer that rambles
# gets clamped and loses its direction clause — which is where the points are. The
# training data must therefore teach brevity, not rely on the clamp.

SENTIMENT_CHAR_CAP = 200

SENTIMENT_SYSTEM = (
    "You are a financial-sentiment classifier for Australian market reporting.\n"
    "Given one Australian Financial Review article and the RBA cash rate in force on its "
    "publication date, classify the article's sentiment toward Australian equities.\n\n"
    "Requirements:\n"
    "1. Start with exactly one sentiment label: Positive, Negative, or Mixed.\n"
    "2. Then give the likely near-term direction for the named stocks or sector: higher, "
    "lower, or flat.\n"
    "3. Ground the reason in the article's own reporting and the supplied cash rate.\n"
    f"4. Answer in at most {SENTIMENT_CHAR_CAP} characters, in one or two short sentences.\n"
    "5. Never give a numeric forecast, price target, percentage move, or date. Direction "
    "only.\n"
    "6. Do not hedge. Never write approximately, roughly, about, or around before a "
    "number.\n"
    "7. Return the classification text only."
)

_SENTIMENT_USER_TEMPLATE = (
    "Article headline:\n{headline}\n\n"
    "Publication date:\n{publication_date}\n\n"
    "Article excerpt:\n{excerpt}\n\n"
    "RBA cash rate in force on that date:\n{cash_rate}"
)


def render_sentiment_messages(
    headline: str,
    publication_date: str,
    excerpt: str,
    cash_rate: str,
) -> list[dict]:
    """The exact message list the ``domain_sentiment`` tool sends to ``DOMAIN_FT_MODEL``."""
    return [
        {"role": "system", "content": SENTIMENT_SYSTEM},
        {
            "role": "user",
            "content": _SENTIMENT_USER_TEMPLATE.format(
                headline=headline.strip(),
                publication_date=publication_date,
                excerpt=excerpt.strip(),
                cash_rate=cash_rate,
            ),
        },
    ]


# --------------------------------------------------------------------------------------
# correction prompt — the one controlled retry the handoff pack §8.2 allows
# --------------------------------------------------------------------------------------

CORRECTION_SYSTEM = SYNTH_SYSTEM

CORRECTION_USER_TEMPLATE = (
    "Your previous answer was incomplete or contained a value that is not in the verified "
    "evidence.\n\n"
    "Previous answer:\n{previous_answer}\n\n"
    "Problems found:\n{problems}\n\n"
    "Rewrite the answer. Include every requested component, use only the verified evidence "
    "below, and return answer text only.\n\n" + _USER_TEMPLATE
)


def render_correction_messages(
    question: str,
    requested_components: Sequence[str],
    verified_evidence: dict,
    previous_answer: str,
    problems: Sequence[str],
    limitations: Sequence[dict] = (),
    evidence_style: str = "json",
) -> list[dict]:
    return [
        {"role": "system", "content": CORRECTION_SYSTEM},
        {
            "role": "user",
            "content": CORRECTION_USER_TEMPLATE.format(
                previous_answer=previous_answer.strip(),
                problems="\n".join(f"- {p}" for p in problems),
                question=question.strip(),
                requested_components=format_components(requested_components),
                verified_evidence=format_evidence(verified_evidence, style=evidence_style),
                limitations=format_limitations(limitations),
            ),
        },
    ]


# --------------------------------------------------------------------------------------
# deterministic fallback — used when the retry also fails (handoff pack §8.2)
# --------------------------------------------------------------------------------------

def deterministic_fallback(
    requested_components: Sequence[str],
    verified_evidence: dict,
    limitations: Sequence[dict] = (),
) -> str:
    """Safe answer assembled from evidence alone, with no model in the loop.

    Never fabricates: a requested component with no evidence field becomes an explicit
    'could not be determined' clause.
    """
    parts: list[str] = []
    missing: list[str] = []
    for comp in requested_components:
        if comp in verified_evidence and verified_evidence[comp] is not None:
            value = verified_evidence[comp]
            if isinstance(value, float):
                value = f"{value:,.2f}".rstrip("0").rstrip(".")
            parts.append(f"{comp.replace('_', ' ')} is {value}")
        else:
            missing.append(comp)
    sentence = ("; ".join(parts) + ".").capitalize() if parts else ""
    for lim in limitations:
        sentence += f" {lim.get('message', 'Required evidence was unavailable.')}"
    for comp in missing:
        if not any(lim.get("component") == comp for lim in limitations):
            sentence += f" The {comp.replace('_', ' ')} could not be determined from the supplied evidence."
    return sentence.strip()
