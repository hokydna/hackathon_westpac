"""Frozen prompt strings, shared by the harness and the fine-tuning run.

FROZEN in the base commit (SESSION_KICKOFF.md §4, F1). Imported by:

  * src/agent/synth.py         (session B)  -- Role 1, final synthesis
  * src/tools/registry.py      (session A)  -- Role 2, domain_sentiment tool
  * training/prepare_data.py   (session D)  -- trains against BOTH pairs

**Editing any string here silently invalidates the adapter.** The model is
trained on these exact bytes; serving it a different prompt is the one failure
mode that turns three hours of training into noise. If a prompt genuinely has
to change, it changes here, before D generates data, and D regenerates.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Role 1 -- final synthesis. NOT a tool. Runs unconditionally after the loop.
#
# Verbatim from FINETUNE_PLAN.md §4.1. Measured at ~95 tokens against the real
# Nemotron tokenizer (node1 /tokenize).
#
# The "single sentence" clause is not stylistic. Four of the fifteen public
# questions are a single 10-point all-or-nothing component bundling 3-4 numbers
# (MHQ001, MHQ040, MHQ049, MHQ076) -- 26.7% of public points. Splitting a
# compound fact across two sentences is how a perfectly-computed answer scores
# zero. This clause is the cheapest available defence on those points.
# --------------------------------------------------------------------------

SYNTH_SYSTEM = (
    "You are a financial data analyst. You are given a question and verified results "
    "from deterministic data tools. Write ONE concise answer that states every fact the "
    "question asks for, using the exact values from the tool results. When the question "
    "asks for several numbers that belong together, state all of them in a single "
    "sentence. Do not hedge. Do not add facts that are not in the tool results. If the "
    "tool results are insufficient to answer, say so plainly and state what is missing."
)

SYNTH_USER = "Question: {question}\n\nVerified tool results:\n{tool_results}"


# --------------------------------------------------------------------------
# Role 2 -- sentiment classification. IS a tool Qwen calls (domain_sentiment).
#
# Required by Setup_Instructions.md L95: article-grounded sentiment questions
# route the retrieved AFR text AND the applicable RBA rate through
# DOMAIN_FT_MODEL, returning a sentiment classification (positive / negative /
# mixed) and a likely market direction. "Do not force the model to emit a
# made-up numeric return or price forecast."
#
# Worth 30 of the 150 public points (MHQ058, MHQ067, MHQ080), where the
# sentiment and direction clauses carry 4 points each and the rate lookup 2.
#
# Output is CLAMPED to config.SENTIMENT_CHAR_CAP by the tool. Three sentences
# is the shape the reference answers use, and it fits the cap -- a rambling
# classification loses its direction clause to truncation.
# --------------------------------------------------------------------------

SENTIMENT_SYSTEM = (
    "You are a financial analyst classifying a news article. You are given an article "
    "and the RBA cash-rate target in force on its publication date. Reply in exactly "
    "three sentences and nothing else: first state the RBA cash-rate target in force, "
    "then classify the article's sentiment as positive, negative, or mixed (a bias such "
    "as 'mixed with a negative bias' is allowed), then state the likely direction for "
    "the relevant ASX shares. Never forecast a numeric value, percentage, or price. "
    "Never add a fourth sentence."
)

SENTIMENT_USER = (
    "Article headline: {headline}\n"
    "Published: {publication_date}\n"
    "Text: {article_text}\n"
    "RBA cash-rate target in force on that date: {rba_rate}%"
)
