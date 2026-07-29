"""
prompt_refiner.py
==================
Asks an LLM to rewrite the RAG prompt template when the rule-based
optimizer has run out of levers -- prompt already strict (COMBINED_PROMPT
in optimizer.py), top_k already at floor -- but a generation-quality
metric (faithfulness / response_relevancy) is still failing its target.

This replaces what used to be a dead end (the
"prompt_and_top_k_already_constrained" no-op in optimizer.py, which just
re-scored the same unchanged config on every subsequent iteration) with a
real next step: a prompt tailored to what the document actually is, and
to which specific metric is still struggling.
"""

from __future__ import annotations

import logging
import re

from config import MistralSettings
from mistral_client import MistralClient

logger = logging.getLogger("phoenix_rag.prompt_refiner")

REQUIRED_PLACEHOLDERS = ("{context}", "{question}")

REFINER_SYSTEM_PROMPT = """You are an expert prompt engineer for RAG \
(Retrieval-Augmented Generation) systems. You will be given:
  1. A summary of the source document the system answers questions about.
  2. The current prompt template being used.
  3. Which evaluation metric is failing and why that metric matters.

Write an IMPROVED prompt template that keeps the same overall structure \
but is better tailored to this specific document and this specific \
failure mode. The template MUST contain exactly one occurrence each of \
the literal placeholders {context} and {question} -- these get filled in \
programmatically, do not remove, rename, or duplicate them.

Return ONLY the new prompt template text. No explanation, no markdown \
fences, no commentary -- just the template itself."""

METRIC_EXPLANATIONS = {
    "faithfulness": (
        "faithfulness -- the generated answer is not staying grounded in the "
        "retrieved context; it may be adding facts not present in the context "
        "or drifting from what the context actually supports."
    ),
    "response_relevancy": (
        "response_relevancy -- the generated answer is not staying focused on "
        "what the question actually asked; it may be too hedged, too broad, "
        "or padded with tangential information not asked for."
    ),
}


def _validate_template(template: str) -> bool:
    """A refined template is only usable if it kept both placeholders,
    each exactly once. Silently accepting a broken template (missing or
    duplicated placeholders) would crash str.format() downstream in
    rag_pipeline.py's build_prompt(), so this is checked before the
    template is ever installed into a RetrievalConfig.
    """
    return all(template.count(ph) == 1 for ph in REQUIRED_PLACEHOLDERS)


def refine_prompt(
    current_prompt: str,
    document_summary: str,
    failing_metric: str,
    mistral_settings: MistralSettings,
) -> str | None:
    """Ask the LLM for an improved prompt template.

    Returns None (rather than installing a malformed template) if the
    LLM's response is missing the required placeholders -- callers should
    keep the existing prompt in that case rather than risk a runtime
    crash from a template that doesn't format cleanly.
    """
    client = MistralClient(mistral_settings)
    metric_note = METRIC_EXPLANATIONS.get(
        failing_metric, f"{failing_metric} -- this metric is below target."
    )

    user_message = (
        f"DOCUMENT SUMMARY:\n{document_summary}\n\n"
        f"CURRENT PROMPT TEMPLATE:\n{current_prompt}\n\n"
        f"FAILING METRIC:\n{metric_note}\n\n"
        "Write an improved prompt template addressing this failure mode, "
        "specific to this document."
    )

    raw = client.chat(
        messages=[
            {"role": "system", "content": REFINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=mistral_settings.generation_model,
        temperature=0.3,
    )

    new_template = raw.strip()
    # Strip stray code fences defensively, same pattern as
    # question_generator.py's _parse_llm_json.
    new_template = re.sub(r"^```\w*\n?", "", new_template)
    new_template = re.sub(r"```$", "", new_template).strip()

    if not _validate_template(new_template):
        logger.warning(
            "LLM-refined prompt missing/duplicating required placeholders %s; "
            "discarding, keeping existing prompt.",
            REQUIRED_PLACEHOLDERS,
        )
        return None

    logger.info("Prompt refined for failing metric '%s'", failing_metric)
    return new_template