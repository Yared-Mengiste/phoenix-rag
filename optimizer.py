"""
optimizer.py
============
Rule-based optimization engine.

Two tiers per metric:
  - "low_*"          : score is below low_*_threshold (something's broken).
                        Apply a full-size step.
  - "near_target_*"  : score is above low_*_threshold but below target_*
                        (fine, but not at goal yet). Apply a smaller step.

faithfulness and response_relevancy share a single prompt_template slot.
Rather than two rules each proposing a different replacement prompt --
which silently clobber one another and oscillate, as seen in practice --
they're handled by ONE rule that switches to a single combined prompt
addressing both concerns at once.

All threshold/target comparisons tolerate a small amount of floating-point
noise (Ragas metrics rarely land on a clean decimal -- e.g.
0.949999999935 instead of 0.95), so a value effectively at a threshold
isn't treated as failing it.
"""

from __future__ import annotations

import logging

from config import OptimizerConfig, RetrievalConfig

logger = logging.getLogger("phoenix_rag.optimizer")

_EPS = 1e-6


# Kept for reference / backward-compat with old saved iteration JSONs.
# No longer used directly in propose_next_config -- see COMBINED_PROMPT.
STRENGTHENED_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY facts "
    "explicitly stated in the context below. Do not use outside knowledge. "
    "If the context does not contain the answer, respond with "
    "\"I don't know based on the given context.\" Be concise and avoid "
    "restating irrelevant parts of the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

IMPROVED_RELEVANCY_PROMPT = (
    "Answer the question directly and completely, using only the context "
    "provided. Stay strictly on-topic: do not add tangential information "
    "that was not asked for. If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Used whenever EITHER faithfulness or response_relevancy needs help.
# Merges grounding/no-hallucination instructions with on-topic/direct-
# answer instructions so the two metrics stop fighting over one slot.
COMBINED_PROMPT = (
    "You are a precise assistant. Answer the question directly and "
    "completely, using ONLY facts explicitly stated in the context below. "
    "Do not use outside knowledge, and do not add tangential information "
    "that was not asked for. If the context does not contain the answer, "
    "respond with \"I don't know based on the given context.\" Be "
    "concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _clamp(value: float | int, bounds: tuple[float, float]) -> float | int:
    low, high = bounds
    return max(low, min(high, value))


def _at_least(value: float, target: float) -> bool:
    """value >= target, tolerant of tiny float noise from LLM-judge metrics."""
    return value >= target - _EPS


def propose_next_config(
    current_config: RetrievalConfig,
    scores: dict[str, float],
    opt_config: OptimizerConfig,
) -> tuple[RetrievalConfig, list[str]]:
    next_config = current_config
    applied_rules: list[str] = []

    # ---- context_recall ----
    context_recall = scores.get("context_recall", 1.0)
    if not _at_least(context_recall, opt_config.low_context_recall_threshold):
        new_top_k = _clamp(
            next_config.top_k + opt_config.top_k_step, opt_config.top_k_bounds
        )
        new_chunk_size = _clamp(
            next_config.chunk_size - opt_config.chunk_size_step,
            opt_config.chunk_size_bounds,
        )
        next_config = next_config.copy_with(
            top_k=int(new_top_k), chunk_size=int(new_chunk_size)
        )
        applied_rules.append("low_context_recall: increase_top_k, reduce_chunk_size")
    elif not _at_least(context_recall, opt_config.target_context_recall):
        new_top_k = _clamp(next_config.top_k + 1, opt_config.top_k_bounds)
        next_config = next_config.copy_with(top_k=int(new_top_k))
        applied_rules.append("near_target_context_recall: nudge_top_k_up")

    # ---- context_precision ----
    context_precision = scores.get("context_precision", 1.0)
    if not _at_least(context_precision, opt_config.low_context_precision_threshold):
        new_top_k = _clamp(
            next_config.top_k - opt_config.top_k_step, opt_config.top_k_bounds
        )
        new_threshold = _clamp(
            next_config.similarity_threshold + opt_config.similarity_threshold_step,
            opt_config.similarity_threshold_bounds,
        )
        next_config = next_config.copy_with(
            top_k=int(new_top_k), similarity_threshold=float(new_threshold)
        )
        applied_rules.append(
            "low_context_precision: reduce_top_k, increase_similarity_threshold"
        )
    elif not _at_least(context_precision, opt_config.target_context_precision):
        new_threshold = _clamp(
            next_config.similarity_threshold + (opt_config.similarity_threshold_step / 2),
            opt_config.similarity_threshold_bounds,
        )
        next_config = next_config.copy_with(similarity_threshold=float(new_threshold))
        applied_rules.append("near_target_context_precision: nudge_similarity_threshold_up")

    # ---- faithfulness + response_relevancy (shared prompt_template slot) ----
    faithfulness = scores.get("faithfulness", 1.0)
    response_relevancy = scores.get("response_relevancy", 1.0)

    faithfulness_broken = not _at_least(faithfulness, opt_config.low_faithfulness_threshold)
    relevancy_broken = not _at_least(response_relevancy, opt_config.low_response_relevancy_threshold)
    faithfulness_off_target = not _at_least(faithfulness, opt_config.target_faithfulness)
    relevancy_off_target = not _at_least(response_relevancy, opt_config.target_response_relevancy)

    if (
        (faithfulness_off_target or relevancy_off_target)
        and next_config.prompt_template != COMBINED_PROMPT
    ):
        next_config = next_config.copy_with(prompt_template=COMBINED_PROMPT)
        if faithfulness_broken or relevancy_broken:
            applied_rules.append(
                "low_faithfulness_or_response_relevancy: apply_combined_prompt"
            )
        else:
            applied_rules.append(
                "near_target_faithfulness_or_response_relevancy: apply_combined_prompt"
            )

    if not applied_rules:
        logger.info("All metrics above thresholds; no configuration changes proposed")
    else:
        logger.info("Applied optimization rules: %s", applied_rules)

    return next_config, applied_rules


def meets_targets(scores: dict[str, float], opt_config: OptimizerConfig) -> bool:
    return (
        _at_least(scores.get("faithfulness", 0.0), opt_config.target_faithfulness)
        and _at_least(scores.get("context_recall", 0.0), opt_config.target_context_recall)
        and _at_least(scores.get("context_precision", 0.0), opt_config.target_context_precision)
        and _at_least(scores.get("response_relevancy", 0.0), opt_config.target_response_relevancy)
    )