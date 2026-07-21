"""
optimizer.py
============
Rule-based optimization engine. Looks at the Ragas scores from the most
recent iteration and nudges the retrieval configuration according to the
`optimization_rules` from the spec:

    low_context_recall      -> increase_top_k, reduce_chunk_size
    low_context_precision   -> reduce_top_k, increase_similarity_threshold
    low_faithfulness        -> strengthen_prompt, reduce_irrelevant_context
    low_response_relevancy  -> improve_prompt_instruction

Only retrieval parameters change between iterations -- the benchmark
question set is never touched here.
"""

from __future__ import annotations

import logging

from config import OptimizerConfig, RetrievalConfig

logger = logging.getLogger("phoenix_rag.optimizer")


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


def _clamp(value: float | int, bounds: tuple[float, float]) -> float | int:
    low, high = bounds
    return max(low, min(high, value))


def propose_next_config(
    current_config: RetrievalConfig,
    scores: dict[str, float],
    opt_config: OptimizerConfig,
) -> tuple[RetrievalConfig, list[str]]:
    """Given the current config and its evaluation scores, propose the next config.

    Returns (new_config, applied_rules) so the caller can log/persist which
    rules fired for this iteration.
    """
    next_config = current_config
    applied_rules: list[str] = []

    if scores.get("context_recall", 1.0) < opt_config.low_context_recall_threshold:
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

    if scores.get("context_precision", 1.0) < opt_config.low_context_precision_threshold:
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

    if scores.get("faithfulness", 1.0) < opt_config.low_faithfulness_threshold:
        next_config = next_config.copy_with(prompt_template=STRENGTHENED_PROMPT)
        applied_rules.append(
            "low_faithfulness: strengthen_prompt, reduce_irrelevant_context"
        )

    if (
        scores.get("response_relevancy", 1.0)
        < opt_config.low_response_relevancy_threshold
        and "low_faithfulness: strengthen_prompt, reduce_irrelevant_context"
        not in applied_rules
    ):
        # Don't clobber the faithfulness prompt fix if both fired this round.
        next_config = next_config.copy_with(prompt_template=IMPROVED_RELEVANCY_PROMPT)
        applied_rules.append("low_response_relevancy: improve_prompt_instruction")

    if not applied_rules:
        logger.info("All metrics above thresholds; no configuration changes proposed")
    else:
        logger.info("Applied optimization rules: %s", applied_rules)

    return next_config, applied_rules


def meets_targets(scores: dict[str, float], opt_config: OptimizerConfig) -> bool:
    return (
        scores.get("faithfulness", 0.0) >= opt_config.target_faithfulness
        and scores.get("context_recall", 0.0) >= opt_config.target_context_recall
        and scores.get("context_precision", 0.0) >= opt_config.target_context_precision
        and scores.get("response_relevancy", 0.0) >= opt_config.target_response_relevancy
    )
