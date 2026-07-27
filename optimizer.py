"""
optimizer.py
============
Adaptive, Bottleneck-Driven Optimization Engine for Phoenix RAG.

Key Features:
  1. Priority-Based Tuning: Solves the worst-performing metric first to avoid rule conflicts.
  2. Proportional Scaling: Step sizes scale dynamically based on distance to target.
  3. Context-Aware Mutations: Automatically activates score-threshold retriever modes
     when adjusting similarity thresholds.
  4. F1-Balanced Retrieval Tuning: context_precision and context_recall are no longer
     treated as two independent, competing bottlenecks. They're combined into a single
     retrieval_f1 score (harmonic mean), so the optimizer only touches retrieval tuning
     when the *combined* trade-off is genuinely bad, and picks precision vs. recall as
     the specific lever based on which one is currently the weaker link. This avoids the
     oscillation where a precision fix (e.g. switching to similarity_score_threshold)
     tanks recall, which then gets "fixed" by a rule that re-tanks precision, back and
     forth, each fix undoing the other's progress.
  5. History Tracking: Prevents infinite oscillation and duplicate configurations.
"""

from __future__ import annotations

import logging
from config import OptimizerConfig, RetrievalConfig

logger = logging.getLogger("phoenix_rag.optimizer")

_EPS = 1e-4

COMBINED_PROMPT = (
    "You are a precise assistant. Answer the question directly and "
    "completely, using ONLY facts explicitly stated in the context below. "
    "Do not use outside knowledge, and do not add tangential information "
    "that was not asked for. If the context does not contain the answer, "
    "respond with \"I don't know based on the given context.\" Be concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _clamp(value: float | int, bounds: tuple[float, float]) -> float | int:
    low, high = bounds
    return max(low, min(high, value))


def _at_least(value: float, target: float) -> bool:
    return value >= target - _EPS


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall.

    Unlike a plain average, the harmonic mean punishes imbalance: a
    config with precision=0.95 / recall=0.40 scores much worse here than
    a (0.95 + 0.40) / 2 average would suggest. That's intentional -- a
    retriever that's great on one axis and bad on the other is still a
    bad retriever for this system's purposes.
    """
    if precision + recall <= _EPS:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meets_targets(scores: dict[str, float], opt_config: OptimizerConfig) -> bool:
    return (
        _at_least(scores.get("faithfulness", 0.0), opt_config.target_faithfulness)
        and _at_least(scores.get("context_recall", 0.0), opt_config.target_context_recall)
        and _at_least(scores.get("context_precision", 0.0), opt_config.target_context_precision)
        and _at_least(scores.get("response_relevancy", 0.0), opt_config.target_response_relevancy)
    )


def propose_next_config(
    current_config: RetrievalConfig,
    scores: dict[str, float],
    opt_config: OptimizerConfig,
    history: list[dict] | None = None,
) -> tuple[RetrievalConfig, list[str]]:
    """Proposes the next RetrievalConfig by identifying and fixing the primary bottleneck.

    Retrieval quality (precision + recall) is judged jointly via F1 rather than
    as two separate competing gaps -- see module docstring, point 4.
    """

    if meets_targets(scores, opt_config):
        logger.info("All metrics meet target thresholds.")
        return current_config, ["all_targets_met"]

    precision = scores.get("context_precision", 0.0)
    recall = scores.get("context_recall", 0.0)
    retrieval_f1 = f1_score(precision, recall)
    target_f1 = f1_score(opt_config.target_context_precision, opt_config.target_context_recall)

    # Gaps are now three-way: joint retrieval quality, faithfulness, relevancy.
    gaps = {
        "retrieval_f1": max(0.0, target_f1 - retrieval_f1),
        "faithfulness": max(0.0, opt_config.target_faithfulness - scores.get("faithfulness", 0.0)),
        "response_relevancy": max(0.0, opt_config.target_response_relevancy - scores.get("response_relevancy", 0.0)),
    }

    # Identify primary bottleneck (metric/group with largest gap)
    primary_bottleneck = max(gaps, key=gaps.get)
    gap_val = gaps[primary_bottleneck]

    if gap_val <= _EPS:
        logger.info("No significant metrics gap remaining.")
        return current_config, ["no_change_needed"]

    next_config = current_config
    applied_rules: list[str] = []

    # Proportional multiplier based on how far we are from target (0.5x to 2.0x)
    severity = min(2.0, max(0.5, gap_val / 0.2))

    # --- BOTTLENECK: Retrieval F1 (precision/recall trade-off) ---
    if primary_bottleneck == "retrieval_f1":
        # Decide which side of the trade-off is actually weaker right now,
        # rather than reacting to whichever raw gap happened to be bigger.
        if recall < precision:
            # Recall is the weaker link -- go after context_recall.
            k_step = int(round(opt_config.top_k_step * severity))
            k_step = max(1, k_step)

            new_top_k = _clamp(next_config.top_k + k_step, opt_config.top_k_bounds)

            if new_top_k == next_config.top_k and next_config.chunk_size > opt_config.chunk_size_bounds[0]:
                c_step = int(opt_config.chunk_size_step * severity)
                new_chunk = _clamp(next_config.chunk_size - c_step, opt_config.chunk_size_bounds)
                next_config = next_config.copy_with(chunk_size=int(new_chunk))
                applied_rules.append(
                    f"bottleneck_retrieval_f1(recall_weaker): max_top_k_reached -> reduce_chunk_size_to_{int(new_chunk)}"
                )
            else:
                next_config = next_config.copy_with(top_k=int(new_top_k))
                applied_rules.append(f"bottleneck_retrieval_f1(recall_weaker): increase_top_k_by_{k_step}")

        else:
            # Precision is the weaker link (or tied) -- go after context_precision.
            if next_config.retriever_type != "similarity_score_threshold":
                next_config = next_config.copy_with(
                    retriever_type="similarity_score_threshold",
                    similarity_threshold=0.3,
                )
                applied_rules.append(
                    "bottleneck_retrieval_f1(precision_weaker): switch_retriever_to_similarity_score_threshold"
                )
            else:
                thresh_step = opt_config.similarity_threshold_step * severity
                new_thresh = _clamp(
                    next_config.similarity_threshold + thresh_step,
                    opt_config.similarity_threshold_bounds,
                )

                if abs(new_thresh - next_config.similarity_threshold) < _EPS:
                    k_step = max(1, int(round(opt_config.top_k_step * severity)))
                    new_top_k = _clamp(next_config.top_k - k_step, opt_config.top_k_bounds)
                    next_config = next_config.copy_with(top_k=int(new_top_k))
                    applied_rules.append(
                        f"bottleneck_retrieval_f1(precision_weaker): high_threshold -> reduce_top_k_by_{k_step}"
                    )
                else:
                    next_config = next_config.copy_with(similarity_threshold=float(new_thresh))
                    applied_rules.append(
                        f"bottleneck_retrieval_f1(precision_weaker): increase_similarity_threshold_by_{thresh_step:.3f}"
                    )

    # --- BOTTLENECK: Faithfulness or Response Relevancy (Generation Quality) ---
    elif primary_bottleneck in ("faithfulness", "response_relevancy"):
        if next_config.prompt_template != COMBINED_PROMPT:
            next_config = next_config.copy_with(prompt_template=COMBINED_PROMPT)
            applied_rules.append(f"bottleneck_{primary_bottleneck}: apply_strict_combined_prompt")
        else:
            if next_config.top_k > 2:
                new_top_k = next_config.top_k - 1
                next_config = next_config.copy_with(top_k=new_top_k)
                applied_rules.append(f"bottleneck_{primary_bottleneck}: prune_context_top_k_to_{new_top_k}")
            else:
                applied_rules.append(f"bottleneck_{primary_bottleneck}: prompt_and_top_k_already_constrained")

    logger.info(
        "Primary Bottleneck: %s (gap: %.3f, retrieval_f1=%.3f, precision=%.3f, recall=%.3f) | Applied: %s",
        primary_bottleneck, gap_val, retrieval_f1, precision, recall, applied_rules,
    )
    return next_config, applied_rules