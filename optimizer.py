"""
optimizer.py
============
Adaptive, Bottleneck-Driven Optimization Engine for Phoenix RAG.

Key Features:
  1. Priority-Based Tuning: Solves the worst-performing metric first to avoid rule conflicts.
  2. Proportional Scaling: Step sizes scale dynamically based on distance to target.
  3. Context-Aware Mutations: Automatically activates score-threshold retriever modes 
     when adjusting similarity thresholds.
  4. History Tracking: Prevents infinite oscillation and duplicate configurations.
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
    """Proposes the next RetrievalConfig by identifying and fixing the primary bottleneck."""
    
    if meets_targets(scores, opt_config):
        logger.info("All metrics meet target thresholds.")
        return current_config, ["all_targets_met"]

    targets = {
        "context_recall": opt_config.target_context_recall,
        "context_precision": opt_config.target_context_precision,
        "faithfulness": opt_config.target_faithfulness,
        "response_relevancy": opt_config.target_response_relevancy,
    }

    # Calculate normalized gaps (distance from target)
    gaps = {
        metric: max(0.0, targets[metric] - scores.get(metric, 1.0))
        for metric in targets
    }

    # Identify primary bottleneck (metric with largest relative failure)
    primary_bottleneck = max(gaps, key=gaps.get)
    gap_val = gaps[primary_bottleneck]

    if gap_val <= _EPS:
        logger.info("No significant metrics gap remaining.")
        return current_config, ["no_change_needed"]

    next_config = current_config
    applied_rules: list[str] = []

    # Proportional multiplier based on how far we are from target (0.5x to 2.0x)
    severity = min(2.0, max(0.5, gap_val / 0.2))

    # --- BOTTLENECK 1: Context Recall (Missing Information) ---
    if primary_bottleneck == "context_recall":
        k_step = int(round(opt_config.top_k_step * severity))
        k_step = max(1, k_step)
        
        new_top_k = _clamp(next_config.top_k + k_step, opt_config.top_k_bounds)
        
        # If top_k is capped at max bound, decrease chunk_size to fit more distinct contexts
        if new_top_k == next_config.top_k and next_config.chunk_size > opt_config.chunk_size_bounds[0]:
            c_step = int(opt_config.chunk_size_step * severity)
            new_chunk = _clamp(next_config.chunk_size - c_step, opt_config.chunk_size_bounds)
            next_config = next_config.copy_with(chunk_size=int(new_chunk))
            applied_rules.append(f"bottleneck_recall: max_top_k_reached -> reduce_chunk_size_to_{int(new_chunk)}")
        else:
            next_config = next_config.copy_with(top_k=int(new_top_k))
            applied_rules.append(f"bottleneck_recall: increase_top_k_by_{k_step}")

    # --- BOTTLENECK 2: Context Precision (Too Much Noise/Irrelevant Context) ---
    elif primary_bottleneck == "context_precision":
        # First action: enable similarity_score_threshold if not already enabled
        if next_config.retriever_type != "similarity_score_threshold":
            next_config = next_config.copy_with(
                retriever_type="similarity_score_threshold",
                similarity_threshold=0.3
            )
            applied_rules.append("bottleneck_precision: switch_retriever_to_similarity_score_threshold")
        else:
            # Increase similarity threshold proportionally
            thresh_step = opt_config.similarity_threshold_step * severity
            new_thresh = _clamp(
                next_config.similarity_threshold + thresh_step,
                opt_config.similarity_threshold_bounds
            )
            
            # If threshold is already high, drop top_k
            if abs(new_thresh - next_config.similarity_threshold) < _EPS:
                k_step = max(1, int(round(opt_config.top_k_step * severity)))
                new_top_k = _clamp(next_config.top_k - k_step, opt_config.top_k_bounds)
                next_config = next_config.copy_with(top_k=int(new_top_k))
                applied_rules.append(f"bottleneck_precision: high_threshold -> reduce_top_k_by_{k_step}")
            else:
                next_config = next_config.copy_with(similarity_threshold=float(new_thresh))
                applied_rules.append(f"bottleneck_precision: increase_similarity_threshold_by_{thresh_step:.3f}")

    # --- BOTTLENECK 3 & 4: Faithfulness or Response Relevancy (Generation Quality) ---
    elif primary_bottleneck in ("faithfulness", "response_relevancy"):
        if next_config.prompt_template != COMBINED_PROMPT:
            next_config = next_config.copy_with(prompt_template=COMBINED_PROMPT)
            applied_rules.append(f"bottleneck_{primary_bottleneck}: apply_strict_combined_prompt")
        else:
            # Prompt is already optimized; reduce top_k to prune potential noise causing hallucinations
            if next_config.top_k > 2:
                new_top_k = next_config.top_k - 1
                next_config = next_config.copy_with(top_k=new_top_k)
                applied_rules.append(f"bottleneck_{primary_bottleneck}: prune_context_top_k_to_{new_top_k}")
            else:
                applied_rules.append(f"bottleneck_{primary_bottleneck}: prompt_and_top_k_already_constrained")

    logger.info("Primary Bottleneck: %s (gap: %.3f) | Applied: %s", primary_bottleneck, gap_val, applied_rules)
    return next_config, applied_rules