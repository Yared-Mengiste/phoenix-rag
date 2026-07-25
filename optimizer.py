"""
optimizer.py
============
Adaptive, Bottleneck-Driven Optimization Engine for Phoenix RAG.

Key Features:
  1. Priority-Based Tuning: Solves the worst-performing metric first to avoid
     rule conflicts within a single iteration.
  2. Proportional Scaling: Step sizes scale dynamically based on distance to
     target.
  3. Context-Aware Mutations: Automatically activates score-threshold retriever
     modes when adjusting similarity thresholds.
  4. History-Damped Convergence: Tracks which numeric knob was last moved in
     which direction. If a metric wants to reverse a knob that was moved the
     OTHER way within the cooldown window, the step is damped (halved) rather
     than applied at full strength. This is what actually stops oscillation --
     two antagonistic metrics (context_recall wanting top_k up,
     response_relevancy wanting it down) will keep proposing valid, non-
     duplicate configs forever, so a plain "have I seen this config before"
     check never triggers. Only a per-parameter direction check catches it.
  5. Two-Tier Generation Fix: response_relevancy gets a prompt-only fix that
     explicitly tells the model to ignore tangential retrieved passages
     BEFORE falling back to pruning top_k -- top_k pruning is what directly
     fights context_recall's preferred direction, so it's now a true last
     resort instead of the immediate fallback.
"""

from __future__ import annotations

import logging
from config import OptimizerConfig, RetrievalConfig

logger = logging.getLogger("phoenix_rag.optimizer")

_EPS = 1e-4
_COOLDOWN_ITERATIONS = 2  # how many past moves count as "recent" for damping

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

# Tier 2 for response_relevancy specifically. Tried only after COMBINED_PROMPT
# is already active and relevancy is STILL the bottleneck. Targets the
# generation-side cause (model including tangential retrieved facts) without
# touching top_k, so it doesn't compete with context_recall for the same knob.
STRICT_FOCUS_PROMPT = (
    "You are a precise assistant. Answer the question directly and "
    "completely, using ONLY facts explicitly stated in the context below. "
    "Do not use outside knowledge. The context below may contain multiple "
    "passages -- some may be unrelated to the question. Identify only the "
    "passage(s) that actually answer the question and ignore the rest "
    "entirely; do not mention or summarize unrelated passages in your "
    "answer. If the context does not contain the answer, respond with "
    "\"I don't know based on the given context.\" Be concise.\n\n"
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


def _recent_opposite_move(history: list[dict], param: str, direction: str) -> bool:
    """True if `param` was moved in the OPPOSITE direction within the cooldown window."""
    recent = history[-_COOLDOWN_ITERATIONS:]
    for move in reversed(recent):
        if move.get("param") == param:
            return move.get("direction") != direction
    return False


def propose_next_config(
    current_config: RetrievalConfig,
    scores: dict[str, float],
    opt_config: OptimizerConfig,
    history: list[dict] | None = None,
) -> tuple[RetrievalConfig, list[str], dict | None]:
    """Proposes the next RetrievalConfig by identifying and fixing the primary bottleneck.

    Returns (next_config, applied_rules, move). `move` describes the single
    numeric knob changed this round (or None if only the prompt changed, or
    nothing changed) -- the caller appends this to `history` and passes the
    updated list back in on the next call.
    """
    history = history or []

    if meets_targets(scores, opt_config):
        logger.info("All metrics meet target thresholds.")
        return current_config, ["all_targets_met"], None

    targets = {
        "context_recall": opt_config.target_context_recall,
        "context_precision": opt_config.target_context_precision,
        "faithfulness": opt_config.target_faithfulness,
        "response_relevancy": opt_config.target_response_relevancy,
    }
    gaps = {m: max(0.0, targets[m] - scores.get(m, 1.0)) for m in targets}
    primary_bottleneck = max(gaps, key=gaps.get)
    gap_val = gaps[primary_bottleneck]

    if gap_val <= _EPS:
        logger.info("No significant metrics gap remaining.")
        return current_config, ["no_change_needed"], None

    next_config = current_config
    applied_rules: list[str] = []
    move: dict | None = None
    severity = min(2.0, max(0.5, gap_val / 0.2))

    # --- BOTTLENECK 1: Context Recall ---
    if primary_bottleneck == "context_recall":
        k_step = max(1, int(round(opt_config.top_k_step * severity)))
        if _recent_opposite_move(history, "top_k", "increase"):
            k_step = max(1, k_step // 2)
            logger.info("Damping top_k increase (recent decrease in history)")

        old_top_k = next_config.top_k
        new_top_k = _clamp(old_top_k + k_step, opt_config.top_k_bounds)

        if new_top_k == old_top_k and next_config.chunk_size > opt_config.chunk_size_bounds[0]:
            c_step = int(opt_config.chunk_size_step * severity)
            new_chunk = _clamp(next_config.chunk_size - c_step, opt_config.chunk_size_bounds)
            next_config = next_config.copy_with(chunk_size=int(new_chunk))
            applied_rules.append(f"bottleneck_recall: max_top_k_reached -> reduce_chunk_size_to_{int(new_chunk)}")
            move = {"param": "chunk_size", "direction": "decrease"}
        else:
            next_config = next_config.copy_with(top_k=int(new_top_k))
            applied_rules.append(f"bottleneck_recall: increase_top_k_by_{int(new_top_k) - old_top_k}")
            move = {"param": "top_k", "direction": "increase"}

    # --- BOTTLENECK 2: Context Precision ---
    elif primary_bottleneck == "context_precision":
        if next_config.retriever_type != "similarity_score_threshold":
            next_config = next_config.copy_with(
                retriever_type="similarity_score_threshold", similarity_threshold=0.3
            )
            applied_rules.append("bottleneck_precision: switch_retriever_to_similarity_score_threshold")
            move = {"param": "retriever_type", "direction": "switch"}
        else:
            thresh_step = opt_config.similarity_threshold_step * severity
            new_thresh = _clamp(
                next_config.similarity_threshold + thresh_step,
                opt_config.similarity_threshold_bounds,
            )
            if abs(new_thresh - next_config.similarity_threshold) < _EPS:
                k_step = max(1, int(round(opt_config.top_k_step * severity)))
                if _recent_opposite_move(history, "top_k", "decrease"):
                    k_step = max(1, k_step // 2)
                    logger.info("Damping top_k decrease (recent increase in history)")
                new_top_k = _clamp(next_config.top_k - k_step, opt_config.top_k_bounds)
                next_config = next_config.copy_with(top_k=int(new_top_k))
                applied_rules.append(f"bottleneck_precision: high_threshold -> reduce_top_k_by_{k_step}")
                move = {"param": "top_k", "direction": "decrease"}
            else:
                next_config = next_config.copy_with(similarity_threshold=float(new_thresh))
                applied_rules.append(f"bottleneck_precision: increase_similarity_threshold_by_{thresh_step:.3f}")
                move = {"param": "similarity_threshold", "direction": "increase"}

    # --- BOTTLENECK 3 & 4: Faithfulness or Response Relevancy ---
    elif primary_bottleneck in ("faithfulness", "response_relevancy"):
        if next_config.prompt_template not in (COMBINED_PROMPT, STRICT_FOCUS_PROMPT):
            next_config = next_config.copy_with(prompt_template=COMBINED_PROMPT)
            applied_rules.append(f"bottleneck_{primary_bottleneck}: apply_strict_combined_prompt")
            move = {"param": "prompt_template", "direction": "tier1"}
        elif primary_bottleneck == "response_relevancy" and next_config.prompt_template != STRICT_FOCUS_PROMPT:
            next_config = next_config.copy_with(prompt_template=STRICT_FOCUS_PROMPT)
            applied_rules.append("bottleneck_response_relevancy: apply_strict_focus_prompt")
            move = {"param": "prompt_template", "direction": "tier2"}
        else:
            if next_config.top_k > 2:
                if _recent_opposite_move(history, "top_k", "decrease"):
                    logger.info(
                        "bottleneck_%s wants top_k down but it was just raised "
                        "for context_recall; holding top_k, no lever left this round.",
                        primary_bottleneck,
                    )
                    applied_rules.append(
                        f"bottleneck_{primary_bottleneck}: top_k_reduction_blocked_recent_recall_increase"
                    )
                else:
                    new_top_k = next_config.top_k - 1
                    next_config = next_config.copy_with(top_k=new_top_k)
                    applied_rules.append(f"bottleneck_{primary_bottleneck}: prune_context_top_k_to_{new_top_k}")
                    move = {"param": "top_k", "direction": "decrease"}
            else:
                applied_rules.append(f"bottleneck_{primary_bottleneck}: prompt_and_top_k_already_constrained")
    logger.info("Primary Bottleneck: %s (gap: %.3f) | Applied: %s", primary_bottleneck, gap_val, applied_rules)
    return next_config, applied_rules, move