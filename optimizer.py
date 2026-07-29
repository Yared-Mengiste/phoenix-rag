"""
optimizer.py
============
Adaptive, Bottleneck-Driven Optimization Engine for Phoenix RAG.
[... same docstring as before, feature 6 now accurately describes what's implemented ...]
"""

from __future__ import annotations

import logging
from config import MistralSettings, OptimizerConfig, RetrievalConfig
from prompt_refiner import refine_prompt

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
    mistral_settings: MistralSettings | None = None,
    document_summary: str | None = None,
    state: dict | None = None,
) -> tuple[RetrievalConfig, list[str], dict]:
    """Proposes the next RetrievalConfig by identifying and fixing the primary bottleneck.

    `state` tracks progress through the generation-quality tiers explicitly,
    rather than inferring "which tier are we at" from string-comparing
    prompt_template against COMBINED_PROMPT. That inference broke as soon as
    a third tier (LLM refinement) was added: an LLM-refined prompt is, by
    definition, never equal to the COMBINED_PROMPT constant, so the old code
    read "not COMBINED_PROMPT" as "tier 1 not yet tried" even after tier 3
    had already run -- silently discarding the LLM's refined prompt and
    regressing backward every subsequent iteration.

    `state["prompt_tier"]` only ever increases (0 -> 1 -> 2 -> 3), so a tier
    already reached is never re-applied or reverted.

    Returns (next_config, applied_rules, updated_state). Caller persists
    `updated_state` and passes it back in on the next call.
    """
    state = dict(state) if state else {"prompt_tier": 0, "top_k_pruned": False}

    if meets_targets(scores, opt_config):
        logger.info("All metrics meet target thresholds.")
        return current_config, ["all_targets_met"], state

    precision = scores.get("context_precision", 0.0)
    recall = scores.get("context_recall", 0.0)
    retrieval_f1 = f1_score(precision, recall)
    target_f1 = f1_score(opt_config.target_context_precision, opt_config.target_context_recall)

    gaps = {
        "retrieval_f1": max(0.0, target_f1 - retrieval_f1),
        "faithfulness": max(0.0, opt_config.target_faithfulness - scores.get("faithfulness", 0.0)),
        "response_relevancy": max(0.0, opt_config.target_response_relevancy - scores.get("response_relevancy", 0.0)),
    }

    primary_bottleneck = max(gaps, key=gaps.get)
    gap_val = gaps[primary_bottleneck]

    if gap_val <= _EPS:
        logger.info("No significant metrics gap remaining.")
        return current_config, ["no_change_needed"], state

    next_config = current_config
    applied_rules: list[str] = []
    severity = min(2.0, max(0.5, gap_val / 0.2))

    # --- BOTTLENECK: Retrieval F1 (precision/recall trade-off) ---
    if primary_bottleneck == "retrieval_f1":
        if recall < precision:
            k_step = max(1, int(round(opt_config.top_k_step * severity)))
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
        tier = state["prompt_tier"]

        if tier < 1:
            next_config = next_config.copy_with(prompt_template=COMBINED_PROMPT)
            state["prompt_tier"] = 1
            applied_rules.append(f"bottleneck_{primary_bottleneck}: apply_strict_combined_prompt")

        elif tier < 2 and not state["top_k_pruned"] and next_config.top_k > 2:
            new_top_k = next_config.top_k - 1
            next_config = next_config.copy_with(top_k=new_top_k)
            state["prompt_tier"] = 2
            state["top_k_pruned"] = True
            applied_rules.append(f"bottleneck_{primary_bottleneck}: prune_context_top_k_to_{new_top_k}")

        elif tier < 3 and mistral_settings is not None and document_summary is not None:
            refined = refine_prompt(
                current_prompt=next_config.prompt_template,
                document_summary=document_summary,
                failing_metric=primary_bottleneck,
                mistral_settings=mistral_settings,
            )
            if refined is not None:
                next_config = next_config.copy_with(prompt_template=refined)
                state["prompt_tier"] = 3
                applied_rules.append(
                    f"bottleneck_{primary_bottleneck}: llm_refined_prompt_from_document_summary"
                )
            else:
                applied_rules.append(
                    f"bottleneck_{primary_bottleneck}: llm_refinement_failed_keeping_current_prompt"
                )
        else:
            applied_rules.append(f"bottleneck_{primary_bottleneck}: all_generation_levers_exhausted")

    logger.info(
        "Primary Bottleneck: %s (gap: %.3f, retrieval_f1=%.3f, precision=%.3f, recall=%.3f) | Applied: %s | prompt_tier=%d",
        primary_bottleneck, gap_val, retrieval_f1, precision, recall, applied_rules, state["prompt_tier"],
    )
    return next_config, applied_rules, state