"""
experiment_runner.py
=====================
Orchestrates the full self-optimization loop:

    1. Load + chunk the source document, build the FAISS index (once,
       rebuilt per-iteration only when chunk_size/overlap change).
    2. Generate (or load cached) the fixed benchmark question set from
       the FULL document.
    2b. Generate (or load cached) a document summary, used by the
        optimizer's LLM-driven prompt-refinement fallback.
    3. For each iteration:
        a. Build a RAG pipeline with the current retrieval config.
        b. Answer every benchmark question.
        c. Score answers with Ragas.
        d. Persist config + scores.
        e. Track the best-performing configuration so far (with safety gates).
        f. Ask the optimizer to propose the next configuration.
        g. Stop early if targets are met or max_iterations is reached.
"""

from __future__ import annotations

import logging

from config import AppConfig
from document_loader import load_document, load_full_text
from chunking import split_documents
from embeddings import MistralEmbeddings
from vector_store import build_vector_store
from question_generator import get_or_create_benchmark
from document_summarizer import get_or_create_summary
from rag_pipeline import RagPipeline
from evaluator import run_evaluation
from optimizer import meets_targets, propose_next_config
import storage

logger = logging.getLogger("phoenix_rag.experiment_runner")


def run_experiment(app_config: AppConfig) -> dict:
    """Run the full optimization loop. Returns the best result found."""

    embeddings = MistralEmbeddings(app_config.mistral)

    # Benchmark is generated ONCE from the full document and never touched again.
    full_text = load_full_text(app_config.source_document)
    benchmark = get_or_create_benchmark(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        qg_config=app_config.question_generation,
        benchmark_path=app_config.benchmark_path,
    )
    question_texts = [q.question for q in benchmark]
    logger.info("Benchmark ready: %d fixed evaluation questions", len(benchmark))

    # Document summary, cached the same way -- used only as context for the
    # optimizer's LLM-driven prompt-refinement fallback (see optimizer.py,
    # prompt_refiner.py). Cheap: generated once per document, reused across
    # every iteration and every future run against this same document.
    document_summary = get_or_create_summary(
        full_text=full_text,
        mistral_settings=app_config.mistral,
        summary_path=app_config.summary_path,
    )
    logger.info("Document summary ready (%d chars)", len(document_summary))

    current_config = app_config.retrieval
    best_score = -1.0
    best_result: dict | None = None
    optimizer_state: dict = {"prompt_tier": 0, "top_k_pruned": False}

    cached_chunk_params: tuple[int, int] | None = None
    vector_store = None

    for iteration in range(1, app_config.optimizer.max_iterations + 1):
        logger.info("=== Iteration %d/%d ===", iteration, app_config.optimizer.max_iterations)

        chunk_params = (current_config.chunk_size, current_config.chunk_overlap)
        if vector_store is None or chunk_params != cached_chunk_params:
            logger.info("Rebuilding FAISS index (chunk params changed)")
            raw_docs = load_document(app_config.source_document)
            chunks = split_documents(
                raw_docs,
                chunk_size=current_config.chunk_size,
                chunk_overlap=current_config.chunk_overlap,
            )
            vector_store = build_vector_store(chunks, embeddings)
            cached_chunk_params = chunk_params

        pipeline = RagPipeline(vector_store, app_config.mistral, current_config)
        results = pipeline.answer_many(question_texts)

        scores = run_evaluation(results, benchmark, app_config.mistral)

        storage.save_iteration_config(iteration, current_config)
        storage.append_experiment_result(iteration, current_config, scores)

        # ------------------------------------------------------------------
        # Weighted Score + Minimum Faithfulness Gate
        # ------------------------------------------------------------------
        weighted_score = (
            scores.get("faithfulness", 0.0) * 0.40 +
            scores.get("context_recall", 0.0) * 0.20 +
            scores.get("context_precision", 0.0) * 0.20 +
            scores.get("response_relevancy", 0.0) * 0.20
        )

        is_safe = scores.get("faithfulness", 0.0) >= 0.80

        if is_safe and weighted_score > best_score:
            best_score = weighted_score
            best_result = {"iteration": iteration, "config": current_config, "scores": scores}
            storage.save_best_configuration(iteration, current_config, scores)
            logger.info("New best configuration saved! (Weighted Score: %.4f)", best_score)

        elif not is_safe and weighted_score > best_score:
            logger.warning(
                "Iteration %d scored highest (%.4f) but failed the Faithfulness safety gate (%.4f). Discarded.",
                iteration, weighted_score, scores.get("faithfulness", 0.0)
            )
        # ------------------------------------------------------------------

        if meets_targets(scores, app_config.optimizer):
            logger.info("Targets met at iteration %d, stopping early", iteration)
            break

        current_config, applied_rules, optimizer_state = propose_next_config(
            current_config,
            scores,
            app_config.optimizer,
            mistral_settings=app_config.mistral,
            document_summary=document_summary,
            state=optimizer_state,
        )

        storage.append_evaluation_scores(iteration, scores, applied_rules=applied_rules)

        if not applied_rules:
            logger.info("No further tuning rules triggered, stopping")
            break

    logger.info("Experiment complete. Best weighted score: %.4f", best_score)
    return best_result or {}