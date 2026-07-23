"""
rag_pipeline.py
================
The actual RAG pipeline being optimized: retrieve -> build prompt -> generate.

Kept intentionally simple/stateless so it can be re-run cheaply for every
question in the benchmark, for every configuration the optimizer tries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import MistralSettings, RetrievalConfig
from mistralai.client import Mistral
from vector_store import get_retriever

logger = logging.getLogger("phoenix_rag.rag_pipeline")


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]


class RagPipeline:
    """Retrieve -> build prompt -> generate, parameterized by RetrievalConfig."""

    def __init__(
        self,
        vector_store: FAISS,
        mistral_settings: MistralSettings,
        retrieval_config: RetrievalConfig,
    ):
        self.vector_store = vector_store
        self.retrieval_config = retrieval_config
        self._client = Mistral(api_key=mistral_settings.api_key)
        self._model = mistral_settings.generation_model
        self._retriever = get_retriever(
            vector_store,
            top_k=retrieval_config.top_k,
            retriever_type=retrieval_config.retriever_type,
            similarity_threshold=retrieval_config.similarity_threshold,
        )

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return self.retrieval_config.prompt_template.format(
            context=joined_context, question=question
        )

    def answer(self, question: str) -> RagResult:
        docs = self.retrieve(question)
        contexts = [d.page_content for d in docs]
        prompt = self.build_prompt(question, contexts)

        response = self._client.chat.complete(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content
        return RagResult(question=question, answer=answer_text, contexts=contexts)

    def answer_many(self, questions: list[str]) -> list[RagResult]:
        results = []
        for i, q in enumerate(questions, start=1):
            logger.info("Answering question %d/%d", i, len(questions))
            results.append(self.answer(q))
        return results