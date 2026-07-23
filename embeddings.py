"""
embeddings.py
=============
LangChain-compatible embeddings class backed by Mistral's `mistral-embed`
model, routed through the shared rate-limited MistralClient.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from config import MistralSettings
from mistralai.client import Mistral

logger = logging.getLogger("phoenix_rag.embeddings")

# Mistral's embedding endpoint accepts a max batch size; chunk larger
# input lists to stay under it.
_MAX_BATCH = 32


class MistralEmbeddings(Embeddings):
    """Adapter so FAISS / LangChain can use Mistral embeddings directly."""

    def __init__(self, settings: MistralSettings | None = None):
        self.settings = settings or MistralSettings()
        self._client = Mistral(api_key=self.settings.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), len(texts))
            response = self._client.embeddings.create(
                model=self.settings.embedding_model,
                inputs=batch,
            )
            # The API is expected to preserve input order, but sort by the
            # response's own index to be safe rather than assume it.
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            inputs=[text],
        )
        return response.data[0].embedding