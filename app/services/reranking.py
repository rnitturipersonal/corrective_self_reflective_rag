from typing import Protocol
from sentence_transformers import CrossEncoder
from app.models import RetrievedChunk
from app.config import get_settings
from loguru import logger


class RerankingBackend(Protocol):
    """Protocol for reranking backends"""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank chunks and return top_k results"""
        ...


class LocalRerankingBackend:
    """Local cross-encoder reranking backend"""

    def __init__(self, settings):
        self.settings = settings
        self._model = None  # Lazy loading

    @property
    def model(self) -> CrossEncoder:
        """Lazy load cross-encoder model on first use"""
        if self._model is None:
            try:
                logger.info(f"Loading local reranker: {self.settings.reranker_model}")
                self._model = CrossEncoder(self.settings.reranker_model)
                logger.info("Local reranker loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local reranker: {e}")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank using local cross-encoder"""
        try:
            # Prepare (query, document) pairs for cross-encoder
            pairs = [(query, chunk.content) for chunk in chunks]

            # Get cross-encoder scores
            logger.info(f"Reranking {len(pairs)} chunks with local cross-encoder")
            scores = self.model.predict(pairs)

            # Update chunk scores with cross-encoder scores
            for i, chunk in enumerate(chunks):
                chunk.score = float(scores[i])

            # Sort by cross-encoder scores (descending)
            reranked = sorted(chunks, key=lambda x: x.score, reverse=True)

            # Return top_k chunks
            result = reranked[:top_k]
            logger.info(f"Local reranking complete: returning top {len(result)} chunks")
            return result

        except Exception as e:
            logger.error(f"Local reranking error: {e}, falling back")
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            return sorted_chunks[:top_k]


class VoyageRerankingBackend:
    """Voyage AI reranking backend"""

    def __init__(self, settings):
        self.settings = settings

        # Validate API key
        if not settings.voyage_api_key:
            raise ValueError(
                "Voyage API key required when reranker_backend='voyage'. "
                "Set VOYAGE_API_KEY in .env file."
            )

        # Initialize Voyage client
        try:
            import voyageai
            self.client = voyageai.Client(api_key=settings.voyage_api_key)
            logger.info("Voyage AI reranking backend initialized")
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage backend. "
                "Install with: uv sync"
            )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank using Voyage AI API"""
        try:
            # Prepare documents for Voyage AI
            documents = [chunk.content for chunk in chunks]

            # Call Voyage AI reranking API
            logger.info(f"Reranking {len(documents)} chunks with Voyage AI")
            reranking = self.client.rerank(
                query=query,
                documents=documents,
                model=self.settings.voyage_model,
                top_k=top_k
            )

            # Map results back to chunks
            index_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
            reranked_chunks = []

            for result in reranking.results:
                chunk = index_to_chunk[result.index]
                chunk.score = float(result.relevance_score)
                reranked_chunks.append(chunk)

            logger.info(f"Voyage reranking complete: {len(reranked_chunks)} chunks")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Voyage reranking error: {e}, falling back")
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            return sorted_chunks[:top_k]


class RerankingService:
    """
    Main reranking service with pluggable backends.

    Supports two backends:
    - 'local': Local cross-encoder model (sentence-transformers)
    - 'voyage': Voyage AI reranking API

    Backend selection controlled by config.reranker_backend
    """

    def __init__(self):
        self.settings = get_settings()

        # Select and initialize backend
        if self.settings.reranker_backend == "local":
            logger.info("Using local cross-encoder reranking backend")
            self.backend = LocalRerankingBackend(self.settings)
        elif self.settings.reranker_backend == "voyage":
            logger.info("Using Voyage AI reranking backend")
            self.backend = VoyageRerankingBackend(self.settings)
        else:
            raise ValueError(
                f"Invalid reranker_backend: '{self.settings.reranker_backend}'. "
                f"Must be 'local' or 'voyage'."
            )

    def rerank(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        top_k: int = None
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using the configured backend.

        Args:
            query: User's query
            retrieved_chunks: Initial chunks from vector search
            top_k: Number of top chunks to return (default from settings)

        Returns:
            Reranked chunks sorted by relevance scores
            Falls back to original chunks on error
        """
        if not retrieved_chunks:
            logger.info("Reranking: No chunks to rerank")
            return retrieved_chunks

        if top_k is None:
            top_k = self.settings.top_k_results

        # Delegate to backend
        return self.backend.rerank(query, retrieved_chunks, top_k)
