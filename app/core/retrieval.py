from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.hyde import HydeService
from app.models import RetrievedChunk, ChunkMetadata
from app.config import get_settings
from loguru import logger


class RetrievalService:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.hyde_service = HydeService()
        self._last_hyde_hypotheses = None  # For metadata tracking
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_hyde: bool = False
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for query with optional HYDE enhancement.

        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            use_hyde: Whether to use HYDE for query expansion

        Returns:
            List of retrieved chunks
        """

        if top_k is None:
            top_k = self.settings.top_k_results

        if use_hyde:
            # Generate hypothetical documents
            hypotheses = self.hyde_service.generate_hypothetical_documents(query)
            self._last_hyde_hypotheses = hypotheses

            # Batch embed all hypotheses
            hypothesis_vectors = self.embedding_service.embed_batch(hypotheses)

            # Run parallel searches for each hypothesis
            all_results = []
            for vector in hypothesis_vectors:
                results = self.vector_store.search(
                    query_vector=vector,
                    top_k=top_k
                )
                all_results.extend(results)

            # Merge and deduplicate results
            retrieved_chunks = self._merge_and_deduplicate(all_results, top_k)
            logger.info(
                f"HYDE: Retrieved {len(retrieved_chunks)} unique chunks "
                f"from {len(hypotheses)} hypotheses"
            )
        else:
            # Standard retrieval pathway
            query_vector = self.embedding_service.embed_text(query)

            # Search vector store
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k
            )

            # Convert to RetrievedChunk models
            retrieved_chunks = self._convert_to_chunks(results)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")

        return retrieved_chunks

    def _convert_to_chunks(self, results: list[dict]) -> list[RetrievedChunk]:
        """Convert vector store results to RetrievedChunk models"""
        chunks = []
        for result in results:
            chunk = RetrievedChunk(
                content=result["content"],
                metadata=ChunkMetadata(**result["metadata"]),
                score=result["score"]
            )
            chunks.append(chunk)
        return chunks

    def _merge_and_deduplicate(
        self,
        all_results: list[dict],
        top_k: int
    ) -> list[RetrievedChunk]:
        """
        Merge results from multiple searches and deduplicate by chunk ID.
        Preserves the highest score for each unique chunk.

        Args:
            all_results: Combined results from multiple searches
            top_k: Number of top chunks to return

        Returns:
            Deduplicated and sorted list of chunks
        """
        # Track best score per unique chunk_id
        chunk_map = {}

        for result in all_results:
            chunk_id = result["metadata"]["chunk_id"]
            score = result["score"]

            if chunk_id not in chunk_map or score > chunk_map[chunk_id]["score"]:
                chunk_map[chunk_id] = result

        # Convert to RetrievedChunk models
        unique_chunks = self._convert_to_chunks(list(chunk_map.values()))

        # Sort by score descending
        sorted_chunks = sorted(
            unique_chunks,
            key=lambda x: x.score,
            reverse=True
        )

        # Return top_k
        return sorted_chunks[:top_k]

    def get_last_hyde_hypotheses(self) -> list[str] | None:
        """Return hypotheses from last HYDE retrieval for response metadata"""
        return self._last_hyde_hypotheses
