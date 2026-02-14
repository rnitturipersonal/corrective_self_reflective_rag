from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from app.config import get_settings
from loguru import logger
from uuid import uuid4


class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key
        )
        self.collection_name = self.settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimensions,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection creation error: {e}")
            raise
    
    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict]
    ) -> list[str]:
        """Insert chunks with embeddings and metadata"""
        points = []
        chunk_ids = []
        
        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas):
            chunk_id = str(uuid4())
            chunk_ids.append(chunk_id)
            
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "content": chunk,
                    **metadata
                }
            ))
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} chunks")
            return chunk_ids
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            raise
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_conditions: dict | None = None
    ) -> list[dict]:
        """Search for similar chunks"""
        try:
            search_filter = None
            if filter_conditions:
                # Build Qdrant filter from conditions if needed
                pass
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            ).points
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("content"),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def delete_by_source(self, source_file: str):
        """Delete all chunks from a source file"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file)
                        )
                    ]
                )
            )
            logger.info(f"Deleted chunks from: {source_file}")
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise
