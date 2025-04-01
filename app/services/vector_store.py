"""Vector store service using Redis Stack."""
import json
from typing import List, Dict, Any, Optional
import redis
from app.config import settings
from app.models.schemas import DocumentChunk, RetrievedDocument


class VectorStoreService:
    """Service for storing and retrieving document embeddings using Redis Stack."""

    def __init__(self):
        """Initialize the vector store service with Redis Stack."""
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=False,  # Keep binary for vector data
        )
        self._ensure_index()

    def _ensure_index(self):
        """Ensure the vector index exists in Redis."""
        try:
            # Check if index exists
            indices = self.redis_client.execute_command("FT._LIST")
            if b"document_index" not in indices:
                # Create index if it doesn't exist
                self.redis_client.execute_command(
                    "FT.CREATE", "document_index", "ON", "HASH", "PREFIX", "1", "doc:",
                    "SCHEMA", 
                    "content", "TEXT", "WEIGHT", "1.0",
                    "metadata", "TEXT",
                    "embedding", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE"
                )
        except redis.exceptions.ResponseError as e:
            # Index might already exist
            if "Index already exists" not in str(e):
                raise

    async def store_document_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> str:
        """
        Store document chunks and their embeddings in Redis.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors for each chunk
            
        Returns:
            Document ID
        """
        if not chunks:
            return None
            
        document_id = chunks[0].metadata.get("document_id")
        
        # Store each chunk with its embedding
        pipeline = self.redis_client.pipeline()
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"doc:{document_id}:{chunk.metadata.get('chunk_id')}"
            
            # Convert embedding to bytes
            embedding_bytes = self._float_list_to_bytes(embedding)
            
            # Store document chunk
            pipeline.hset(
                chunk_id,
                mapping={
                    "content": chunk.content,
                    "metadata": json.dumps(chunk.metadata),
                    "embedding": embedding_bytes
                }
            )
        
        # Execute pipeline
        pipeline.execute()
        
        return document_id

    async def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedDocument]:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with relevance scores
        """
        # Convert embedding to bytes
        embedding_bytes = self._float_list_to_bytes(query_embedding)
        
        # Search for similar vectors
        query = f"*=>[KNN {top_k} @embedding $embedding AS score]"
        result = self.redis_client.execute_command(
            "FT.SEARCH", "document_index", query,
            "PARAMS", "2", "embedding", embedding_bytes,
            "RETURN", "3", "content", "metadata", "score",
            "SORTBY", "score", "ASC"
        )
        
        # Parse results
        retrieved_documents = []
        if result and isinstance(result, list) and len(result) > 1:
            # Skip the first element (count)
            for i in range(1, len(result), 2):
                doc_id = result[i].decode('utf-8')
                doc_data = result[i + 1]
                
                # Extract fields
                content = None
                metadata = {}
                score = 0.0
                
                for j in range(0, len(doc_data), 2):
                    field_name = doc_data[j].decode('utf-8')
                    field_value = doc_data[j + 1]
                    
                    if field_name == "content":
                        content = field_value.decode('utf-8')
                    elif field_name == "metadata":
                        metadata = json.loads(field_value.decode('utf-8'))
                    elif field_name == "score":
                        score = float(field_value)
                
                # Create retrieved document
                if content:
                    retrieved_document = RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        score=1.0 - score  # Convert distance to similarity score
                    )
                    retrieved_documents.append(retrieved_document)
        
        return retrieved_documents

    def _float_list_to_bytes(self, float_list: List[float]) -> bytes:
        """Convert a list of floats to bytes for Redis storage."""
        import struct
        return struct.pack(f'{len(float_list)}f', *float_list)


# Initialize vector store service
vector_store_service = VectorStoreService()
