"""Embedding service using Azure OpenAI."""
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from app.config import settings


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""

    def __init__(self):
        """Initialize the embedding service with Azure OpenAI."""
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_openai_embedding_deployment,
            openai_api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        return await self.embeddings.aembed_documents(texts)

    async def get_query_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to generate embedding for
            
        Returns:
            Embedding vector
        """
        return await self.embeddings.aembed_query(text)


# Initialize embedding service
embedding_service = EmbeddingService()
