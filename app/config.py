"""Configuration module for the Self RAG application."""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    azure_openai_api_version: str = Field("2023-12-01-preview", description="Azure OpenAI API version")
    azure_openai_chat_deployment: str = Field("gpt-4o", description="Azure OpenAI chat model deployment name")
    azure_openai_embedding_deployment: str = Field("text-embedding-ada-002", description="Azure OpenAI embedding model deployment name")
    
    # Azure Document Intelligence Configuration
    azure_document_intelligence_key: str = Field(..., description="Azure Document Intelligence key")
    azure_document_intelligence_endpoint: str = Field(..., description="Azure Document Intelligence endpoint URL")
    
    # Redis Configuration
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: str = Field(..., description="Redis password")
    redis_ssl: bool = Field(False, description="Use SSL for Redis connection")
    
    # Application Configuration
    debug: bool = Field(False, description="Debug mode")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


# Initialize settings
settings = Settings()
