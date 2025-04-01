"""Pydantic models for the Self RAG application."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Model for a question request."""
    question: str = Field(..., description="The question to be answered")


class ProcessDocumentRequest(BaseModel):
    """Model for processing a document and answering questions."""
    questions: List[str] = Field(..., description="List of questions to be answered based on the document")


class DocumentChunk(BaseModel):
    """Model for a document chunk."""
    content: str = Field(..., description="The content of the document chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the document chunk")


class RetrievedDocument(BaseModel):
    """Model for a retrieved document with relevance score."""
    content: str = Field(..., description="The content of the retrieved document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the retrieved document")
    score: float = Field(..., description="Relevance score for the document")


class CritiqueResult(BaseModel):
    """Model for the critique result in Self RAG."""
    is_relevant: bool = Field(..., description="Whether the retrieved documents are relevant")
    reasoning: str = Field(..., description="Reasoning for the relevance assessment")


class QueryResult(BaseModel):
    """Model for the query result in Self RAG."""
    answer: str = Field(..., description="The answer to the question")
    retrieved_documents: List[RetrievedDocument] = Field(
        default_factory=list, 
        description="The documents retrieved for answering the question"
    )
    critique: Optional[CritiqueResult] = Field(
        None, 
        description="Critique of the retrieved documents and answer"
    )


class ProcessDocumentResponse(BaseModel):
    """Model for the response to processing a document and answering questions."""
    document_id: str = Field(..., description="The ID of the processed document")
    results: List[QueryResult] = Field(..., description="Results for each question")
