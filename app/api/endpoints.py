"""API endpoints for the Self RAG application."""
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import parse_obj_as

from app.models.schemas import ProcessDocumentRequest, ProcessDocumentResponse, QueryResult
from app.services.document import document_service
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service
from app.services.rag_pipeline import self_rag_pipeline


router = APIRouter()


@router.post("/process", response_model=ProcessDocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    questions_json: str = Form(...),
):
    """
    Process a document and answer questions based on its content.
    
    Args:
        file: The document file to process
        questions_json: JSON string containing a list of questions
        
    Returns:
        Answers to the questions based on document content
    """
    try:
        # Parse questions from JSON
        request = ProcessDocumentRequest.model_validate_json(questions_json)
        
        # Process document
        document_chunks = await document_service.process_document(file)
        if not document_chunks:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Get document ID
        document_id = document_chunks[0].metadata.get("document_id")
        
        # Generate embeddings for document chunks
        chunk_texts = [chunk.content for chunk in document_chunks]
        embeddings = await embedding_service.get_embeddings(chunk_texts)
        
        # Store document chunks and embeddings
        await vector_store_service.store_document_chunks(document_chunks, embeddings)
        
        # Process each question
        results = []
        for question in request.questions:
            # Process question using Self RAG pipeline
            query_result = await self_rag_pipeline.process_question(question)
            results.append(query_result)
        
        # Create response
        response = ProcessDocumentResponse(
            document_id=document_id,
            results=results
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
