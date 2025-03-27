import os
import uuid
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from ..utils.document_processor import process_document
from ..utils.embedding_utils import create_vector_store, get_vector_store
from ..database.mongodb import save_document_metadata, get_document_by_id
from ..models.self_rag import SelfRAG

router = APIRouter()

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    document_id: str
    sources: List[dict]

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """Upload a document for processing."""
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Check file extension
    allowed_extensions = [".pdf", ".docx", ".txt"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Process document
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "user_id": user_id
        }
        
        processed_doc = process_document(temp_file_path, metadata)
        
        # Create vector store from document chunks
        chunks = processed_doc["chunks"]
        metadatas = [{"document_id": document_id, "chunk_id": i} for i in range(len(chunks))]
        
        vector_store = create_vector_store(
            texts=chunks,
            metadatas=metadatas
        )
        
        # Save document metadata to MongoDB
        save_document_metadata(processed_doc["metadata"])
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            message="Document processed successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about a document."""
    try:
        # Get document from MongoDB
        document = get_document_by_id(request.document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {request.document_id} not found"
            )
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Initialize Self-RAG
        self_rag = SelfRAG(vector_store=vector_store)
        
        # Process question and get answer
        result = self_rag.process_document_and_answer(request.question)
        
        # Extract relevant sources
        sources = []
        if "documents" in result:
            for doc in result["documents"]:
                if doc.get("relevant", False):
                    sources.append({
                        "content": doc["page_content"],
                        "metadata": doc.get("metadata", {})
                    })
        
        return AnswerResponse(
            question=request.question,
            answer=result.get("generation", "No answer generated"),
            document_id=request.document_id,
            sources=sources
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
