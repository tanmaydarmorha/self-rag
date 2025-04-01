"""Document processing service using Azure AI Document Intelligence."""
import os
import uuid
import tempfile
from typing import List, Optional
from fastapi import UploadFile
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models.schemas import DocumentChunk


class DocumentService:
    """Service for processing documents using Azure AI Document Intelligence."""

    def __init__(self):
        """Initialize the document service with Azure Document Intelligence client."""
        self.client = DocumentIntelligenceClient(
            endpoint=settings.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(settings.azure_document_intelligence_key),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def process_document(self, file: UploadFile) -> List[DocumentChunk]:
        """
        Process a document using Azure AI Document Intelligence and split into chunks.
        
        Args:
            file: The uploaded file to process
            
        Returns:
            List of document chunks with content and metadata
        """
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Process the document with Azure Document Intelligence
            with open(temp_path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    "prebuilt-layout", 
                    analyze_request=AnalyzeDocumentRequest(
                        base64_source=f.read()
                    )
                )
            result = poller.result()
            
            # Extract text content from the document
            document_text = ""
            for page in result.pages:
                for line in page.lines:
                    document_text += line.content + "\n"
            
            # Create document ID
            document_id = str(uuid.uuid4())
            
            # Split the document into chunks
            texts = self.text_splitter.split_text(document_text)
            
            # Create document chunks with metadata
            chunks = []
            for i, text in enumerate(texts):
                chunk = DocumentChunk(
                    content=text,
                    metadata={
                        "document_id": document_id,
                        "chunk_id": i,
                        "source": file.filename,
                        "chunk_index": i,
                        "total_chunks": len(texts),
                    }
                )
                chunks.append(chunk)
                
            return chunks
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Initialize document service
document_service = DocumentService()
