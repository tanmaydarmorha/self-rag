import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import pypdf
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
    return text

def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    elif extension == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def process_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process a document and return its text and metadata."""
    if metadata is None:
        metadata = {}
    
    # Extract text from document
    text = extract_text(file_path)
    
    # Split text into chunks
    chunks = split_text(text)
    
    # Generate document ID if not provided
    if "document_id" not in metadata:
        metadata["document_id"] = str(uuid.uuid4())
    
    # Add additional metadata
    metadata.update({
        "filename": os.path.basename(file_path),
        "processed_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "total_characters": len(text)
    })
    
    return {
        "text": text,
        "chunks": chunks,
        "metadata": metadata
    }
