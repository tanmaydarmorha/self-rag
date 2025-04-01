"""Test script for the Self RAG API."""
import json
import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_URL = "http://localhost:8000/api/process"

async def test_self_rag_api():
    """Test the Self RAG API with a sample document and questions."""
    # Sample questions
    questions = [
        "What are the key components of a Self RAG system?",
        "How does the critique mechanism work in Self RAG?",
        "What advantages does Self RAG have over traditional RAG?"
    ]
    
    # Create form data
    form_data = {
        "questions_json": json.dumps({"questions": questions})
    }
    
    # Sample document path (replace with your own document)
    document_path = "sample_document.pdf"  # Replace with your document path
    
    if not os.path.exists(document_path):
        print(f"Error: Document not found at {document_path}")
        return
    
    # Send request to API
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(document_path, "rb") as f:
            files = {"file": (os.path.basename(document_path), f, "application/pdf")}
            response = await client.post(API_URL, data=form_data, files=files)
    
    # Print response
    if response.status_code == 200:
        result = response.json()
        print(f"Document ID: {result['document_id']}")
        print("\nResults:")
        for i, query_result in enumerate(result['results']):
            print(f"\nQuestion {i+1}: {questions[i]}")
            print(f"Answer: {query_result['answer']}")
            print("\nRetrieved Documents:")
            for j, doc in enumerate(query_result['retrieved_documents']):
                print(f"Document {j+1} (Score: {doc['score']:.4f}):")
                print(f"{doc['content'][:150]}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    asyncio.run(test_self_rag_api())
