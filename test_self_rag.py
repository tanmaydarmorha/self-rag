import os
import sys
import tempfile
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.document_processor import process_document
from app.utils.embedding_utils import create_vector_store
from app.models.self_rag import SelfRAG

def test_self_rag_with_pdf(pdf_path, question):
    """Test the Self-RAG agent with a PDF document."""
    print(f"Processing document: {pdf_path}")
    print(f"Question: {question}")
    print("-" * 50)
    
    try:
        # Process the document
        processed_doc = process_document(pdf_path)
        
        # Create vector store from document chunks
        chunks = processed_doc["chunks"]
        metadatas = [{"chunk_id": i} for i in range(len(chunks))]
        
        print(f"Document processed successfully. Found {len(chunks)} chunks.")
        
        # Create vector store
        vector_store = create_vector_store(
            texts=chunks,
            metadatas=metadatas
        )
        
        print("Vector store created successfully.")
        
        # Initialize Self-RAG
        self_rag = SelfRAG(vector_store=vector_store)
        
        print("Self-RAG initialized. Processing question...")
        
        # Process question and get answer
        result = self_rag.process_document_and_answer(question)
        
        print("\nFinal Answer:")
        print(result.get("generation", "No answer generated"))
        
        print("\nRelevant Sources:")
        if "documents" in result:
            for i, doc in enumerate(result["documents"]):
                if doc.get("relevant", False):
                    print(f"\nSource {i+1}:")
                    print(doc["page_content"])
        
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_self_rag.py <pdf_path> [question]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if question is provided
    if len(sys.argv) >= 3:
        question = sys.argv[2]
    else:
        question = "What is the main topic of this document?"
    
    # Test Self-RAG
    test_self_rag_with_pdf(pdf_path, question)
