import os
import sys
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.document_processor import process_document, split_text, extract_text
from app.utils.embedding_utils import create_vector_store
from app.models.self_rag import SelfRAG

def query_event_hub_docs(pdf_path, question):
    """Process the Event Hub documentation and answer a specific question."""
    print(f"Processing document: {pdf_path}")
    print(f"Question: {question}")
    print("-" * 50)
    
    try:
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = extract_text(pdf_path)
        
        # Split text into smaller chunks for better processing
        print("Splitting text into chunks...")
        chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        
        print(f"Document processed successfully. Found {len(chunks)} chunks.")
        
        # Create vector store from document chunks
        metadatas = [{"chunk_id": i, "source": os.path.basename(pdf_path)} for i in range(len(chunks))]
        
        print("Creating vector store...")
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
            relevant_count = 0
            for i, doc in enumerate(result["documents"]):
                if doc.get("relevant", False):
                    relevant_count += 1
                    print(f"\nSource {relevant_count}:")
                    # Print a shorter snippet to avoid overwhelming output
                    content = doc["page_content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(content)
            
            if relevant_count == 0:
                print("No relevant sources found.")
        
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Set the PDF path and question
    pdf_path = "resources/event-hub.pdf"
    question = "how can read messages from event hub?"
    
    # Run the query
    query_event_hub_docs(pdf_path, question)
