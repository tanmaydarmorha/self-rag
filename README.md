# Self-RAG Document Analysis System

This project implements a Self-RAG (Retrieval Augmented Generation) system for document analysis using Gemma 3 from Ollama and nomic-embed-text for embeddings.

## Features

- PDF, DOCX, and TXT document processing
- Document chunking and embedding using nomic-embed-text
- Self-RAG implementation using LangGraph
- Document relevance grading
- Hallucination detection
- Answer quality assessment
- FastAPI backend with document upload and question answering endpoints

## Setup

### Prerequisites

- Python 3.8+
- Ollama installed locally (for Gemma 3 model)
- MongoDB (for document metadata storage)
- Redis (for vector storage)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `backend/.env`:
   ```
   # Database Configuration
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB_NAME=selfrag_db

   # Redis Configuration
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=
   ```

4. Pull the Gemma 3 model using Ollama:
   ```
   ollama pull gemma3:latest
   ```

## Usage

### Running the Backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Testing the Self-RAG Implementation

You can test the Self-RAG implementation directly using the test script:

```bash
python backend/test_self_rag.py path/to/your/document.pdf "Your question about the document"
```

### API Endpoints

- `POST /api/documents/upload`: Upload a document for processing
- `POST /api/documents/ask`: Ask a question about a document

## Self-RAG Flow

The Self-RAG implementation follows this flow:

1. **Retrieve**: Get relevant documents from the vector store
2. **Grade Documents**: Assess the relevance of retrieved documents
3. **Generate/Transform Query**: Generate an answer or transform the query if no relevant documents are found
4. **Check Answer Quality**: Ensure the answer is grounded in the documents and addresses the question

## Project Structure

```
selfRag/
|── app/
│   |── api/
│   │   └── document.py
│   |── database/
│   │   |── mongodb.py
│   │   └── redis_db.py
│   |── models/
│   │   └── self_rag.py
│   |── utils/
│   │   |── document_processor.py
│   │   └── embedding_utils.py
│   └── main.py
└── test_self_rag.py
└── requirements.txt
```
