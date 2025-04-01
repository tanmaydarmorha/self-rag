# Self RAG Implementation with FastAPI and Azure AI

This project implements a Self RAG (Retrieval-Augmented Generation) system based on LangGraph's reference implementation. The system processes documents, stores embeddings in Redis Stack, and answers user queries based on document content using Azure AI services.

## Tech Stack

- **FastAPI**: API layer
- **LangGraph**: Structured reasoning in RAG
- **Azure AI Services**:
  - Azure OpenAI Chat (gpt-4o) for response generation
  - Azure OpenAI Embedding (text-embedding-ada-002) for vector embeddings
  - Azure AI Document Intelligence for document processing
- **Redis Stack**: Vector store for embeddings

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file (see `.env.example`)
4. Start Redis Stack: `docker-compose up -d`
5. Run the application: `uvicorn app.main:app --reload`

## API Endpoints

- `POST /api/process`: Process a document and answer questions
  - Input: Document file upload and list of questions
  - Output: Answers to the questions based on document content

## Project Structure

```
self-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration and environment variables
│   ├── models/              # Pydantic models
│   │   └── __init__.py
│   ├── services/            # Service layer
│   │   ├── __init__.py
│   │   ├── document.py      # Document processing with Azure AI
│   │   ├── embedding.py     # Embedding service with Azure OpenAI
│   │   ├── vector_store.py  # Redis Stack vector store
│   │   └── rag_pipeline.py  # LangGraph Self RAG implementation
│   └── api/                 # API routes
│       ├── __init__.py
│       └── endpoints.py
├── .env.example             # Example environment variables
├── .env                     # Environment variables (gitignored)
├── requirements.txt         # Dependencies
└── docker-compose.yml       # Redis Stack configuration
```
