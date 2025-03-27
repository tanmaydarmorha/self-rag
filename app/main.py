import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .api.document import router as document_router

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Self-RAG Document Analysis API",
    description="API for document analysis using Self-RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Self-RAG Document Analysis API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
