import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Use text-embedding-3-small from OpenAI
def get_embeddings_model():
    """Get the embeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(texts: List[str], metadatas: List[Dict[str, Any]] = None):
    """Create a vector store from texts and metadata."""
    embeddings = get_embeddings_model()
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"
    )
    
    return vector_store

def get_vector_store(persist_directory: str = "./chroma_db"):
    """Get an existing vector store."""
    embeddings = get_embeddings_model()
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_store

def add_texts_to_vector_store(vector_store, texts: List[str], metadatas: List[Dict[str, Any]] = None):
    """Add texts to an existing vector store."""
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    vector_store.persist()
    
    return vector_store

def similarity_search(vector_store, query: str, k: int = 4):
    """Perform similarity search on the vector store."""
    return vector_store.similarity_search(query, k=k)
