import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection details from environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "selfrag_db")

# Create MongoDB client
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Collections
documents_collection = db["documents"]
users_collection = db["users"]
analysis_collection = db["analysis"]

def get_document_by_id(document_id):
    """Retrieve a document by its ID."""
    return documents_collection.find_one({"_id": document_id})

def save_document_metadata(metadata):
    """Save document metadata to MongoDB."""
    return documents_collection.insert_one(metadata).inserted_id

def save_analysis_result(analysis_data):
    """Save analysis result to MongoDB."""
    return analysis_collection.insert_one(analysis_data).inserted_id

def get_user_documents(user_id):
    """Get all documents for a specific user."""
    return list(documents_collection.find({"user_id": user_id}))
