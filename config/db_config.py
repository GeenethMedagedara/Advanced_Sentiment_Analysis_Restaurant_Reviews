"""
Handles the connection to the MongoDB database.
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")  # Change if hosted elsewhere
DATABASE_NAME = "predictions_db"

def get_db():
    """Establish a connection to MongoDB and return the database instance."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    return db
