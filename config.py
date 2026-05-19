import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Paths
PERSONAS_DIR = "./personas"
LOG_DIR = "./logs"

# Mentors
MENTORS = ["marcus", "feynman", "darwin"]

# Models 
GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Validate all keys on startup
missing = []
if not GROQ_API_KEY:
    missing.append("GROQ_API_KEY")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")
if not PINECONE_API_KEY:
    missing.append("PINECONE_API_KEY")
if not PINECONE_INDEX_NAME:
    missing.append("PINECONE_INDEX_NAME")

if missing:
    raise ValueError(f"Missing keys in .env: {', '.join(missing)}")