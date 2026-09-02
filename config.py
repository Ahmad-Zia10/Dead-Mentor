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

# CORS: comma-separated list of allowed frontend origins.
# Defaults cover the local Vite dev server and the Streamlit UI.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501",
    ).split(",")
    if origin.strip()
]

# Retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
# Relevance below this is treated as "not in the source texts" rather than being
# passed to the model as though it were relevant. Gemini embeddings compress
# these scores into a narrow band, so this sits higher than it might look:
# measured on Meditations, off-topic questions top out at ~0.81 and on-topic
# ones start at ~0.83. Re-check with scripts/check_retrieval.py if the
# embedding model or the corpus changes.
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.82"))

# Models 
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
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