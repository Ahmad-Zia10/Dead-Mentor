import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VECTORSTORE_DIR = "./vectorstore"
PERSONAS_DIR = "./personas"
LOG_DIR = "./logs"

MENTORS = ["marcus", "feynman", "darwin"]

# Validate on startup
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")