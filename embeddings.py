"""Single source of truth for the embedding model.

Ingestion and query-time retrieval must use identical embeddings; if they ever
drift, retrieval silently degrades rather than failing loudly. The client is
cached because building it per request adds latency to every message.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
