"""Retrieval over the per-mentor Pinecone namespaces."""

from functools import lru_cache

from langchain_pinecone import PineconeVectorStore
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RETRIEVAL_K,
    RETRIEVAL_SCORE_THRESHOLD,
)
from embeddings import get_embeddings


@lru_cache(maxsize=None)
def get_vectorstore(mentor_name: str) -> PineconeVectorStore:
    """Cached per mentor; rebuilding this per request adds latency to every message."""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=mentor_name,
        pinecone_api_key=PINECONE_API_KEY,
    )


def retrieve_context(mentor_name: str, query: str) -> list:
    """Return chunks relevant to the query, or [] if none clear the threshold.

    Returning fewer than RETRIEVAL_K documents is deliberate. Always handing the
    model k chunks means an off-topic question still arrives with source material
    attached, which is how a grounded system ends up confidently inventing an
    answer. An empty list lets the caller say "not in my writings" instead.
    """
    store = get_vectorstore(mentor_name)
    scored = store.similarity_search_with_relevance_scores(query, k=RETRIEVAL_K)

    return [doc for doc, score in scored if score >= RETRIEVAL_SCORE_THRESHOLD]


def retrieve_with_scores(mentor_name: str, query: str) -> list[tuple]:
    """Same retrieval, but keeps the scores. Useful for tuning the threshold."""
    store = get_vectorstore(mentor_name)
    return store.similarity_search_with_relevance_scores(query, k=RETRIEVAL_K)
