from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, PINECONE_INDEX_NAME, GEMINI_EMBEDDING_MODEL, PINECONE_API_KEY
import os

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )

def get_retriever(mentor_name: str):
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=mentor_name
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

def retrieve_context(mentor_name: str, query: str) -> list:
    retriever = get_retriever(mentor_name)
    docs = retriever.invoke(query)
    return docs

if __name__ == "__main__":
    docs = retrieve_context("marcus", "How do I deal with failure?")
    for doc in docs:
        print("---")
        print(doc.page_content)
        print("Source:", doc.metadata)