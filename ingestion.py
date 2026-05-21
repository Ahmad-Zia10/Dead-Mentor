import os
import time
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, MENTORS, GEMINI_EMBEDDING_MODEL


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def load_documents(data_path: str, mentor_name: str) -> list:
    documents = []

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            print(f"  Skipping unsupported file: {filename}")
            continue

        docs = loader.load()

        # Tag every chunk with mentor and source
        for doc in docs:
            doc.metadata["mentor"] = mentor_name
            doc.metadata["source_file"] = filename

        documents.extend(docs)
        print(f"  Loaded: {filename} ({len(docs)} pages/sections)")

    return documents


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks


def store_in_pinecone(chunks: list, mentor_name: str, start_batch: int = 0):
    embeddings = get_embeddings()
    batch_size = 50
    total_batches = -(-len(chunks) // batch_size)

    for i in range(start_batch * batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=mentor_name
        )

        print(f"  Batch {batch_num}/{total_batches} upserted")

        if i + batch_size < len(chunks):
            print(f"  Waiting 65 seconds for rate limit...")
            time.sleep(65)

    print(f"  Stored chunks for {mentor_name} ✅")


def ingest_mentor(mentor_name: str, start_batch: int = 0):
    data_path = f"./data/raw/{mentor_name}"

    if not os.path.exists(data_path):
        print(f"  No data folder found for {mentor_name}, skipping.")
        return

    print(f"\nIngesting {mentor_name}...")
    documents = load_documents(data_path, mentor_name)

    if not documents:
        print(f"  No documents found for {mentor_name}")
        return

    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks, resuming from batch {start_batch + 1}")

    store_in_pinecone(chunks, mentor_name, start_batch=start_batch)
    print(f"  Done: {mentor_name} ✅")


if __name__ == "__main__":
    print("Starting ingestion pipeline...")
    
    # ingest_mentor("marcus", start_batch=23)   # fully done, skip
    # ingest_mentor("feynman", start_batch=21)  # fully done, skip
    ingest_mentor("darwin", start_batch=36)    # resume from batch 37
    
    print("\nAll mentors ingested successfully. ✅")