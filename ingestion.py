import argparse
import hashlib
import os
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MENTORS,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from embeddings import get_embeddings
from preprocessing import citation_at, clean_text, find_sections, work_title
from ratelimit import DEFAULT_MAX_ATTEMPTS, DailyQuotaExhausted, call_with_backoff

# The Gemini free tier caps embedding calls per minute as well as per day.
# Smaller batches with a short pause keep a long ingest under the per-minute
# ceiling; call_with_backoff covers whatever still slips through.
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "25"))
BATCH_PAUSE_SECONDS = float(os.getenv("INGEST_BATCH_PAUSE", "20"))


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def load_documents(data_path: str, mentor_name: str) -> list:
    """Read each source file, strip boilerplate, and tag it with its citation.

    One Document per source file; splitting happens in chunk_documents so that
    section offsets can be resolved against the cleaned full text.
    """
    documents = []

    for filename in sorted(os.listdir(data_path)):
        filepath = os.path.join(data_path, filename)

        if not filename.endswith(".txt"):
            print(f"  Skipping unsupported file: {filename}")
            continue

        with open(filepath, encoding="utf-8") as f:
            raw = f.read()

        text = clean_text(raw, filename)
        removed = len(raw) - len(text)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "mentor": mentor_name,
                    "source_file": filename,
                    "work": work_title(filename),
                },
            )
        )
        print(
            f"  Loaded: {filename} "
            f"({len(text):,} chars, stripped {removed:,} of boilerplate)"
        )

    return documents


def chunk_documents(documents: list) -> list:
    """Split into chunks, tagging each with the section it came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []
    for doc in documents:
        filename = doc.metadata["source_file"]
        sections = find_sections(doc.page_content, filename)
        title = doc.metadata["work"]

        for chunk in splitter.split_documents([doc]):
            offset = chunk.metadata.get("start_index", 0)
            chunk.metadata["citation"] = citation_at(sections, offset, title)
            chunks.append(chunk)

        print(f"    {filename}: {len(sections)} sections detected")

    return chunks


def chunk_id(chunk) -> str:
    """Deterministic ID so re-ingesting overwrites instead of duplicating."""
    key = f"{chunk.metadata['source_file']}:{chunk.metadata.get('start_index', 0)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def store_in_pinecone(
    chunks: list,
    mentor_name: str,
    start_chunk: int = 0,
    batch_size: int = BATCH_SIZE,
    pause: float = BATCH_PAUSE_SECONDS,
):
    embeddings = get_embeddings()
    total_batches = -(-len(chunks) // batch_size)

    def note_retry(attempt, delay, exc):
        print(f"    rate limited, waiting {delay:.0f}s "
              f"(attempt {attempt}/{DEFAULT_MAX_ATTEMPTS - 1})")

    for i in range(start_chunk, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            call_with_backoff(
                PineconeVectorStore.from_documents,
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace=mentor_name,
                ids=[chunk_id(c) for c in batch],
                on_retry=note_retry,
            )
        except DailyQuotaExhausted as e:
            # Deterministic IDs make the completed batches safe to keep, so
            # report exactly where to resume instead of losing the run.
            print(f"\n  Daily embedding quota exhausted at batch {batch_num}.")
            print(f"  Completed batches are stored. Resume tomorrow with:")
            print(f"    python ingestion.py {mentor_name} --skip-chunks {i}")
            raise SystemExit(3) from e

        print(f"  Batch {batch_num}/{total_batches} upserted")

        # Pace the next batch to stay under the per-minute ceiling. The retry
        # above is the safety net; this is what keeps it from firing.
        if i + batch_size < len(chunks):
            time.sleep(pause)

    print(f"  Stored chunks for {mentor_name}")


def clear_namespace(mentor_name: str):
    """Delete every vector for a mentor. Used before a clean re-ingest."""
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    if mentor_name in stats.get("namespaces", {}):
        index.delete(delete_all=True, namespace=mentor_name)
        print(f"  Cleared existing vectors in namespace '{mentor_name}'")
    else:
        print(f"  Namespace '{mentor_name}' is already empty")


def ingest_mentor(mentor_name: str, start_chunk: int = 0, clear: bool = False):
    data_path = f"./data/raw/{mentor_name}"

    if not os.path.exists(data_path):
        print(f"  No data folder found for {mentor_name}, skipping.")
        return

    print(f"\nIngesting {mentor_name}...")

    if clear:
        clear_namespace(mentor_name)

    documents = load_documents(data_path, mentor_name)

    if not documents:
        print(f"  No documents found for {mentor_name}")
        return

    chunks = chunk_documents(documents)
    if start_chunk:
        print(f"  Created {len(chunks)} chunks, skipping the first {start_chunk}")
    else:
        print(f"  Created {len(chunks)} chunks")

    store_in_pinecone(chunks, mentor_name, start_chunk=start_chunk)
    print(f"  Done: {mentor_name}")


def main():
    parser = argparse.ArgumentParser(description="Ingest mentor source texts into Pinecone.")
    parser.add_argument(
        "mentors",
        nargs="*",
        default=MENTORS,
        help=f"Mentors to ingest (default: all of {', '.join(MENTORS)})",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Resume from this batch number (0-indexed), for rate-limit restarts. "
             "Batch numbers depend on the batch size -- prefer --skip-chunks, "
             "which does not.",
    )
    parser.add_argument(
        "--skip-chunks",
        type=int,
        default=None,
        help="Resume after this many chunks. Safer than --start-batch because it "
             "stays correct if the batch size changes; pass the mentor's current "
             "vector count.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the mentor's existing vectors first (use after changing chunking)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clean and chunk locally without embedding or upserting",
    )
    args = parser.parse_args()

    # --skip-chunks wins; --start-batch is kept for existing muscle memory.
    if args.skip_chunks is not None:
        start_chunk = args.skip_chunks
    else:
        start_chunk = args.start_batch * BATCH_SIZE

    unknown = [m for m in args.mentors if m not in MENTORS]
    if unknown:
        parser.error(f"Unknown mentor(s): {', '.join(unknown)}. Choose from {MENTORS}")

    for mentor in args.mentors:
        if args.dry_run:
            docs = load_documents(f"./data/raw/{mentor}", mentor)
            chunks = chunk_documents(docs)
            print(f"  [dry run] {mentor}: {len(chunks)} chunks")
            for chunk in chunks[:2]:
                print(f"    citation: {chunk.metadata['citation']}")
                print(f"    text: {chunk.page_content[:100]!r}")
        else:
            ingest_mentor(mentor, start_chunk=start_chunk, clear=args.clear)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
