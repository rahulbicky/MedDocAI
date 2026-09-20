"""
Builds/updates the Pinecone vector index from the medical PDFs in data/.

Usage:
    python store_index.py

Safe to re-run: chunks are upserted using a content-based deterministic ID,
so re-running this script updates existing vectors instead of duplicating
the whole dataset.
"""

import hashlib
import os
import sys

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME,
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
DATA_DIR = "data/"

if not PINECONE_API_KEY:
    sys.exit(
        "Missing PINECONE_API_KEY. Please add it to your .env file "
        "(see .env.example)."
    )


def make_chunk_id(chunk) -> str:
    """Deterministic ID from source path + content, so re-indexing upserts
    instead of duplicating vectors."""
    source = chunk.metadata.get("source", "unknown")
    digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}-{digest}"


def main() -> None:
    print(f"[1/5] Loading PDF files from '{DATA_DIR}'...")
    if not os.path.isdir(DATA_DIR) or not any(
        f.lower().endswith(".pdf") for f in os.listdir(DATA_DIR)
    ):
        sys.exit(f"No PDF files found in '{DATA_DIR}'. Add medical PDFs there first.")

    extracted_data = load_pdf_file(data=DATA_DIR)
    filtered_data = filter_to_minimal_docs(extracted_data)
    print(f"      Loaded {len(filtered_data)} page(s).")

    print("[2/5] Splitting documents into chunks...")
    text_chunks = text_split(filtered_data)
    print(f"      Created {len(text_chunks)} chunk(s).")

    print("[3/5] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = download_hugging_face_embeddings()

    print("[4/5] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"      Index '{PINECONE_INDEX_NAME}' not found. Creating it "
              f"(dimension={EMBEDDING_DIMENSION}, metric=cosine)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"      Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"      Index '{PINECONE_INDEX_NAME}' already exists. Reusing it.")

    print("[5/5] Embedding and upserting chunks into Pinecone...")
    docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    ids = [make_chunk_id(chunk) for chunk in text_chunks]
    batch_size = 100
    total = len(text_chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        docsearch.add_documents(documents=text_chunks[start:end], ids=ids[start:end])
        print(f"      Upserted {end}/{total} chunks...")

    print("Done. Documents successfully indexed in Pinecone.")


if __name__ == "__main__":
    main()
