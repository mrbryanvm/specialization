"""
chroma_helper.py — Build the Vector Database from the Business Catalog
======================================================================

This script reads the business catalog (catalogo_muebles.txt), splits it into
logical chunks, generates embeddings for each chunk, and stores everything in
a persistent ChromaDB database.

HOW IT WORKS (step by step):
1. TextLoader reads the raw .txt file
2. RecursiveCharacterTextSplitter breaks it into chunks of ~500 characters
   - Why 500? Small enough that each chunk is about 1 product or 1 FAQ
   - Big enough to keep all the details together (price, dimensions, etc.)
   - The separator "\n\n" ensures we split between products, not mid-sentence
3. HuggingFaceEndpointEmbeddings converts each chunk into a vector of 384 numbers
   - Calls HuggingFace's API (free tier) instead of loading the model locally
   - Same model "all-MiniLM-L6-v2", same quality, but uses ~5MB RAM instead of ~300MB
   - This makes it compatible with free cloud hosting (Render, Railway, etc.)
   - Requires a free HuggingFace token (HF_TOKEN) in your .env file
4. Chroma.from_documents() stores everything in a local folder (chroma_db/)
   - This folder IS the database. No external server needed.

RUN THIS SCRIPT:
    python chroma_helper.py

RUN IT AGAIN whenever you update the catalog file.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from pydantic import SecretStr


load_dotenv()

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

# Path to the catalog file (same folder as this script)
CATALOG_FILE = os.path.join(os.path.dirname(__file__), "catalogo_muebles.txt")

# Where ChromaDB will store the vector database
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# The embedding model — calls HuggingFace API (free), no local RAM needed
# Same model as before: all-MiniLM-L6-v2 (384-dimensional vectors)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace API token — free, needed to call the Inference API
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ChromaDB collection name (like a "table" in a regular database)
COLLECTION_NAME = "catalogo_muebleria_america"


def build_vector_database():
    """
    Main function: reads catalog → splits → embeds → stores in ChromaDB.
    """

    # ── Step 1: Load the catalog file ──
    print("📄 Step 1: Loading catalog file...")
    if not os.path.exists(CATALOG_FILE):
        print(f"❌ ERROR: Catalog file not found at: {CATALOG_FILE}")
        sys.exit(1)

    loader = TextLoader(CATALOG_FILE, encoding="utf-8")
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} document(s) from {CATALOG_FILE}")

    # ── Step 2: Split into chunks ──
    print("✂️  Step 2: Splitting into chunks...")

    # RecursiveCharacterTextSplitter tries to split by these separators IN ORDER:
    # First it tries "\n\n" (double newline = between products)
    # If that's not enough, it tries "\n" (single newline)
    # Last resort: splits mid-sentence at 500 chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Max characters per chunk
        chunk_overlap=50,        # Overlap between chunks (keeps context)
        separators=["\n\n", "\n"],  # Split on blank lines first
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks from the catalog")

    # Show a preview of the first 3 chunks so you can verify
    print("\n   📋 Preview of first 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:120].replace("\n", " ")
        print(f"   [{i+1}] {preview}...")

    # ── Step 3: Initialize the embedding model (via HuggingFace API) ──
    print("\n🧠 Step 3: Connecting to HuggingFace Inference API...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print("   (No local download needed — computations run on HuggingFace servers)")

    if not HF_TOKEN:
        print("   ❌ ERROR: HF_TOKEN not found in .env file!")
        print("   Get your free token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model=EMBEDDING_MODEL,
    )

    print("   ✅ Connected to HuggingFace API successfully")

    # ── Step 4: Create and persist ChromaDB ──
    print(f"\n💾 Step 4: Creating ChromaDB at {CHROMA_DB_DIR}...")

    # If the database already exists, delete it first (clean rebuild)
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)
        print("   🗑️  Removed old database (rebuilding from scratch)")

    # Create the vector store from our chunks
    # This is where the magic happens:
    # - Each chunk gets converted to a 384-dimensional vector
    # - The vector + original text get stored in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    print(f"   ✅ ChromaDB created with {len(chunks)} vectors!")

    # ── Step 5: Quick test — verify the search works ──
    print("\n🔍 Step 5: Testing semantic search...")

    test_queries = [
        "¿Tienen sofás de cuero?",
        "¿Cuánto cuesta la mesa Milano?",
        "¿Hacen delivery a Tiquipaya?",
        "¿Aceptan tarjeta de crédito?",
    ]

    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n   🔎 Query: \"{query}\"")
        for j, doc in enumerate(results):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"      Result {j+1}: {preview}...")

    print("\n" + "=" * 60)
    print("🎉 VECTOR DATABASE BUILT SUCCESSFULLY!")
    print(f"   📁 Location: {CHROMA_DB_DIR}")
    print(f"   📊 Total chunks: {len(chunks)}")
    print(f"   🧠 Embedding model: {EMBEDDING_MODEL}")
    print("=" * 60)
    print("\nYou can now start the server with: uvicorn server:app --reload --port 8000")


if __name__ == "__main__":
    build_vector_database()
