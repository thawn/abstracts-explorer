#!/usr/bin/env python3
"""
Add missing session and event_type metadata to existing ChromaDB embeddings.

This script updates the ChromaDB collection to include session and event_type
metadata from the SQLite database without regenerating embeddings.
"""

import sqlite3
from pathlib import Path
import chromadb
from tqdm import tqdm
from neurips_abstracts.config import get_config

# Get configuration
config = get_config()

# Paths
DB_PATH = Path(config.paper_db_path)
CHROMA_PATH = Path(config.embedding_db_path)
COLLECTION_NAME = config.collection_name

print("=" * 70)
print("Adding Missing Metadata to ChromaDB")
print("=" * 70)
print(f"Database: {DB_PATH}")
print(f"ChromaDB: {CHROMA_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print()

# Connect to SQLite database
print("📊 Loading paper data from SQLite...")
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all papers with session and eventtype (not event_type!)
cursor.execute("SELECT id, session, eventtype FROM papers")
papers_data = {
    str(row["id"]): {"session": row["session"] or "", "eventtype": row["eventtype"] or ""}
    for row in cursor.fetchall()
}
conn.close()

print(f"✅ Loaded {len(papers_data):,} papers from database")

# Connect to ChromaDB
print("\n🔌 Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"✅ Found collection with {collection.count():,} documents")

# Get all documents with their current metadata
print("\n📥 Fetching existing metadata...")
all_data = collection.get(include=["metadatas"])
doc_ids = all_data["ids"]
existing_metadata = all_data["metadatas"]

print(f"✅ Retrieved metadata for {len(doc_ids):,} documents")

# Update metadata
print("\n🔄 Adding session and event_type to metadata...")
updated_count = 0
missing_count = 0

for i, doc_id in enumerate(tqdm(doc_ids, desc="Updating metadata")):
    if doc_id in papers_data:
        # Get existing metadata
        current_meta = existing_metadata[i] or {}

        # Add session and eventtype
        current_meta["session"] = papers_data[doc_id]["session"]
        current_meta["eventtype"] = papers_data[doc_id]["eventtype"]

        # Update the document
        collection.update(ids=[doc_id], metadatas=[current_meta])
        updated_count += 1
    else:
        missing_count += 1

print(f"\n✅ Updated {updated_count:,} documents")
if missing_count > 0:
    print(f"⚠️  {missing_count} documents not found in SQLite database")

# Verify by sampling
print("\n🔍 Verifying updates (sampling 3 documents)...")
sample_data = collection.get(limit=3, include=["metadatas"])
for i, (doc_id, meta) in enumerate(zip(sample_data["ids"], sample_data["metadatas"])):
    print(f"\n  Document {i+1} (ID: {doc_id}):")
    print(f"    Session: {meta.get('session', 'N/A')}")
    print(f"    Event Type: {meta.get('eventtype', 'N/A')}")
    print(f"    Topic: {meta.get('topic', 'N/A')}")

print("\n" + "=" * 70)
print("✅ Metadata update complete!")
print("=" * 70)
