"""
Embeddings Demo
===============

This example demonstrates how to use the EmbeddingsManager to:
1. Generate embeddings for paper abstracts
2. Store them in a vector database
3. Perform semantic similarity search

Prerequisites
-------------
- LM Studio running locally on http://localhost:1234
- The text-embedding-qwen3-embedding-4b model loaded in LM Studio
- A SQLite database with NeurIPS papers (created using basic_usage.py)
"""

from pathlib import Path
from neurips_abstracts import EmbeddingsManager
from neurips_abstracts.config import get_config


def main():
    """Main function demonstrating embeddings functionality."""

    # Get configuration
    config = get_config()

    # Configuration
    db_path = Path("authors_demo.db")
    chroma_path = Path(config.embedding_db_path)

    print("NeurIPS Embeddings Demo")
    print("=" * 50)

    # Initialize the embeddings manager
    print("\n1. Initializing EmbeddingsManager...")
    em = EmbeddingsManager(
        lm_studio_url="http://localhost:1234",
        model_name="text-embedding-qwen3-embedding-4b",
        chroma_path=chroma_path,
        collection_name="neurips_papers",
    )

    # Test LM Studio connection
    print("\n2. Testing LM Studio connection...")
    if not em.test_lm_studio_connection():
        print("❌ Failed to connect to LM Studio!")
        print("   Please ensure:")
        print("   - LM Studio is running on http://localhost:1234")
        print("   - The text-embedding-qwen3-embedding-4b model is loaded")
        return
    print("✓ Successfully connected to LM Studio")

    # Connect to ChromaDB
    print("\n3. Connecting to ChromaDB...")
    em.connect()
    print(f"✓ Connected to ChromaDB at {chroma_path}")

    # Create collection
    print("\n4. Creating/retrieving collection...")
    em.create_collection(reset=False)  # Set reset=True to start fresh
    print(f"✓ Collection '{em.collection_name}' ready")

    # Get initial stats
    stats = em.get_collection_stats()
    print(f"\nCollection contains {stats['count']} papers")

    # Check if database exists
    if not db_path.exists():
        print(f"\n⚠️  Database not found: {db_path}")
        print("   Please run basic_usage.py first to create the database.")
        print("\n5. Adding sample papers manually...")

        # Add some sample papers
        sample_papers = [
            (
                1,
                "We present a novel deep learning architecture based on transformer "
                "models that achieves state-of-the-art performance on multiple benchmarks. "
                "Our approach uses attention mechanisms to capture long-range dependencies.",
                {
                    "title": "Deep Learning with Transformers",
                    "authors": "John Doe, Jane Smith",
                    "topic": "Machine Learning",
                    "decision": "Accept",
                },
            ),
            (
                2,
                "This paper introduces a new natural language processing technique "
                "for sentiment analysis using BERT-based models. We achieve 95% accuracy "
                "on standard datasets.",
                {
                    "title": "Advanced Sentiment Analysis",
                    "authors": "Alice Johnson",
                    "topic": "Natural Language Processing",
                    "decision": "Accept",
                },
            ),
            (
                3,
                "We propose a novel computer vision method for object detection in "
                "autonomous vehicles. Our approach uses convolutional neural networks "
                "and achieves real-time performance.",
                {
                    "title": "Real-time Object Detection",
                    "authors": "Bob Wilson, Carol Davis",
                    "topic": "Computer Vision",
                    "decision": "Accept",
                },
            ),
        ]

        print(f"   Adding {len(sample_papers)} sample papers...")
        for paper_id, abstract, metadata in sample_papers:
            em.add_paper(paper_id, abstract, metadata)
        print(f"✓ Added {len(sample_papers)} sample papers")

    else:
        # Embed papers from database
        print(f"\n5. Embedding papers from database: {db_path}")

        # Only embed accepted papers (optional filter)
        count = em.embed_from_database(
            db_path, batch_size=50, where_clause="decision = 'Accept' AND abstract IS NOT NULL AND abstract != ''"
        )
        print(f"✓ Successfully embedded {count} papers")

    # Get updated stats
    stats = em.get_collection_stats()
    print(f"\nCollection now contains {stats['count']} papers")

    # Perform similarity searches
    print("\n6. Performing similarity searches...")
    print("-" * 50)

    # Search 1: Deep learning
    print("\n🔍 Search: 'deep learning transformers'")
    results = em.search_similar("deep learning transformers", n_results=3)
    print_search_results(results)

    # Search 2: NLP
    print("\n🔍 Search: 'natural language processing'")
    results = em.search_similar("natural language processing", n_results=3)
    print_search_results(results)

    # Search 3: Computer vision
    print("\n🔍 Search: 'computer vision object detection'")
    results = em.search_similar("computer vision object detection", n_results=3)
    print_search_results(results)

    # Close connection
    print("\n7. Closing connections...")
    em.close()
    print("✓ Done!")

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nYou can now use the embeddings for:")
    print("  - Semantic paper search")
    print("  - Finding similar research")
    print("  - Clustering papers by topic")
    print("  - Building recommendation systems")


def print_search_results(results):
    """
    Print search results in a readable format.

    Parameters
    ----------
    results : dict
        Search results from ChromaDB.
    """
    if not results or not results["ids"] or len(results["ids"][0]) == 0:
        print("  No results found.")
        return

    for i, paper_id in enumerate(results["ids"][0], 1):
        metadata = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]
        document = results["documents"][0][i - 1]

        print(f"\n  {i}. Paper ID: {paper_id}")
        print(f"     Title: {metadata.get('title', 'N/A')}")
        print(f"     Authors: {metadata.get('authors', 'N/A')}")
        print(f"     Topic: {metadata.get('topic', 'N/A')}")
        print(f"     Similarity Score: {1 - distance:.4f}")
        print(f"     Abstract Preview: {document[:150]}...")


if __name__ == "__main__":
    main()
