#!/usr/bin/env python3
"""
Demo script for RAG chat feature.

This script demonstrates how to use the RAG chat programmatically
to query NeurIPS papers with natural language.
"""

from pathlib import Path
from neurips_abstracts import EmbeddingsManager, RAGChat
from neurips_abstracts.config import get_config


def main():
    """Demonstrate RAG chat capabilities."""

    print("=" * 70)
    print("RAG Chat Demo")
    print("=" * 70)

    # Get configuration
    config = get_config()

    # Check if embeddings exist
    embeddings_path = Path(config.embedding_db_path)
    if not embeddings_path.exists():
        print(f"\n❌ Embeddings database not found: {embeddings_path}")
        print("\nPlease create embeddings first:")
        print(f"  neurips-abstracts create-embeddings --db-path {config.paper_db_path}")
        return 1

    # Initialize embeddings manager
    print("\n📊 Loading embeddings...")
    em = EmbeddingsManager(config.embedding_db_path)
    em.connect()

    stats = em.get_collection_stats()
    print(f"✅ Loaded {stats['count']:,} papers from collection '{stats['name']}'")

    # Test LM Studio connection
    print("\n🔌 Testing LM Studio connection...")
    if not em.test_lm_studio_connection():
        print("❌ Failed to connect to LM Studio!")
        print("\nPlease ensure:")
        print("  - LM Studio is running at http://localhost:1234")
        print("  - A language model is loaded")
        em.close()
        return 1
    print("✅ Successfully connected to LM Studio")

    # Initialize RAG chat
    print("\n🤖 Initializing RAG chat system...")
    chat = RAGChat(embeddings_manager=em, max_context_papers=5, temperature=0.7)

    # Example queries
    queries = [
        "What are the main approaches to graph neural networks in recent papers?",
        "How do transformers compare to traditional RNNs?",
        "What are the latest techniques for training large language models?",
    ]

    print("\n" + "=" * 70)
    print("Running Example Queries")
    print("=" * 70)

    for i, question in enumerate(queries, 1):
        print(f"\n[Query {i}]")
        print(f"Q: {question}")
        print("-" * 70)

        # Query the system
        result = chat.query(question)

        # Display response
        print(f"A: {result['response'][:500]}...")
        print(f"\n📚 Based on {result['metadata']['n_papers']} papers:")

        # Show source papers
        for j, paper in enumerate(result["papers"][:3], 1):
            print(f"  {j}. {paper['title'][:60]}... (similarity: {paper['similarity']:.3f})")

        print()

    # Export conversation
    export_path = Path("rag_demo_conversation.json")
    chat.export_conversation(export_path)
    print(f"\n💾 Conversation exported to: {export_path}")

    # Show conversation stats
    print(f"📊 Conversation stats:")
    print(f"   - Total messages: {len(chat.conversation_history)}")
    print(f"   - User messages: {len([m for m in chat.conversation_history if m['role'] == 'user'])}")
    print(f"   - Assistant messages: {len([m for m in chat.conversation_history if m['role'] == 'assistant'])}")

    # Clean up
    em.close()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nTo start an interactive chat session, run:")
    print("  neurips-abstracts chat")
    print()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
