#!/usr/bin/env python3
"""
Example: RAG Chat with MCP Tools Integration

This script demonstrates how the RAG chat automatically uses MCP clustering tools
to answer questions about conference topics, trends, and developments.

The LLM decides when to use clustering tools vs. paper retrieval based on the question.
"""

import logging
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.rag import RAGChat

# Enable logging to see tool calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("RAG Chat with MCP Tools Integration Demo")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing embeddings and database...")
    em = EmbeddingsManager()
    em.connect()
    em.create_collection()
    
    db = DatabaseManager()
    db.connect()
    
    # Create RAG chat with MCP tools enabled
    print("Creating RAG chat with MCP tools enabled...")
    chat = RAGChat(em, db, enable_mcp_tools=True)
    print()
    
    # Example 1: General topic analysis (uses get_cluster_topics)
    print("-" * 80)
    print("Example 1: General Topic Analysis")
    print("-" * 80)
    question1 = "What are the main research topics at this conference?"
    print(f"Question: {question1}")
    print()
    print("Expected: LLM will call get_cluster_topics() to analyze clusters")
    print()
    
    try:
        response1 = chat.query(question1)
        response_text = response1.get('response', '')
        metadata = response1.get('metadata', {})
        print(f"Response: {response_text[:500]}...")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 2: Topic evolution (uses get_topic_evolution)
    print("-" * 80)
    print("Example 2: Topic Evolution Analysis")
    print("-" * 80)
    question2 = "How have transformer architectures evolved at NeurIPS over the years?"
    print(f"Question: {question2}")
    print()
    print("Expected: LLM will call get_topic_evolution(topic_keywords='transformers')")
    print()
    
    try:
        response2 = chat.query(question2)
        response_text = response2.get('response', '')
        metadata = response2.get('metadata', {})
        print(f"Response: {response_text[:500]}...")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 3: Recent developments (uses get_recent_developments)
    print("-" * 80)
    print("Example 3: Recent Developments")
    print("-" * 80)
    question3 = "What are the latest papers on large language models?"
    print(f"Question: {question3}")
    print()
    print("Expected: LLM will call get_recent_developments(topic_keywords='large language models')")
    print()
    
    try:
        response3 = chat.query(question3)
        response_text = response3.get('response', '')
        metadata = response3.get('metadata', {})
        print(f"Response: {response_text[:500]}...")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 4: Specific paper query (uses standard RAG, no tools)
    print("-" * 80)
    print("Example 4: Specific Paper Query (Standard RAG)")
    print("-" * 80)
    question4 = "Explain the Vision Transformer paper"
    print(f"Question: {question4}")
    print()
    print("Expected: Standard RAG paper retrieval (no clustering tools)")
    print()
    
    try:
        response4 = chat.query(question4)
        response_text = response4.get('response', '')
        metadata = response4.get('metadata', {})
        papers = response4.get('papers', [])
        print(f"Response: {response_text[:500]}...")
        print(f"Metadata: {metadata}")
        print(f"Papers used: {len(papers)} papers")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 5: Combined query (might use both tools and RAG)
    print("-" * 80)
    print("Example 5: Combined Query")
    print("-" * 80)
    question5 = "What are the main topics at the conference, and can you explain papers about attention?"
    print(f"Question: {question5}")
    print()
    print("Expected: Might use both clustering tools AND paper retrieval")
    print()
    
    try:
        response5 = chat.query(question5)
        response_text = response5.get('response', '')
        metadata = response5.get('metadata', {})
        print(f"Response: {response_text[:500]}...")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Cleanup
    em.close()
    db.close()
    
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("- MCP tools are automatically used when appropriate")
    print("- LLM decides which tool to call based on the question")
    print("- Can combine tools with standard RAG paper retrieval")
    print("- Enable logging to see tool calls in action")
    print()
    print("To disable MCP tools: chat = RAGChat(em, db, enable_mcp_tools=False)")


if __name__ == "__main__":
    main()
