# RAG Module

The RAG (Retrieval-Augmented Generation) module provides a chat interface for querying papers using LLMs.

## Overview

The `RAGChat` class implements:

- Retrieval-Augmented Generation for paper queries
- Conversation history management
- Integration with LM Studio LLM backend
- Context building from relevant papers
- **NEW: MCP Clustering Tools Integration** - Automatic tool calling for topic analysis

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.rag
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Setup

```python
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat

# Initialize embeddings manager
em = EmbeddingsManager()

# Initialize RAG chat
chat = RAGChat(
    embeddings_manager=em,
    lm_studio_url="http://localhost:1234",
    model="gemma-3-4b-it-qat"
)
```

### Simple Query

```python
# Ask a question
response = chat.query(
    "What are the latest developments in transformer architectures?"
)

print(response)
```

### Conversation

```python
# Start a conversation
response1 = chat.query("Tell me about vision transformers")
print(response1)

# Continue the conversation (maintains context)
response2 = chat.chat("What are their main advantages?")
print(response2)

response3 = chat.chat("Can you explain the first paper in more detail?")
print(response3)
```

### Custom Parameters

```python
# Query with custom settings
response = chat.query(
    query="Explain self-attention mechanisms",
    n_results=10,              # Use 10 papers for context
    temperature=0.8,           # More creative responses
    max_tokens=2000,           # Longer responses
    system_prompt="You are a helpful research assistant."
)
```

## MCP Clustering Tools Integration

**NEW in v0.1.0**: RAGChat now integrates with MCP clustering tools, allowing the LLM to automatically analyze conference topics, trends, and developments.

### What Are MCP Tools?

MCP (Model Context Protocol) tools are specialized functions that the LLM can call to perform specific tasks. The RAG system includes four clustering tools:

1. **get_cluster_topics** - Analyze overall conference topics
2. **get_topic_evolution** - Track how topics evolve over time
3. **get_recent_developments** - Find recent papers in specific areas
4. **get_cluster_visualization** - Generate cluster visualizations

### How It Works

When MCP tools are enabled (default), the LLM automatically decides when to use clustering tools based on the user's question:

```python
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.rag import RAGChat

# Initialize components
em = EmbeddingsManager()
em.connect()
db = DatabaseManager()
db.connect()

# Create RAG chat with MCP tools enabled (default)
chat = RAGChat(em, db, enable_mcp_tools=True)

# Ask about general topics - LLM will use clustering tools
response = chat.query("What are the main research topics at NeurIPS?")
print(response)
# The LLM automatically calls get_cluster_topics() and analyzes the results

# Ask about trends - LLM uses topic evolution tool
response = chat.query("How have transformers evolved at NeurIPS over the years?")
print(response)
# The LLM calls get_topic_evolution(topic_keywords="transformers")

# Ask about recent work - LLM uses recent developments tool
response = chat.query("What are the latest papers on reinforcement learning?")
print(response)
# The LLM calls get_recent_developments(topic_keywords="reinforcement learning")
```

### Tool Selection

The LLM automatically decides which tool(s) to use based on the question:

- **Questions about "main topics", "themes", "areas"** → Uses `get_cluster_topics`
- **Questions about "evolution", "trends", "over time"** → Uses `get_topic_evolution`
- **Questions about "recent", "latest", "new"** → Uses `get_recent_developments`
- **Questions about specific papers** → Uses standard RAG (no tools)

### Disabling MCP Tools

If you want to disable MCP tools and use only standard RAG:

```python
# Disable MCP tools
chat = RAGChat(em, db, enable_mcp_tools=False)

# Now all queries use only paper embeddings search
response = chat.query("What are the main topics?")
# Will search for relevant papers instead of clustering
```

### Combining Tools with RAG

The LLM can use both tools AND paper retrieval in the same query:

```python
# Complex query that might use both
response = chat.query(
    "What are the main topics at NeurIPS, and can you explain "
    "the attention mechanism paper in detail?"
)
# The LLM might:
# 1. Call get_cluster_topics() to identify main topics
# 2. Search for "attention mechanism" papers
# 3. Combine both to generate comprehensive answer
```

### Tool Call Examples

Here are example questions that trigger different tools:

```python
# Cluster analysis
chat.query("What topics are covered in the conference?")
chat.query("Show me the main research areas")
chat.query("Analyze the distribution of papers by topic")

# Topic evolution
chat.query("How has deep learning research changed over time?")
chat.query("Show me the trend for transformer papers from 2020-2025")
chat.query("Has interest in GANs increased or decreased?")

# Recent developments
chat.query("What are the latest papers on LLMs?")
chat.query("Show me recent work in computer vision")
chat.query("What's new in the last 2 years for neural architecture search?")

# Specific papers (no tools, standard RAG)
chat.query("Explain the Vision Transformer paper")
chat.query("What does the BERT paper say about pre-training?")
chat.query("Find papers about attention mechanisms")
```

### Advanced: Tool Call Debugging

To see which tools the LLM is calling, check the logs:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

# Now tool calls will be logged
chat = RAGChat(em, db, enable_mcp_tools=True)
response = chat.query("What are the main topics?")
# Logs: "LLM requested tool: get_cluster_topics with args: {'n_clusters': 8}"
```

### Requirements

MCP tools require:

- **Embeddings created**: Run `abstracts-explorer create-embeddings` first
- **Papers downloaded**: Have conference data in database
- **Tool-capable LLM**: Model must support function calling (e.g., GPT-3.5+, Gemma 3, Claude)

If your LLM doesn't support function calling, disable MCP tools:

```python
chat = RAGChat(em, db, enable_mcp_tools=False)
```

### Metadata Filtering

```python
# Query papers from specific year
response = chat.query(
    "What are the main themes in 2025?",
    where={"year": 2025}
)

# Multiple filters
response = chat.query(
    "Explain recent attention mechanisms",
    where={
        "year": {"$gte": 2024},
    },
    n_results=5
)
```

## Conversation Management

### Reset Conversation

```python
# Clear conversation history
chat.reset_conversation()

# Start fresh conversation
response = chat.query("New topic...")
```

### Export Conversation

```python
# Export to JSON file
chat.export_conversation("conversation.json")

# Export returns the conversation data
conversation_data = chat.export_conversation("chat_history.json")
```

### Conversation Format

Exported conversations include:

```python
{
    "timestamp": "2025-11-26T10:00:00",
    "model": "gemma-3-4b-it-qat",
    "messages": [
        {
            "role": "user",
            "content": "What is a transformer?",
            "timestamp": "2025-11-26T10:00:00"
        },
        {
            "role": "assistant",
            "content": "A transformer is...",
            "papers_used": [
                {"id": "123", "title": "Attention Is All You Need"}
            ],
            "timestamp": "2025-11-26T10:00:05"
        }
    ]
}
```

## LLM Backend Configuration

### Supported Backends

The module is designed for LM Studio but works with any OpenAI-compatible API:

- **LM Studio** (default)
- OpenAI API
- LocalAI
- Ollama (with OpenAI compatibility)

### Authentication

```python
# With authentication token
chat = RAGChat(
    em,
    lm_studio_url="https://api.example.com",
    model="gpt-4",
    auth_token="sk-..."
)
```

### Custom Endpoints

```python
# Different backend URL
chat = RAGChat(
    em,
    lm_studio_url="http://localhost:8080",
    model="custom-model"
)
```

## Response Generation

### How RAG Works

1. **Retrieve**: Search for relevant papers using embeddings
2. **Augment**: Build context from paper abstracts
3. **Generate**: Send context + query to LLM
4. **Return**: Get AI-generated response with citations

### Context Building

The RAG system builds context from retrieved papers:

```
Context includes:
- Paper titles
- Paper abstracts
- Paper years
- Relevance scores

Formatted for optimal LLM comprehension
```

### System Prompts

Default system prompt:

```
You are a helpful research assistant specializing in NeurIPS papers.
Answer questions based on the provided paper abstracts.
Cite papers by title when referencing them.
```

Custom system prompt:

```python
chat.query(
    "Your question",
    system_prompt="You are an expert in computer vision..."
)
```

## Error Handling

```python
try:
    response = chat.query("What is deep learning?")
except requests.RequestException:
    print("LLM backend connection failed")
except ValueError as e:
    print(f"Invalid response: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Response Time

Factors affecting response time:

- Number of papers retrieved (n_results)
- LLM model size and speed
- Token generation length (max_tokens)
- Network latency to LLM backend

### Memory Usage

- Conversation history stored in memory
- Each message adds to context
- Use `reset_conversation()` for long sessions

### Optimization Tips

```python
# Faster responses
chat.query(query, n_results=3, max_tokens=500)

# More comprehensive but slower
chat.query(query, n_results=10, max_tokens=2000)

# Balance quality and speed
chat.query(query, n_results=5, max_tokens=1000)
```

## Best Practices

1. **Start specific** - Focused queries get better results
2. **Use filters** - Narrow search space with metadata filters
3. **Manage history** - Reset conversation when changing topics
4. **Export important conversations** - Save valuable interactions
5. **Adjust parameters** - Tune n_results and temperature for your needs
6. **Monitor backend** - Ensure LM Studio/LLM is running and responsive
7. **Handle errors** - Wrap calls in try-except for production use
