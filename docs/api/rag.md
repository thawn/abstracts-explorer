# RAG Module

The RAG (Retrieval-Augmented Generation) module provides a chat interface for querying papers using LLMs.

## Overview

The `RAGChat` class implements:

- Retrieval-Augmented Generation for paper queries
- Conversation history management
- Integration with LM Studio LLM backend
- Context building from relevant papers

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
