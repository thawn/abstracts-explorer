# RAG Chat Feature Documentation

## Overview

The RAG (Retrieval Augmented Generation) chat feature provides an interactive conversational interface to query and discuss NeurIPS papers. It combines semantic search through embeddings with large language models to provide contextual, accurate responses about papers.

## Features

- **Interactive Chat**: Have natural conversations about papers
- **Context-Aware**: Uses relevant papers as context for responses
- **Conversation History**: Maintains conversation flow across multiple questions
- **Source Attribution**: Optionally shows which papers were used to generate responses
- **Export Conversations**: Save chat history to JSON for later reference
- **Customizable**: Adjust context size, temperature, and other parameters

## Architecture

The RAG system consists of three main components:

1. **Semantic Search**: Uses text embeddings to find relevant papers
2. **Context Building**: Formats paper abstracts and metadata as context
3. **Response Generation**: Uses LM Studio's language models to generate answers

```
User Question
     ↓
Semantic Search (Embeddings)
     ↓
Retrieve Top-K Papers
     ↓
Build Context (Papers + Abstracts)
     ↓
LLM Generation (LM Studio)
     ↓
Response + Source Papers
```

## Usage

### Basic Usage

Start a chat session:

```bash
neurips-abstracts chat
```

This will:
1. Connect to the embeddings database (`chroma_db` by default)
2. Connect to LM Studio at `http://localhost:1234`
3. Start an interactive chat session

### With Custom Settings

```bash
neurips-abstracts chat \
    --embeddings-path my_embeddings/ \
    --collection my_papers \
    --model llama-3.1-70b \
    --max-context 10 \
    --temperature 0.5 \
    --show-sources
```

### Export Conversation

Save your conversation to a JSON file:

```bash
neurips-abstracts chat --export conversation.json
```

## Chat Commands

During a chat session, you can use these commands:

- **`exit`, `quit`, `q`**: Exit the chat session
- **`reset`**: Clear conversation history
- **`help`**: Show available commands

## Example Session

```
$ neurips-abstracts chat --show-sources

======================================================================
NeurIPS RAG Chat
======================================================================
Embeddings: chroma_db
Collection: neurips_papers
Model: auto
LM Studio: http://localhost:1234
======================================================================

🔌 Testing LM Studio connection...
✅ Successfully connected to LM Studio

📊 Loaded 5,989 papers from collection 'neurips_papers'

💬 Chat started! Type 'exit' or 'quit' to end, 'reset' to clear history.
======================================================================

You: What are the main approaches to graph neural networks in this year's papers?

Assistant (based on 5 papers):
Based on the papers, there are several main approaches to graph neural networks:

1. **Message Passing GNNs** (Paper 1): The most common approach where nodes 
   aggregate information from their neighbors through message passing. The paper
   "Graph Attention Networks for Molecular Generation" uses this with attention
   mechanisms.

2. **Spectral Methods** (Paper 2): Papers like "Spectral Graph Convolution for
   Large-Scale Graphs" use graph Fourier transforms and spectral decomposition.

3. **Graph Transformers** (Paper 3, 4): Recent approaches adapt transformer
   architectures to graphs, as seen in "Graph Transformer Networks" and 
   "Efficient Graph Transformers via Sparse Attention".

4. **Equivariant GNNs** (Paper 5): Papers focusing on geometric deep learning
   ensure equivariance to graph symmetries.

📚 Source papers:
  1. Graph Attention Networks for Molecular Generation (similarity: 0.876)
  2. Spectral Graph Convolution for Large-Scale Graphs (similarity: 0.854)
  3. Graph Transformer Networks (similarity: 0.842)
  4. Efficient Graph Transformers via Sparse Attention (similarity: 0.831)
  5. Geometric Deep Learning on Molecular Graphs (similarity: 0.812)

You: Which of these is best for molecular generation?

Assistant (based on 5 papers):
For molecular generation specifically, **message passing GNNs with attention**
(Paper 1) tend to perform best because...

You: exit

👋 Goodbye!
💾 Conversation exported to: conversation.json
```

## Configuration Options

### `--embeddings-path`
Path to the ChromaDB vector database containing paper embeddings.
- Default: `chroma_db`

### `--collection`
Name of the collection within the ChromaDB database.
- Default: `neurips_papers`

### `--lm-studio-url`
URL for the LM Studio API server.
- Default: `http://localhost:1234`
- Make sure LM Studio is running and a language model is loaded

### `--model`
Name of the language model to use.
- Default: `auto` (uses the currently loaded model in LM Studio)
- Can specify model name for multi-model setups

### `--max-context`
Maximum number of papers to use as context for each response.
- Default: `5`
- Higher values provide more context but may slow responses
- Recommended range: 3-10 papers

### `--temperature`
Sampling temperature for text generation.
- Default: `0.7`
- Lower values (0.1-0.5): More focused, deterministic responses
- Higher values (0.8-1.0): More creative, diverse responses

### `--show-sources`
Display source papers used for each response.
- Default: `False`
- When enabled, shows paper titles and similarity scores

### `--export`
Path to export conversation history as JSON.
- Optional parameter
- Saves all messages when chat exits

## Python API

You can also use the RAG chat programmatically:

```python
from neurips_abstracts import EmbeddingsManager, RAGChat

# Initialize embeddings
em = EmbeddingsManager("chroma_db")
em.connect()

# Create chat instance
chat = RAGChat(
    embeddings_manager=em,
    lm_studio_url="http://localhost:1234",
    max_context_papers=5,
    temperature=0.7
)

# Query the system
result = chat.query("What are the latest advances in transformers?")
print(result["response"])
print(f"Based on {result['metadata']['n_papers']} papers")

# Continue conversation
result = chat.chat("Tell me more about the attention mechanism")
print(result["response"])

# Reset conversation
chat.reset_conversation()

# Export conversation
chat.export_conversation("conversation.json")

# Close
em.close()
```

## Requirements

1. **Embeddings Database**: Must have generated embeddings first:
   ```bash
   neurips-abstracts create-embeddings --db-path neurips_2025.db
   ```

2. **LM Studio**: Must be running with a language model loaded
   - Download from: https://lmstudio.ai/
   - Load a model (recommended: Llama 3.1, Mistral, or similar)
   - Ensure API server is enabled on port 1234

3. **Python Dependencies**: Already included in package
   - `requests` for API calls (standard library alternative available)

## Tips for Best Results

1. **Be Specific**: Ask focused questions about specific topics
2. **Use Context**: The system works best when papers are relevant to your query
3. **Iterate**: Follow-up questions can dive deeper into topics
4. **Review Sources**: Use `--show-sources` to see which papers informed the response
5. **Adjust Context**: Increase `--max-context` for complex questions
6. **Reset When Needed**: Use `reset` command if conversation goes off-track

## Troubleshooting

### "Failed to connect to LM Studio"
- Ensure LM Studio is running
- Check that a model is loaded
- Verify API server is enabled (Settings → Server → Enable)
- Confirm URL is correct (default: http://localhost:1234)

### "No relevant papers found"
- Question may be too specific or off-topic
- Try broader questions
- Check embeddings database has papers: `neurips-abstracts search "test"`

### Slow responses
- Reduce `--max-context` to use fewer papers
- Use a smaller/faster language model
- Check LM Studio performance settings

### Irrelevant responses
- Increase `--max-context` for more context
- Make questions more specific
- Lower `--temperature` for more focused responses

## Advanced Usage

### Custom System Prompts

When using the Python API, you can customize the system prompt:

```python
custom_prompt = """
You are an expert in machine learning reviewing NeurIPS papers.
Provide technical, detailed answers with specific citations.
Focus on methodological contributions and experimental results.
"""

result = chat.query(
    "Explain the attention mechanism",
    system_prompt=custom_prompt
)
```

### Metadata Filtering

Combine with metadata filters to focus on specific paper types:

```python
# Only use accepted papers
result = chat.query(
    "What are the best graph neural networks?",
    metadata_filter={"decision": "Accept (oral)"}
)
```

### Batch Processing

Process multiple questions programmatically:

```python
questions = [
    "What are the main deep learning architectures?",
    "How do transformers compare to RNNs?",
    "What are the latest optimization techniques?"
]

for question in questions:
    result = chat.query(question)
    print(f"Q: {question}")
    print(f"A: {result['response']}\n")
```

## Performance

- **Response Time**: Typically 5-15 seconds depending on model size
- **Context Size**: ~5 papers = ~2500 tokens context
- **Memory Usage**: Depends on LM Studio model (4-32GB typical)
- **Accuracy**: High for questions directly related to paper content

## Limitations

- Responses limited to information in paper abstracts
- Cannot access full paper text or figures
- Depends on quality of loaded language model
- Conversation history may need periodic resets for very long sessions
- Best with well-scoped, technical questions about papers

## Future Enhancements

Potential improvements for future versions:
- Streaming responses for faster perceived performance
- Multi-turn refinement with clarifying questions
- Automatic source citation in responses
- Support for full paper text (beyond abstracts)
- Custom retrieval strategies (hybrid search, re-ranking)
- Voice input/output support
- Web UI for chat interface
