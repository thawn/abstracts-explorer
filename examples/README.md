# Abstracts Explorer Examples

This directory contains example scripts demonstrating key features of the Abstracts Explorer package.

## Examples

### RAG Chat with MCP Tools (`rag_with_mcp_tools.py`)

Demonstrates the automatic integration of MCP clustering tools with the RAG chat system.

**Features shown:**
- LLM automatically selecting appropriate tools based on questions
- Clustering analysis for general topic questions
- Topic evolution tracking over time
- Recent developments search
- Standard RAG paper retrieval
- Combined queries using both tools and RAG

**Requirements:**
- Embeddings created (`abstracts-explorer create-embeddings`)
- Papers downloaded in database
- LLM backend running (LM Studio or compatible)
- LLM must support function calling (e.g., GPT-3.5+, Gemma 3, Claude)

**Usage:**
```bash
# Make sure your .env file is configured
uv run python examples/rag_with_mcp_tools.py
```

**What to expect:**
- The script will demonstrate 5 different types of queries
- You'll see which MCP tools the LLM decides to call
- Responses will be generated based on tool results and/or paper retrieval

## Running Examples

All examples assume you've set up the environment:

1. **Install dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Configure environment** (create `.env` file):
   ```bash
   LLM_BACKEND_URL=http://localhost:1234
   CHAT_MODEL=your-model-name
   EMBEDDING_MODEL=your-embedding-model
   ```

3. **Download papers:**
   ```bash
   uv run abstracts-explorer download --year 2025
   ```

4. **Create embeddings:**
   ```bash
   uv run abstracts-explorer create-embeddings
   ```

5. **Run examples:**
   ```bash
   uv run python examples/rag_with_mcp_tools.py
   ```

## About MCP Tools

MCP (Model Context Protocol) tools are functions that the LLM can call to perform specialized tasks:

- **get_cluster_topics** - Analyze overall conference topics using clustering
- **get_topic_evolution** - Track how specific topics evolve over time
- **get_recent_developments** - Find the most recent papers in a topic
- **get_cluster_visualization** - Generate visualization data for clusters

These tools are automatically available to the RAG chat when `enable_mcp_tools=True` (default).

## Troubleshooting

### "No embeddings found"
Make sure to create embeddings first:
```bash
uv run abstracts-explorer create-embeddings
```

### "Connection refused" or LLM errors
Ensure your LLM backend is running:
- **LM Studio**: Start LM Studio and load a model that supports function calling
- Check the model is loaded and server is running on the configured port

### "Tool not supported"
If your LLM doesn't support function calling, disable MCP tools:
```python
chat = RAGChat(em, db, enable_mcp_tools=False)
```

## Contributing Examples

To add a new example:

1. Create a new Python file in this directory
2. Add clear comments explaining what the example demonstrates
3. Include error handling for common issues
4. Update this README with information about your example
5. Test the example works from a fresh environment

## See Also

- [RAG API Documentation](../docs/api/rag.md)
- [MCP Server Documentation](../docs/mcp_server.md)
- [Configuration Guide](../docs/configuration.md)
