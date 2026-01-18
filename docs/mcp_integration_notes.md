# MCP Tools Integration in RAG Systems: Implementation Notes

## Overview

This document describes the implementation of MCP (Model Context Protocol) clustering tools integration into the RAG (Retrieval-Augmented Generation) chat system in Abstracts Explorer. This represents a state-of-the-art approach to combining semantic search with clustering analysis tools.

## Background Research

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol for enabling LLM applications to access external tools and data sources. It was developed by Anthropic and is gaining adoption across the AI ecosystem.

Key concepts:
- **MCP Servers** expose tools that can be called by LLM clients
- **MCP Clients** connect to servers and make tool calls available to LLMs
- **Tools** are functions with well-defined schemas that the LLM can invoke

### State-of-the-Art RAG + Tool Integration

Current best practices for integrating tools with RAG systems:

1. **Function Calling Native to LLMs**: Modern LLMs (GPT-3.5+, Claude 3+, Gemma 3+) have built-in function calling capabilities. The LLM can:
   - Decide when to call functions based on user queries
   - Select appropriate function(s) to call
   - Parse function results and incorporate them into responses

2. **Tool Schema in OpenAI Format**: Tools are described using JSON schema compatible with OpenAI's function calling API:
   ```json
   {
     "type": "function",
     "function": {
       "name": "tool_name",
       "description": "What the tool does",
       "parameters": { "type": "object", "properties": {...} }
     }
   }
   ```

3. **Automatic Tool Selection**: The LLM automatically decides:
   - Whether to use tools at all
   - Which tool(s) to use
   - What parameters to pass
   - How to combine tool results with retrieved documents

4. **Multi-Turn Conversations**: Tool calls can be part of multi-turn conversations:
   ```
   User → LLM (decides to call tool) → Tool execution → LLM (generates final response) → User
   ```

## Architecture

### Design Decisions

**1. Direct Integration vs. Separate MCP Server**

We chose **direct integration** over running a separate MCP server because:

- **Simpler deployment**: No need to manage separate server process
- **Lower latency**: Direct function calls instead of HTTP/IPC overhead
- **Easier development**: Shared code, dependencies, and configuration
- **Better error handling**: Unified exception handling and logging

The standalone MCP server is still available for other use cases (Claude Desktop, external tools).

**2. OpenAI Function Calling vs. Custom Prompt Engineering**

We use **OpenAI's native function calling** because:

- **Standardized**: Works with any OpenAI-compatible API
- **Reliable**: LLM has been trained for function calling
- **Flexible**: LLM decides when and how to use tools
- **Maintainable**: No custom prompt engineering needed

**3. Tool Execution Approach**

Tools are executed **directly in Python** rather than via IPC:

```python
# Direct execution (our approach)
result = get_cluster_topics(n_clusters=8)

# vs. IPC approach (not used)
result = mcp_client.call_tool("get_cluster_topics", {"n_clusters": 8})
```

Benefits:
- No serialization overhead
- Shared database connections
- Unified error handling
- Simpler code

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAGChat Class                            │
│  - Manages conversation history                             │
│  - Handles query rewriting                                  │
│  - Decides between tool use and paper retrieval             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              _generate_response() Method                     │
│  - Builds messages with conversation history                │
│  - Adds MCP tools schema if enabled                         │
│  - Calls OpenAI-compatible API                              │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├─── No tool calls ──────────────────┐
                │                                    │
                └─── Tool calls detected ───────┐   │
                                                │   │
                                                ▼   ▼
                ┌───────────────────────────────────────────┐
                │     _handle_tool_calls() Method           │
                │  - Executes each tool call                │
                │  - Formats results for LLM                │
                │  - Gets final response from LLM           │
                └───────────┬───────────────────────────────┘
                            │
                            ▼
                ┌─────────────────────────────────────────┐
                │      MCP Tools (mcp_tools.py)            │
                │  - execute_mcp_tool()                    │
                │  - format_tool_result_for_llm()          │
                └───────────┬─────────────────────────────┘
                            │
                            ▼
                ┌─────────────────────────────────────────┐
                │   Clustering Tools (mcp_server.py)       │
                │  - get_cluster_topics()                  │
                │  - get_topic_evolution()                 │
                │  - get_recent_developments()             │
                │  - get_cluster_visualization()           │
                └─────────────────────────────────────────┘
```

## Implementation Details

### 1. Tool Schema Definition (mcp_tools.py)

Tools are defined in OpenAI function calling format:

```python
MCP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_cluster_topics",
            "description": "Analyze clustered papers to identify topics...",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (default: 8)"
                    },
                    # ... more parameters
                },
                "required": []
            }
        }
    },
    # ... more tools
]
```

Key aspects:
- Clear, detailed descriptions help LLM decide when to use each tool
- Parameters include types, descriptions, defaults, and required flags
- Enums are used to constrain valid values

### 2. Enhanced System Prompt

When MCP tools are enabled, the system prompt is enhanced:

```python
if self.enable_mcp_tools:
    system_prompt += (
        "\n\nYou have access to clustering analysis tools that can help answer "
        "questions about overall conference topics, trends over time, and recent "
        "developments. Use these tools when appropriate for questions about general "
        "themes, topic evolution, or recent research in specific areas."
    )
```

This guides the LLM to use tools appropriately.

### 3. Tool Execution Flow

The execution flow handles multi-turn tool calling:

```python
# 1. Initial LLM call with tools
response = self.openai_client.chat.completions.create(
    model=self.model,
    messages=messages,
    tools=get_mcp_tools_schema(),
    tool_choice="auto"
)

# 2. Check for tool calls
if response.choices[0].message.tool_calls:
    # 3. Execute tools
    for tool_call in tool_calls:
        result = execute_mcp_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )
        
        # 4. Format result
        formatted = format_tool_result_for_llm(tool_call.function.name, result)
        
        # 5. Add to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": formatted
        })
    
    # 6. Get final response
    final_response = self.openai_client.chat.completions.create(
        model=self.model,
        messages=messages
    )
```

### 4. Tool Result Formatting

Raw tool results are JSON with statistics and data. We format them for better LLM consumption:

```python
# Raw result (complex JSON)
{
  "statistics": {"n_clusters": 8, "total_papers": 1000},
  "clusters": [
    {"cluster_id": 0, "keywords": [...], ...},
    ...
  ]
}

# Formatted for LLM (readable text)
Cluster Analysis Results:

Found 8 clusters covering 1000 papers.

Cluster 0 (150 papers):
  Top keywords: transformer (45), attention (38), neural (30)

Cluster 1 (120 papers):
  Top keywords: reinforcement (50), learning (40), policy (35)
...
```

This approach:
- Reduces token usage
- Makes results easier for LLM to understand
- Highlights most important information
- Maintains structure for LLM parsing

## Testing Strategy

### Unit Tests

1. **Tool Schema Tests** (`test_mcp_tools.py`):
   - Validate schema structure
   - Check parameter definitions
   - Verify all tools are present

2. **Tool Execution Tests**:
   - Mock tool implementations
   - Test parameter passing
   - Verify error handling
   - Check result formatting

3. **RAG Integration Tests** (`test_rag.py`):
   - Mock OpenAI responses with tool calls
   - Verify tool calling flow
   - Test multi-tool scenarios
   - Check system prompt enhancement

### Integration Tests

Integration tests with real LLM require:
- LM Studio or compatible backend running
- Model supporting function calling
- Embeddings created
- Papers in database

These are marked with `@pytest.mark.integration` and skipped by default.

## Best Practices Followed

### 1. Let the LLM Decide

We use `tool_choice="auto"` rather than forcing tool use. This allows:
- LLM to decide when tools are appropriate
- Natural mixing of tools and RAG
- Better responses when tools aren't needed

### 2. Clear Tool Descriptions

Tool descriptions include:
- What the tool does
- When to use it (use cases)
- Example scenarios

This helps the LLM make better decisions.

### 3. Graceful Degradation

If tools fail:
- Errors are caught and returned as JSON
- LLM can still generate response
- Error messages are informative

```python
try:
    result = execute_tool()
except Exception as e:
    return json.dumps({"error": f"Tool failed: {str(e)}"})
```

### 4. Formatted Results

Tool results are formatted to:
- Reduce token usage (extract key information)
- Improve LLM comprehension (text instead of JSON)
- Highlight important data (top clusters, trends)

### 5. Enable/Disable Support

MCP tools can be disabled:
- For LLMs that don't support function calling
- For simpler, faster queries
- For debugging and comparison

```python
chat = RAGChat(em, db, enable_mcp_tools=False)
```

## Performance Considerations

### Latency

Tool calls add latency:
- Initial LLM call: ~2-3 seconds
- Tool execution: ~1-10 seconds (depends on clustering size)
- Final LLM call: ~2-3 seconds
- **Total**: 5-16 seconds for tool-based queries

Compare to:
- Standard RAG: ~2-3 seconds

This is acceptable for questions requiring clustering analysis.

### Token Usage

- Tools schema: ~1500 tokens added to each request
- Tool results: ~500-2000 tokens (formatted)
- Total overhead: ~2000-3500 tokens per tool call

This is acceptable for the value provided.

### Optimization Opportunities

1. **Cache clustering results**: Don't re-cluster for similar queries
2. **Limit result size**: Return top N clusters only
3. **Async tool execution**: Execute multiple tools in parallel
4. **Result caching**: Cache formatted results

## Lessons Learned

### What Worked Well

1. **Direct integration**: Simpler than MCP client-server
2. **OpenAI function calling**: Reliable and well-supported
3. **Result formatting**: Improves LLM comprehension
4. **Auto tool selection**: LLM makes good decisions

### Challenges Overcome

1. **Type hints**: Mixed Dict types for messages (solved with Dict[str, Any])
2. **Tool result size**: Large JSON responses (solved with formatting)
3. **Error handling**: Tool failures (solved with try-except)
4. **Testing**: Mocking OpenAI tool calls (solved with detailed mocks)

### Future Improvements

1. **Streaming**: Stream tool execution progress
2. **Parallel tools**: Execute multiple tools concurrently
3. **Tool chaining**: Allow tools to call other tools
4. **Result caching**: Cache expensive clustering operations
5. **Adaptive clustering**: Adjust clustering parameters based on results

## Conclusion

This implementation represents a state-of-the-art integration of MCP clustering tools with RAG chat. Key innovations:

1. **Automatic tool selection**: LLM decides when to use clustering vs. retrieval
2. **Seamless integration**: No separate MCP server required
3. **Intelligent formatting**: Tool results optimized for LLM consumption
4. **Flexible architecture**: Can easily add new tools

The system successfully combines the strengths of:
- **RAG**: Specific paper retrieval and detailed Q&A
- **Clustering tools**: High-level topic analysis and trends

This provides users with a more powerful and versatile research assistant.

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [Best Practices for RAG with Tools](https://www.anthropic.com/research/tool-use)
- Abstracts Explorer Documentation (docs/api/rag.md)
