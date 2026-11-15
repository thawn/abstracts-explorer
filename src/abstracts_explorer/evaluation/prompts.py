"""Prompt templates for automatic evaluation."""

_QA_GENERATION_SYSTEM_PROMPT = """\
You are an expert evaluation dataset creator for a conference paper search and \
analysis system.  The system allows users to search papers, analyze research \
topics, track topic evolution, and visualize clusters.

Your task is to generate realistic evaluation queries that test the system's \
capabilities.  Each entry consists of a user *query* and the *tool_arguments* \
needed to call the corresponding MCP tool.  The actual expected answer will be \
produced by running the tool.

IMPORTANT RULES:
1. Queries must be natural questions a researcher would ask.
2. Each query must clearly map to the specified MCP tool.
3. tool_arguments must be a valid JSON object matching the tool's parameter schema.
4. Base queries on the provided paper information (topics, titles, conferences, years).
"""


_QA_GENERATION_USER_PROMPT = """\
Generate {n_pairs} evaluation query(ies) for the MCP tool "{tool_name}".

Tool description: {tool_description}
Example tool arguments: {tool_arguments_example}

Here are some papers from the database to base your queries on:
{papers_context}

Respond with a JSON array.  Each element must have these keys:
- "query": the user's natural-language question
- "tool_arguments": a JSON object of arguments to call the "{tool_name}" tool

Example format:
[
  {{
    "query": "What are the main topics in NeurIPS 2025?",
    "tool_arguments": {{"n_clusters": 8}}
  }}
]

Return ONLY the JSON array, no other text.
"""


_FOLLOWUP_GENERATION_PROMPT = """\
Given the following initial query and answer, generate {n_followups} natural \
follow-up question(s).  The follow-ups should explore the topic deeper or ask \
for clarification.

Initial query: {initial_query}
Initial answer: {initial_answer}

Tool to use: {tool_name}
Example tool arguments: {tool_arguments_example}

Available papers for context:
{papers_context}

Respond with a JSON array.  Each element must have these keys:
- "query": the follow-up question
- "tool_arguments": a JSON object of arguments to call the "{tool_name}" tool

Return ONLY the JSON array, no other text.
"""


_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's answer \
about conference papers.  Compare the actual answer to the expected reference \
answer.

Score the answer on a scale of 1-5:
  5 = Excellent: fully correct, comprehensive, well-structured
  4 = Good: mostly correct with minor omissions
  3 = Adequate: partially correct but missing important details
  2 = Poor: significant inaccuracies or missing key information
  1 = Bad: incorrect, irrelevant, or empty response

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""


_JUDGE_USER_PROMPT = """\
Query: {query}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Score the actual answer (1-5) and explain briefly.
"""
