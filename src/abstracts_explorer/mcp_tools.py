"""
MCP Tools Integration for RAG Chat
==================================

This module provides integration between MCP clustering tools and the RAG chat system.
It converts MCP tool definitions to OpenAI function calling format and handles tool execution.

The integration allows the LLM to automatically decide when to use clustering tools
to answer questions about conference topics, trends, and developments.
"""

import copy
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, Tuple

from abstracts_explorer.mcp_server import (
    get_conference_topics,
    get_topic_evolution,
    search_papers,
    get_cluster_visualization,
    analyze_topic_relevance,
    get_paper_details,
    summarize_interesting_papers,
    recommend_related_papers,
)

logger = logging.getLogger(__name__)


class MCPToolsError(Exception):
    """Exception raised for MCP tools-related errors."""

    pass


def _abbreviate_result(text: str, max_length: int = 200) -> str:
    """
    Abbreviate a result string for logging.

    Parameters
    ----------
    text : str
        The text to abbreviate.
    max_length : int
        Maximum number of characters to keep (default: 200).

    Returns
    -------
    str
        The original text if short enough, otherwise truncated with '…'.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "…"


def _identity(value: Any) -> Any:
    """Return *value* unchanged."""
    return value


def _as_list(value: Any) -> List[Any]:
    """Wrap a scalar in a list; keep lists unchanged."""
    return value if isinstance(value, list) else [value]


def _join_list(value: Any) -> Any:
    """Join a list of keywords into a space-separated string."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return value


def _first_or_none(value: Any) -> Any:
    """Return the first list element, or *value* when it is not a list."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _first_int_or_none(value: Any) -> Optional[int]:
    """Convert a scalar or first list element to an int."""
    value = _first_or_none(value)
    if value is None:
        return None
    return int(value)


def _to_int_list_or_none(value: Any) -> Optional[List[int]]:
    """Convert a value to a list of ints, discarding invalid entries."""
    values = value if isinstance(value, list) else [value]
    converted: List[int] = []
    for item in values:
        try:
            converted.append(int(item))
        except (TypeError, ValueError):
            continue
    return converted or None


def _to_conference_list(value: Any) -> Any:
    """Normalize a conference value to list form."""
    if isinstance(value, list):
        return value
    if value is None:
        return None
    return [value]


def _to_paper_ids(value: Any) -> Any:
    """Split a comma-separated paper ID string into a list."""
    if isinstance(value, str):
        return [uid.strip() for uid in value.split(",") if uid.strip()]
    return value


@dataclass(frozen=True)
class NormRule:
    """
    One normalization rule for LLM-generated tool arguments.

    Parameters
    ----------
    target : str
        Canonical argument name.
    aliases : tuple of str, optional
        Alternate names produced by the LLM.  Aliases are dropped when the
        canonical name is already present.
    convert : callable, optional
        Conversion applied to the selected value.
    convert_existing : bool, optional
        Whether to convert the value when the canonical name is already
        present.  Defaults to ``True``.
    drop_on_error : bool, optional
        Remove the canonical argument when conversion raises.  Otherwise the
        argument is set to ``None``.
    """

    target: str
    aliases: Tuple[str, ...] = ()
    convert: Callable[[Any], Any] = _identity
    convert_existing: bool = True
    drop_on_error: bool = False


SEARCH_PAPERS_RULES = [
    NormRule("years", ("year",), _as_list, convert_existing=False),
    NormRule("topic_keywords", ("query",), _join_list),
    NormRule("conference", ("conferences",), _first_or_none),
    NormRule(
        "n_results",
        ("limit", "num_results", "max_results", "count"),
        int,
        drop_on_error=True,
    ),
]

GET_TOPIC_EVOLUTION_RULES = [
    NormRule("topic_keywords", (), _join_list),
    NormRule("start_year", (), _first_or_none),
    NormRule("end_year", (), _first_or_none),
]

ANALYZE_TOPIC_RELEVANCE_RULES = [
    NormRule("topic", (), _join_list),
    NormRule("conferences", ("conference",), _to_conference_list),
]

GET_PAPER_DETAILS_RULES = [
    NormRule("year", (), _first_int_or_none),
    NormRule("conference", (), _first_or_none),
]


def _interesting_paper_rules(array_conference: bool) -> List[NormRule]:
    """Build normalization rules for the interesting-paper tools."""
    if array_conference:
        conference_rule = NormRule("conferences", ("conference",), _to_conference_list)
    else:
        conference_rule = NormRule("conference", ("conferences",), _first_or_none)

    return [
        NormRule("paper_ids", ("ids", "paper_uids", "uids"), _to_paper_ids),
        NormRule("years", ("year",), _to_int_list_or_none),
        conference_rule,
        NormRule(
            "n_results",
            ("limit", "num_results", "max_results", "count"),
            int,
            drop_on_error=True,
        ),
    ]


INTERESTING_PAPER_SINGLE_RULES = _interesting_paper_rules(array_conference=False)
INTERESTING_PAPER_ARRAY_RULES = _interesting_paper_rules(array_conference=True)


def _normalize_args(arguments: Dict[str, Any], rules: List[NormRule]) -> Dict[str, Any]:
    """
    Normalize argument shapes produced by LLMs using declarative rules.

    Parameters
    ----------
    arguments : dict
        Raw arguments dict from the LLM.
    rules : list of NormRule
        Rules to apply in order.

    Returns
    -------
    dict
        A new dict with normalized argument values.
    """
    args = dict(arguments)

    for rule in rules:
        if rule.target in args:
            value = args[rule.target]
            for alias in rule.aliases:
                args.pop(alias, None)
            if not rule.convert_existing:
                continue
        else:
            alias = next((name for name in rule.aliases if name in args), None)
            if alias is None:
                continue
            value = args.pop(alias)
            for other_alias in rule.aliases:
                args.pop(other_alias, None)

        try:
            args[rule.target] = rule.convert(value)
        except (ValueError, TypeError):
            if rule.drop_on_error:
                args.pop(rule.target, None)
            else:
                args[rule.target] = None

    return args


def _execute_tool(
    func: Callable,
    arguments: Dict[str, Any],
    rules: List[NormRule],
) -> Any:
    """Normalize arguments, filter unknown keys, and invoke *func*."""
    args = _filter_unknown_kwargs(func, _normalize_args(arguments, rules))
    return func(**args)


def _filter_unknown_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out keyword arguments that are not accepted by *func*, logging a
    warning for each unexpected key.

    This makes MCP tool dispatch tolerant of extra keys that an LLM may send
    (e.g. it produces ``{"year": 2025}`` in addition to ``{"years": [2025]}``
    after normalisation has already renamed the field).

    Parameters
    ----------
    func : callable
        The target function whose signature is used to determine valid keys.
    kwargs : dict
        Keyword arguments intended for *func*.

    Returns
    -------
    dict
        A copy of *kwargs* with unrecognised keys removed.
    """
    try:
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())
        # If the function accepts **kwargs itself, pass everything through
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            return dict(kwargs)
    except (ValueError, TypeError):
        # If we can't inspect the signature, pass everything through unchanged
        return dict(kwargs)

    filtered: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in valid_params:
            filtered[key] = value
        else:
            logger.warning(f"Ignoring unknown argument '{key}' for {func.__name__}(); " "this key will be dropped.")
    return filtered


MCP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_topic_relevance",
            "description": (
                "Analyze the relevance of a research topic by counting papers within a specified "
                "distance in embedding space. Use this tool when the user asks about: topic relevance, "
                "popularity of a research area, how many papers cover a topic, or identifying significant "
                "research themes at a conference. A conference must be specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or research question to analyze (e.g., 'Uncertainty quantification')",
                    },
                    "distance_threshold": {
                        "type": "number",
                        "description": "Maximum Euclidean distance to consider papers relevant (default: 1.1)",
                    },
                    "conferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conference names to analyze (e.g., ['NeurIPS']). Required.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter by specific years (e.g., [2024, 2025])",
                    },
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic", "conferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conference_topics",
            "description": (
                "Get the main research topics of a conference. "
                "Use this tool when the user asks about: overall themes, main topics, research areas, "
                "or wants to understand what topics are covered in the conference. "
                "Returns topic names, representative keywords, paper counts, and example titles. "
                "A conference must be specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conference names (e.g., ['NeurIPS']). Required.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter by specific years (e.g., [2024, 2025])",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional, uses config default)",
                    },
                },
                "required": ["conferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_evolution",
            "description": (
                "Analyze how specific topics have evolved over the years. "
                "Use this tool when the user asks about: trends over time, historical development, how a topic has developed,"
                "how a topic has changed, or evolution of research areas. At least one conference must be specified. "
                "If requested, multiple conferences can be compared in the same analysis."
                "The chat frontend can use the returned data to generate a plot with plotly.js showing the topic evolution over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_keywords": {
                        "type": "string",
                        "description": "Keywords describing the topic (e.g., 'transformers attention', 'reinforcement learning')",
                    },
                    "conferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conference names to analyze (e.g., ['NeurIPS', 'ICLR']). Required.",
                    },
                    "start_year": {"type": "integer", "description": "Start year for analysis (inclusive)"},
                    "end_year": {"type": "integer", "description": "End year for analysis (inclusive)"},
                    "distance_threshold": {
                        "type": "number",
                        "description": (
                            "Maximum Euclidean distance in embedding space to consider papers "
                            "relevant (default: 1.1). Lower values mean stricter matching."
                        ),
                    },
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic_keywords", "conferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search for papers on a specific topic. "
                "Use this tool when the user asks about: papers on a topic, research about something, "
                "specific work, or wants to find papers related to a particular area. "
                "Can filter by specific years or search all years. A conference must be specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_keywords": {
                        "type": "string",
                        "description": "Keywords describing the topic to search for",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of specific years to filter by (e.g., [2024, 2025]). If not provided, searches all years.",
                    },
                    "n_results": {"type": "integer", "description": "Number of papers to return (default: 10)"},
                    "conference": {
                        "type": "string",
                        "description": "Conference name to search (e.g., 'NeurIPS', 'ICLR'). Required.",
                    },
                    "where": {"type": "object", "description": "Custom ChromaDB WHERE clause for filtering"},
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic_keywords", "conference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_visualization",
            "description": (
                "Retrieve pre-computed visualization data for clustered embeddings. "
                "Use this tool when the user asks for: a visual representation, graphical view, "
                "or wants to see clusters displayed. A conference must be specified."
                "The chat frontend can use the returned data to generate a plot with plotly.js showing the clusters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conference names to retrieve visualization for (e.g., ['NeurIPS']). Required.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter by specific years (e.g., [2024, 2025])",
                    },
                    "output_path": {"type": "string", "description": "Path to save visualization JSON (optional)"},
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["conferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": (
                "Get detailed information about papers from the database, including authors, "
                "URLs, PDF links, session information, keywords, and awards. "
                "Use this tool when the user asks about: who wrote a paper, paper authors, "
                "where to find a paper, PDF or poster links, session or room details, "
                "paper awards, or any other metadata about a specific paper."
                "must specify either paper_id or title to identify the paper."
                "Conference + year are optional but can help disambiguate papers with similar titles or multiple versions. "
                "Do not use for searching for papers on a topic; use the 'search_papers' tool for that instead. "
                "This tool is useful for specific follow-up questions after searching for papers using semantic search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title or partial title to search for (case-insensitive)",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Unique paper identifier (uid or original conference/OpenReview ID)",
                    },
                    "conference": {
                        "type": "string",
                        "description": "Filter by conference name (e.g., 'NeurIPS', 'ICLR')",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Filter by publication year",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of papers to return when searching by title (default: 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_interesting_papers",
            "description": (
                "Summarize the papers the user marked, starred, liked, or rated as interesting. "
                "Use this tool when the user asks for an overview or summary of their interesting papers, "
                "for example 'Give me a summary about the papers I marked as interesting for ICLR 2026'. "
                "The web UI supplies the user's ratings automatically; do not ask the user for paper IDs. "
                "Returns rating statistics, conferences, years, sessions, search terms, top keywords, "
                "highest-rated papers, and compact paper records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional paper UIDs. Usually supplied automatically from the web UI.",
                    },
                    "priorities": {
                        "type": "object",
                        "description": (
                            "Optional mapping of paper UID to a dict with 'priority' and optional "
                            "'searchTerm'. Usually supplied automatically from the web UI."
                        ),
                    },
                    "conference": {
                        "type": "string",
                        "description": "Filter by conference (e.g., 'ICLR' or 'ICLR 2026').",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter by years (e.g., [2026]).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_related_papers",
            "description": (
                "Recommend further papers based on the papers the user marked as interesting. "
                "Use this tool when the user asks for suggestions, recommendations, or related papers "
                "based on their starred/interesting papers. "
                "The web UI supplies the user's ratings automatically; do not ask the user for paper IDs. "
                "Already-rated papers are excluded. Each recommendation includes a relevance score and reasons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional paper UIDs. Usually supplied automatically from the web UI.",
                    },
                    "priorities": {
                        "type": "object",
                        "description": (
                            "Optional mapping of paper UID to a dict with 'priority' and optional "
                            "'searchTerm'. Usually supplied automatically from the web UI."
                        ),
                    },
                    "conferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict recommendations to these conferences.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Restrict recommendations to these years (e.g., [2026]).",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of recommendations to return (default: 10).",
                    },
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional paper UIDs to exclude.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional).",
                    },
                },
                "required": [],
            },
        },
    },
]


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute an MCP tool with the given arguments.

    Arguments are normalized before dispatch to handle common LLM output
    quirks (e.g. ``"year"`` instead of ``"years"``, list values for scalar
    string fields).  After normalization, any keyword arguments that are not
    accepted by the target function are silently dropped with a ``WARNING``
    log entry so that tools never raise ``TypeError`` for unexpected keys.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute
    arguments : dict
        Arguments to pass to the tool

    Returns
    -------
    str
        Tool execution result (JSON string)

    Raises
    ------
    MCPToolsError
        If tool execution fails or tool is unknown
    """
    logger.info("Executing MCP tool: %s with arguments: %s", tool_name, arguments)

    try:
        if tool_name == "analyze_topic_relevance":
            result = _execute_tool(analyze_topic_relevance, arguments, ANALYZE_TOPIC_RELEVANCE_RULES)
        elif tool_name == "get_conference_topics":
            result = _execute_tool(get_conference_topics, arguments, [])
        elif tool_name == "get_topic_evolution":
            result = _execute_tool(get_topic_evolution, arguments, GET_TOPIC_EVOLUTION_RULES)
        elif tool_name == "search_papers":
            result = _execute_tool(search_papers, arguments, SEARCH_PAPERS_RULES)
        elif tool_name == "get_cluster_visualization":
            result = _execute_tool(get_cluster_visualization, arguments, [])
        elif tool_name == "get_paper_details":
            result = _execute_tool(get_paper_details, arguments, GET_PAPER_DETAILS_RULES)
        elif tool_name == "summarize_interesting_papers":
            result = _execute_tool(
                summarize_interesting_papers,
                arguments,
                INTERESTING_PAPER_SINGLE_RULES,
            )
        elif tool_name == "recommend_related_papers":
            result = _execute_tool(
                recommend_related_papers,
                arguments,
                INTERESTING_PAPER_ARRAY_RULES,
            )
        else:
            # Return error JSON for unknown tools
            error_result = {"error": f"Unknown MCP tool: {tool_name}"}
            result = json.dumps(error_result, indent=2)

        logger.info("MCP tool %s returned: %s", tool_name, _abbreviate_result(result))
        return result

    except Exception as e:
        logger.error(f"MCP tool execution failed: {str(e)}")
        error_result = {"error": f"Tool execution failed: {str(e)}"}
        return json.dumps(error_result, indent=2)


def get_mcp_tools_schema(
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Get the MCP tools schema for OpenAI function calling.

    When *conferences* or *years* are provided (typically queried from the
    database), they are injected as ``enum`` constraints into every tool
    property that accepts conference or year values.  This lets the LLM
    pick from the exact values stored in the database instead of guessing.

    Parameters
    ----------
    conferences : list of str, optional
        Available conference names (e.g. ``["NeurIPS", "ICLR"]``).
        Injected as ``enum`` into conference-related properties.
    years : list of int, optional
        Available years (e.g. ``[2024, 2025]``).
        Injected as ``enum`` into year-related properties.

    Returns
    -------
    list
        List of tool definitions in OpenAI format
    """
    schema: List[Dict[str, Any]] = copy.deepcopy(MCP_TOOLS_SCHEMA)

    if not conferences and not years:
        return schema

    for tool in schema:
        func_def: Dict[str, Any] = tool.get("function", {})
        params: Dict[str, Any] = func_def.get("parameters", {})
        props: Dict[str, Any] = params.get("properties", {})

        if conferences:
            # Single-value conference fields (type: string)
            if "conference" in props and props["conference"].get("type") == "string":
                props["conference"]["enum"] = conferences

            # Array conference fields (type: array, items: {type: string})
            if "conferences" in props and props["conferences"].get("type") == "array":
                items = props["conferences"].get("items", {})
                if items.get("type") == "string":
                    items["enum"] = conferences

        if years:
            # Array year fields (type: array, items: {type: integer})
            if "years" in props and props["years"].get("type") == "array":
                items = props["years"].get("items", {})
                if items.get("type") == "integer":
                    items["enum"] = years

            # Single integer year fields
            for key in ("year", "start_year", "end_year"):
                if key in props and props[key].get("type") == "integer":
                    props[key]["enum"] = years

    return schema


def format_tool_result_for_llm(tool_name: str, result: str) -> str:
    """
    Format tool execution result for LLM consumption.

    This function extracts the most relevant information from tool results
    and formats it in a way that's easy for the LLM to process.

    Parameters
    ----------
    tool_name : str
        Name of the tool that was executed
    result : str
        Raw tool result (JSON string)

    Returns
    -------
    str
        Formatted result suitable for LLM processing
    """
    try:
        result_data = json.loads(result)

        # Check for errors
        if "error" in result_data:
            return f"Tool execution failed: {result_data['error']}"

        # Format based on tool type
        if tool_name == "analyze_topic_relevance":
            return _format_topic_relevance_result(result_data)
        elif tool_name == "get_conference_topics":
            return _format_conference_topics_result(result_data)
        elif tool_name == "get_topic_evolution":
            return _format_topic_evolution_result(result_data)
        elif tool_name == "search_papers":
            return _format_search_papers_result(result_data)
        elif tool_name == "get_cluster_visualization":
            return _format_visualization_result(result_data)
        elif tool_name == "get_paper_details":
            return _format_paper_details_result(result_data)
        elif tool_name == "summarize_interesting_papers":
            return _format_summarize_interesting_papers_result(result_data)
        elif tool_name == "recommend_related_papers":
            return _format_recommend_related_papers_result(result_data)
        else:
            # Return raw result for unknown tools
            return result

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool result as JSON: {result[:100]}...")
        return result


def _format_topic_relevance_result(data: Dict[str, Any]) -> str:
    """Format topic relevance result for LLM."""
    lines = [f"Topic Relevance Analysis for '{data.get('topic', 'unknown')}':\n"]

    total = data.get("total_papers", 0)
    total_considered = data.get("total_considered", 0)
    distance = data.get("distance_threshold", 0)
    relevance = data.get("relevance_score", 0)

    lines.append(f"Papers found: {total}/{total_considered} within distance {distance}")
    lines.append(f"Relevance score: {relevance}/100\n")

    if total > 0:
        # Show conferences
        conferences = data.get("conferences", {})
        if conferences:
            lines.append("Conferences:")
            for conf, count in list(conferences.items())[:5]:
                lines.append(f"  {conf}: {count} papers")

        # Show years
        years = data.get("years", {})
        if years:
            lines.append("\nYears:")
            for year, count in sorted(years.items()):
                lines.append(f"  {year}: {count} papers")

        # Show sample papers
        sample_papers = data.get("sample_papers", [])
        if sample_papers:
            lines.append("\nClosest papers:")
            for i, paper in enumerate(sample_papers[:3], 1):
                title = paper.get("title", "Unknown")
                dist = paper.get("distance", 0)
                lines.append(f"  {i}. {title} (distance: {dist:.3f})")

        closest = data.get("closest_distance")
        if closest is not None:
            lines.append(f"\nClosest paper distance: {closest:.3f}")
    else:
        lines.append("\nNo papers found matching the topic within the distance threshold.")

    return "\n".join(lines)


def _format_conference_topics_result(data: Dict[str, Any]) -> str:
    """Format conference topics result for LLM."""
    conference = data.get("conference", "unknown")
    total = data.get("total_papers", 0)
    n_topics = data.get("n_topics", 0)
    lines = [f"Conference Topics for {conference} ({total} papers, {n_topics} topics):\n"]

    topics = data.get("topics", [])
    for topic in topics[:10]:  # Limit to top 10 topics
        name = topic.get("topic") or "Unnamed"
        paper_count = topic.get("paper_count", 0)
        keywords = topic.get("keywords", [])

        lines.append(f"\n{name} ({paper_count} papers):")
        if keywords:
            lines.append(f"  Keywords: {', '.join(keywords[:8])}")

    return "\n".join(lines)


def _format_topic_evolution_result(data: Dict[str, Any]) -> str:
    """Format topic evolution result for LLM."""
    lines = [f"Topic Evolution Analysis for '{data.get('topic', 'unknown')}':\n"]

    conference_data = data.get("conference_data", {})
    if conference_data:
        for conference, cdata in conference_data.items():
            lines.append(f"Conference: {conference}")
            year_counts = cdata.get("year_counts", {})
            year_relative = cdata.get("year_relative", {})
            if year_counts:
                lines.append("  Papers per year:")
                for year, count in sorted(year_counts.items()):
                    rel = year_relative.get(year, year_relative.get(str(year), 0))
                    lines.append(f"    {year}: {count} papers ({rel}%)")
            lines.append("")

    total = data.get("total_papers", 0)
    lines.append(f"Total papers found: {total}")

    return "\n".join(lines)


def _format_search_papers_result(data: Dict[str, Any]) -> str:
    """Format search papers result for LLM."""
    lines = [f"Search Results for '{data.get('topic', 'unknown')}':\n"]

    papers = data.get("papers", [])
    years_filter = data.get("years_filter")
    if years_filter:
        lines.append(f"Filtered by years: {years_filter}")
    lines.append(f"Found {len(papers)} papers:\n")

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        year = paper.get("year", "")
        lines.append(f"{i}. {title} ({year})")

        # Add abstract snippet if available
        abstract = paper.get("abstract", "")
        if abstract:
            snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
            lines.append(f"   {snippet}")

    return "\n".join(lines)


def _format_visualization_result(data: Dict[str, Any]) -> str:
    """Format visualization result for LLM."""
    lines = ["Cluster Visualization Data Generated:\n"]

    stats = data.get("statistics", {})
    n_points = data.get("n_points", 0)
    n_dims = data.get("n_dimensions", 0)

    lines.append(f"Generated {n_dims}D visualization with {n_points} points")
    lines.append(f"Clusters: {stats.get('n_clusters', 0)}")

    if data.get("visualization_saved"):
        lines.append(f"Saved to: {data.get('output_path')}")

    return "\n".join(lines)


def _format_paper_details_result(data: Dict[str, Any]) -> str:
    """Format paper details result for LLM."""
    papers = data.get("papers", [])
    lines = [f"Paper Details ({len(papers)} found):\n"]

    if not papers:
        lines.append("No papers found matching the given criteria.")
        return "\n".join(lines)

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        year = paper.get("year", "")
        conference = paper.get("conference", "")
        lines.append(f"\n{i}. {title} ({conference} {year})")

        authors = paper.get("authors") or []
        if authors:
            lines.append(f"   Authors: {', '.join(authors)}")

        url = paper.get("url") or ""
        if url:
            lines.append(f"   URL: {url}")

        pdf = paper.get("paper_pdf_url") or ""
        if pdf:
            lines.append(f"   PDF: {pdf}")

        session = paper.get("session") or ""
        room = paper.get("room_name") or ""
        if session or room:
            location = f"{session}" + (f" ({room})" if room else "")
            lines.append(f"   Session: {location}")

        keywords = paper.get("keywords") or ""
        if keywords:
            lines.append(f"   Keywords: {keywords}")

        award = paper.get("award") or ""
        if award:
            lines.append(f"   Award: {award}")

        abstract = paper.get("abstract") or ""
        if abstract:
            snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
            lines.append(f"   Abstract: {snippet}")

    return "\n".join(lines)


def _format_summarize_interesting_papers_result(data: Dict[str, Any]) -> str:
    """Format an interesting-paper summary for the LLM."""
    total = data.get("total_interesting", 0)
    filters = data.get("filters", {})
    lines = [f"Interesting Papers Summary ({total} papers):\n"]

    if filters.get("conferences") or filters.get("years"):
        lines.append(f"Filters: conferences={filters.get('conferences')}, years={filters.get('years')}")

    if data.get("warning"):
        lines.append(f"Note: {data['warning']}")

    if total == 0:
        return "\n".join(lines)

    lines.append(f"Rated papers: {data.get('rated_count', 0)}; average rating: {data.get('average_rating', 0)}/5")

    distribution = data.get("rating_distribution", {})
    if distribution:
        dist_text = ", ".join(f"{rating}★: {count}" for rating, count in distribution.items() if count)
        lines.append(f"Rating distribution: {dist_text}")

    conferences = data.get("conferences", {})
    if conferences:
        lines.append("Conferences: " + ", ".join(f"{conf} ({count})" for conf, count in conferences.items()))

    years = data.get("years", {})
    if years:
        lines.append("Years: " + ", ".join(f"{year} ({count})" for year, count in years.items()))

    keywords = data.get("top_keywords", [])
    if keywords:
        lines.append("Top keywords: " + ", ".join(keywords[:10]))

    search_terms = data.get("search_terms", [])
    if search_terms:
        lines.append(
            "Search themes: "
            + ", ".join(f"{item.get('search_term')} ({item.get('count')})" for item in search_terms[:8])
        )

    sessions = data.get("sessions", [])
    if sessions:
        lines.append(
            "Sessions: " + ", ".join(f"{item.get('session')} ({item.get('count')})" for item in sessions[:8])
        )

    lines.append("\nHighest-rated papers:")
    for paper in data.get("top_rated_papers", [])[:5]:
        lines.append(f"  - {paper.get('title')} ({paper.get('priority')}/5)")

    papers = data.get("papers", [])
    if papers:
        lines.append("\nPaper list:")
        for paper in papers:
            lines.append(
                f"  - {paper.get('title')} — {paper.get('conference')} {paper.get('year')} "
                f"({paper.get('priority')}/5, search term: {paper.get('searchTerm')})"
            )
        if data.get("papers_truncated"):
            lines.append("  (paper list truncated)")

    return "\n".join(lines)


def _format_recommend_related_papers_result(data: Dict[str, Any]) -> str:
    """Format related-paper recommendations for the LLM."""
    papers = data.get("papers", [])
    lines = [f"Recommended Papers ({len(papers)} found):\n"]

    filters = data.get("filters", {})
    if filters.get("conferences") or filters.get("years"):
        lines.append(f"Scope: conferences={filters.get('conferences')}, years={filters.get('years')}")

    if data.get("warning"):
        lines.append(f"Note: {data['warning']}")

    if not papers:
        return "\n".join(lines)

    for index, paper in enumerate(papers, 1):
        score = paper.get("recommendation_score", 0)
        lines.append(
            f"\n{index}. {paper.get('title')} ({paper.get('conference')} {paper.get('year')}) "
            f"— relevance: {score}"
        )
        reasons = paper.get("recommendation_reasons", [])
        if reasons:
            lines.append(f"   Why: {'; '.join(reasons)}")
        abstract = paper.get("abstract_snippet", "")
        if abstract:
            lines.append(f"   Abstract: {abstract}")

    return "\n".join(lines)
