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
from typing import Callable, Dict, List, Any, Optional

from .mcp_server import (
    get_cluster_topics,
    get_topic_evolution,
    search_papers,
    get_cluster_visualization,
    analyze_topic_relevance,
)

logger = logging.getLogger(__name__)


class MCPToolsError(Exception):
    """Exception raised for MCP tools-related errors."""

    pass


def _normalize_search_papers_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise argument shapes produced by LLMs for the ``search_papers`` tool.

    LLMs occasionally produce slightly wrong argument shapes, e.g. a singular
    ``"year"`` key instead of ``"years"``, or a list for scalar string fields.
    This function corrects those mismatches so the downstream function call
    always receives the expected types.

    Normalizations applied:
    * ``year`` (int or list) → ``years`` (list of int)
    * ``topic_keywords`` as a list → joined string
    * ``conference`` as a list → first element string
    * ``conferences`` (list, wrong field name) → ``conference`` (str, first element)

    Parameters
    ----------
    arguments : dict
        Raw arguments dict coming from the LLM / ``execute_mcp_tool`` caller.

    Returns
    -------
    dict
        A new dict with normalized argument values.
    """
    args = dict(arguments)

    # Normalize 'year' → 'years' (LLMs often use singular form)
    if "year" in args and "years" not in args:
        year_val = args.pop("year")
        if isinstance(year_val, list):
            args["years"] = year_val
        else:
            args["years"] = [year_val]
    elif "year" in args:
        args.pop("year")  # 'years' already present; drop the duplicate

    # Normalize topic_keywords: list → space-joined string
    if "topic_keywords" in args and isinstance(args["topic_keywords"], list):
        args["topic_keywords"] = " ".join(str(k) for k in args["topic_keywords"])

    # Normalize conference: list → first element string
    if "conference" in args and isinstance(args["conference"], list):
        args["conference"] = args["conference"][0] if args["conference"] else None

    # Normalize 'conferences' (wrong field name) → 'conference' if not already set
    if "conferences" in args and "conference" not in args:
        conferences_val = args.pop("conferences")
        if isinstance(conferences_val, list) and conferences_val:
            args["conference"] = conferences_val[0]
        elif isinstance(conferences_val, str):
            args["conference"] = conferences_val
    elif "conferences" in args:
        args.pop("conferences")  # 'conference' already present; drop duplicate

    return args


def _normalize_get_topic_evolution_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise argument shapes produced by LLMs for the ``get_topic_evolution`` tool.

    Parameters
    ----------
    arguments : dict
        Raw arguments dict from the LLM.

    Returns
    -------
    dict
        A new dict with normalized argument values.
    """
    args = dict(arguments)

    # Normalize topic_keywords: list → space-joined string
    if "topic_keywords" in args and isinstance(args["topic_keywords"], list):
        args["topic_keywords"] = " ".join(str(k) for k in args["topic_keywords"])

    # Normalize conference: list → first element string
    if "conference" in args and isinstance(args["conference"], list):
        args["conference"] = args["conference"][0] if args["conference"] else None

    # Normalize start_year / end_year: list → first element int
    for key in ("start_year", "end_year"):
        if key in args and isinstance(args[key], list):
            args[key] = args[key][0] if args[key] else None

    return args


def _normalize_analyze_topic_relevance_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise argument shapes produced by LLMs for the ``analyze_topic_relevance`` tool.

    Parameters
    ----------
    arguments : dict
        Raw arguments dict from the LLM.

    Returns
    -------
    dict
        A new dict with normalized argument values.
    """
    args = dict(arguments)

    # Normalize topic: list → space-joined string
    if "topic" in args and isinstance(args["topic"], list):
        args["topic"] = " ".join(str(k) for k in args["topic"])

    # Normalize conference → conferences if wrong field name used
    if "conference" in args and "conferences" not in args:
        conf_val = args.pop("conference")
        if isinstance(conf_val, str):
            args["conferences"] = [conf_val]
        elif isinstance(conf_val, list):
            args["conferences"] = conf_val

    return args


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
                "research themes at a conference."
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
                        "description": "Filter by specific conferences (e.g., ['NeurIPS', 'ICLR'])",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter by specific years (e.g., [2024, 2025])",
                    },
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_topics",
            "description": (
                "Analyze clustered paper embeddings to identify the most frequently mentioned topics. "
                "Use this tool when the user asks about: overall themes, main topics, research areas, "
                "or wants to understand what topics are covered in the conference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {"type": "integer", "description": "Number of clusters to create (default: 8)"},
                    "reduction_method": {
                        "type": "string",
                        "enum": ["pca", "tsne"],
                        "description": "Dimensionality reduction method (default: 'pca')",
                    },
                    "clustering_method": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan", "agglomerative"],
                        "description": "Clustering algorithm (default: 'kmeans')",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional, uses config default)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_evolution",
            "description": (
                "Analyze how specific topics have evolved over the years. "
                "Use this tool when the user asks about: trends over time, historical development, "
                "how a topic has changed, or evolution of research areas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_keywords": {
                        "type": "string",
                        "description": "Keywords describing the topic (e.g., 'transformers attention', 'reinforcement learning')",
                    },
                    "conference": {
                        "type": "string",
                        "description": "Filter by conference name (e.g., 'neurips', 'iclr')",
                    },
                    "start_year": {"type": "integer", "description": "Start year for analysis (inclusive)"},
                    "end_year": {"type": "integer", "description": "End year for analysis (inclusive)"},
                    "where": {"type": "object", "description": "Custom ChromaDB WHERE clause for advanced filtering"},
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic_keywords"],
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
                "specific work, or wants to find papers related to a particular area. Can filter by specific years or search all years."
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
                    "conference": {"type": "string", "description": "Filter by conference name"},
                    "where": {"type": "object", "description": "Custom ChromaDB WHERE clause for filtering"},
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
                },
                "required": ["topic_keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_visualization",
            "description": (
                "Generate visualization data for clustered embeddings. "
                "Use this tool when the user asks for: a visual representation, graphical view, "
                "or wants to see clusters displayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {"type": "integer", "description": "Number of clusters (default: 8)"},
                    "reduction_method": {
                        "type": "string",
                        "enum": ["pca", "tsne"],
                        "description": "Reduction method (default: 'tsne')",
                    },
                    "clustering_method": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan", "agglomerative"],
                        "description": "Clustering method (default: 'kmeans')",
                    },
                    "n_components": {
                        "type": "integer",
                        "enum": [2, 3],
                        "description": "Number of dimensions: 2 or 3 (default: 2)",
                    },
                    "output_path": {"type": "string", "description": "Path to save visualization JSON (optional)"},
                    "collection_name": {"type": "string", "description": "Name of ChromaDB collection (optional)"},
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
    logger.info(f"Executing MCP tool: {tool_name} with arguments: {arguments}")

    try:
        if tool_name == "analyze_topic_relevance":
            args = _filter_unknown_kwargs(analyze_topic_relevance, _normalize_analyze_topic_relevance_args(arguments))
            return analyze_topic_relevance(**args)
        elif tool_name == "get_cluster_topics":
            args = _filter_unknown_kwargs(get_cluster_topics, arguments)
            return get_cluster_topics(**args)
        elif tool_name == "get_topic_evolution":
            args = _filter_unknown_kwargs(get_topic_evolution, _normalize_get_topic_evolution_args(arguments))
            return get_topic_evolution(**args)
        elif tool_name == "search_papers":
            # Normalize argument names/types — LLMs may send 'query', 'year', or list values
            args = _normalize_search_papers_args(arguments)
            if "query" in args and "topic_keywords" not in args:
                args["topic_keywords"] = args.pop("query")
            args = _filter_unknown_kwargs(search_papers, args)
            return search_papers(**args)
        elif tool_name == "get_cluster_visualization":
            args = _filter_unknown_kwargs(get_cluster_visualization, arguments)
            return get_cluster_visualization(**args)
        else:
            # Return error JSON for unknown tools
            error_result = {"error": f"Unknown MCP tool: {tool_name}"}
            return json.dumps(error_result, indent=2)

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
            for key in ("start_year", "end_year"):
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
        elif tool_name == "get_cluster_topics":
            return _format_cluster_topics_result(result_data)
        elif tool_name == "get_topic_evolution":
            return _format_topic_evolution_result(result_data)
        elif tool_name == "search_papers":
            return _format_search_papers_result(result_data)
        elif tool_name == "get_cluster_visualization":
            return _format_visualization_result(result_data)
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
    distance = data.get("distance_threshold", 0)
    relevance = data.get("relevance_score", 0)

    lines.append(f"Papers found: {total} within distance {distance}")
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


def _format_cluster_topics_result(data: Dict[str, Any]) -> str:
    """Format cluster topics result for LLM."""
    lines = ["Cluster Analysis Results:\n"]

    stats = data.get("statistics", {})
    lines.append(f"Found {stats.get('n_clusters', 0)} clusters covering {stats.get('total_papers', 0)} papers.\n")

    clusters = data.get("clusters", [])
    for cluster in clusters[:10]:  # Limit to top 10 clusters
        cluster_id = cluster.get("cluster_id")
        paper_count = cluster.get("paper_count", 0)
        keywords = cluster.get("keywords", [])[:5]  # Top 5 keywords

        lines.append(f"\nCluster {cluster_id} ({paper_count} papers):")
        if keywords:
            keyword_strs = [f"{kw['keyword']} ({kw['count']})" for kw in keywords]
            lines.append(f"  Top keywords: {', '.join(keyword_strs)}")

    return "\n".join(lines)


def _format_topic_evolution_result(data: Dict[str, Any]) -> str:
    """Format topic evolution result for LLM."""
    lines = [f"Topic Evolution Analysis for '{data.get('topic', 'unknown')}':\n"]

    year_counts = data.get("year_counts", {})
    if year_counts:
        lines.append("Papers per year:")
        for year, count in sorted(year_counts.items()):
            lines.append(f"  {year}: {count} papers")

    total = data.get("total_papers", 0)
    lines.append(f"\nTotal papers found: {total}")

    return "\n".join(lines)


def _format_search_papers_result(data: Dict[str, Any]) -> str:
    """Format search papers result for LLM."""
    lines = [f"Search Results for '{data.get('topic', 'unknown')}':\n"]

    papers = data.get("papers", [])
    years_filter = data.get("years_filter")
    if years_filter:
        lines.append(f"Filtered by years: {years_filter}")
    lines.append(f"Found {len(papers)} papers:\n")

    for i, paper in enumerate(papers[:5], 1):  # Top 5 papers
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
