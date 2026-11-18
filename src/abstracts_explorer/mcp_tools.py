"""
MCP Tools Integration for RAG Chat
==================================

This module provides integration between MCP clustering tools and the RAG chat system.
It normalizes tool arguments to absorb common LLM output quirks and formats tool
execution results for LLM consumption.

The integration allows the LLM to automatically decide when to use clustering tools
to answer questions about conference topics, trends, and developments.
"""

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
from abstracts_explorer.utils.paper_presentation import truncate_abstract

logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search_papers": "Search for papers on a specific topic. Returns relevant papers from the database.",
    "get_conference_topics": "Get the main research topics of a conference.",
    "analyze_topic_relevance": (
        "Analyze the relevance and popularity of a research topic by counting "
        "papers within a specified distance in embedding space."
    ),
    "get_topic_evolution": "Analyze how a specific topic has evolved over the years.",
    "get_cluster_visualization": "Retrieve pre-computed visualization data for clustered embeddings.",
    "get_paper_details": "Get detailed metadata for one or more papers by title or paper ID.",
    "summarize_interesting_papers": "Summarize the papers the user marked as interesting.",
    "recommend_related_papers": "Recommend further papers based on the user's interesting papers.",
}

TOOL_ARGUMENT_EXAMPLES: Dict[str, str] = {
    "search_papers": '{"topic_keywords": "deep learning", "conference": "NeurIPS", "n_results": 5}',
    "get_conference_topics": '{"conferences": ["NeurIPS"], "years": [2025]}',
    "analyze_topic_relevance": '{"topic": "transformers", "conferences": ["NeurIPS"], "distance_threshold": 1.1}',
    "get_topic_evolution": '{"topic_keywords": "reinforcement learning", "conferences": ["NeurIPS"], "start_year": 2020}',
    "get_cluster_visualization": '{"conferences": ["NeurIPS"], "years": [2025]}',
    "get_paper_details": '{"title": "attention is all you need"}',
    "summarize_interesting_papers": '{"conference": "ICLR", "years": [2026]}',
    "recommend_related_papers": '{"conferences": ["ICLR"], "years": [2026], "n_results": 10}',
}


def abbreviate_result(text: str, max_length: int = 200) -> str:
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
    NormRule("years", ("year",), _to_int_list_or_none),
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
    NormRule("start_year", (), _first_int_or_none),
    NormRule("end_year", (), _first_int_or_none),
]

GET_CONFERENCE_TOPICS_RULES = [
    NormRule("conferences", ("conference",), _to_conference_list),
    NormRule("years", ("year",), _to_int_list_or_none),
]

GET_CLUSTER_VISUALIZATION_RULES = [
    NormRule("conferences", ("conference",), _to_conference_list),
    NormRule("years", ("year",), _to_int_list_or_none),
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
            matched_alias = next((name for name in rule.aliases if name in args), None)
            if matched_alias is None:
                continue
            value = args.pop(matched_alias)
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
        Tool execution result (JSON string). Errors (including unknown tool
        names) are returned as ``{"error": ...}`` JSON strings rather than
        raised, so callers always receive a string.
    """
    logger.info("Executing MCP tool: %s with argument keys: %s", tool_name, sorted(arguments))

    try:
        if tool_name == "analyze_topic_relevance":
            result = _execute_tool(analyze_topic_relevance, arguments, ANALYZE_TOPIC_RELEVANCE_RULES)
        elif tool_name == "get_conference_topics":
            result = _execute_tool(get_conference_topics, arguments, GET_CONFERENCE_TOPICS_RULES)
        elif tool_name == "get_topic_evolution":
            result = _execute_tool(get_topic_evolution, arguments, GET_TOPIC_EVOLUTION_RULES)
        elif tool_name == "search_papers":
            result = _execute_tool(search_papers, arguments, SEARCH_PAPERS_RULES)
        elif tool_name == "get_cluster_visualization":
            result = _execute_tool(get_cluster_visualization, arguments, GET_CLUSTER_VISUALIZATION_RULES)
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

        logger.info("MCP tool %s returned: %s", tool_name, abbreviate_result(result))
        return result

    except Exception as e:
        logger.error(f"MCP tool execution failed: {str(e)}")
        error_result = {"error": f"Tool execution failed: {str(e)}"}
        return json.dumps(error_result, indent=2)


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
            snippet = truncate_abstract(abstract, 150)
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
            snippet = truncate_abstract(abstract, 200)
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
