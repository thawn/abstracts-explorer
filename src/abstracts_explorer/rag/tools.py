"""
Pydantic-AI tool wrappers around the MCP tools.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic_ai import RunContext

from abstracts_explorer.mcp import (
    analyze_topic_relevance as mcp_analyze_topic_relevance,
    get_cluster_visualization as mcp_get_cluster_visualization,
    get_conference_topics as mcp_get_conference_topics,
    get_paper_details as mcp_get_paper_details,
    get_topic_evolution as mcp_get_topic_evolution,
    recommend_related_papers as mcp_recommend_related_papers,
    search_papers as mcp_search_papers,
    summarize_interesting_papers as mcp_summarize_interesting_papers,
)
from abstracts_explorer.mcp_tools import abbreviate_result, format_tool_result_for_llm
from abstracts_explorer.rag.deps import RAGDeps

logger = logging.getLogger(__name__)


def _run_rag_tool(
    ctx: RunContext[RAGDeps],
    tool_name: str,
    mcp_func: Callable[..., str],
    kwargs: Dict[str, Any],
) -> str:
    """
    Execute an MCP tool and record its result for post-processing.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    tool_name : str
        Name of the tool as exposed to the LLM.
    mcp_func : callable
        MCP server function to invoke.
    kwargs : dict
        Keyword arguments for *mcp_func*.

    Returns
    -------
    str
        Human-readable tool result for the LLM.
    """
    logger.info("Tool call: %s(%s)", tool_name, kwargs)
    raw = mcp_func(**kwargs)
    logger.info("Tool result: %s → %s", tool_name, abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": tool_name, "raw_result": raw})
    return format_tool_result_for_llm(tool_name, raw)


def _tool_search_papers(
    ctx: RunContext[RAGDeps],
    topic_keywords: str,
    n_results: Optional[int] = None,
    years: Optional[List[int]] = None,
    conference: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Search for papers about a specific topic using semantic search.

    Use this tool to find papers related to a research topic. Returns
    matching papers with titles, abstracts, and relevance scores.

    A conference must be specified. If not provided by the user, the
    currently selected conference from the web UI is used automatically.
    The currently selected year is also applied automatically when not
    explicitly provided.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    topic_keywords : str
        Keywords describing the topic to search for.
    n_results : int, optional
        Number of papers to return (default: 10, or session default).
    years : list of int, optional
        Filter by publication years (e.g. [2024, 2025]).
        When omitted, the default year(s) from the current session are used.
    conference : str, optional
        Filter by conference name (e.g. "NeurIPS", "ICLR").
        When omitted, the default conference from the current session is used.
    limit : int, optional
        Alias for n_results.
    """
    count = n_results if n_results is not None else (limit if limit is not None else (ctx.deps.n_results or 10))
    kwargs: Dict[str, Any] = {"topic_keywords": topic_keywords, "n_results": count}

    # Determine years: explicit param > deps default
    if years is not None:
        kwargs["years"] = years
    elif ctx.deps.years:
        kwargs["years"] = ctx.deps.years

    # Determine conference: explicit param > deps default
    if conference:
        kwargs["conference"] = conference
    elif ctx.deps.conferences:
        kwargs["conference"] = ctx.deps.conferences[0]

    return _run_rag_tool(ctx, "search_papers", mcp_search_papers, kwargs)


def _tool_get_conference_topics(
    ctx: RunContext[RAGDeps],
    conference: Optional[str] = None,
) -> str:
    """Get the main research topics of a conference.

    Returns the key research topics covered at the conference, each with
    a descriptive name, representative keywords, paper count, and example
    paper titles.

    A conference must be specified. If not provided by the user, the
    currently selected conference from the web UI is used automatically.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    conference : str, optional
        Conference name (e.g. "NeurIPS").
        When omitted, the default conference from the current session is used.
    """
    # Determine conferences: explicit param > deps default
    if conference:
        conferences = [conference]
    elif ctx.deps.conferences:
        conferences = ctx.deps.conferences
    else:
        conferences = None

    years = ctx.deps.years if ctx.deps.years else None

    kwargs = {"conferences": conferences, "years": years}
    return _run_rag_tool(ctx, "get_conference_topics", mcp_get_conference_topics, kwargs)


def _tool_get_topic_evolution(
    ctx: RunContext[RAGDeps],
    topic_keywords: str,
    conferences: Optional[list[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> str:
    """Analyze how a topic has evolved over the years.

    Use this tool to understand trends and year-over-year changes
    in a research topic. Returns paper counts and relative percentages
    by year for one or more conferences.

    At least one conference must be specified. If not provided by the user,
    the currently selected conferences from the web UI are used automatically.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    topic_keywords : str
        Keywords describing the topic to analyze (e.g. "transformers attention").
    conferences : list of str, optional
        Conference names to analyze (e.g. ["NeurIPS", "ICLR"]).
        When omitted, the default conferences from the current session are used.
    start_year : int, optional
        Start year for analysis (inclusive).
    end_year : int, optional
        End year for analysis (inclusive).
    """
    kwargs: Dict[str, Any] = {"topic_keywords": topic_keywords}

    # Determine conferences: explicit param > deps default
    if conferences:
        kwargs["conferences"] = conferences
    elif ctx.deps.conferences:
        kwargs["conferences"] = ctx.deps.conferences

    if start_year is not None:
        kwargs["start_year"] = start_year
    if end_year is not None:
        kwargs["end_year"] = end_year

    return _run_rag_tool(ctx, "get_topic_evolution", mcp_get_topic_evolution, kwargs)


def _tool_analyze_topic_relevance(
    ctx: RunContext[RAGDeps],
    topic: str,
    distance_threshold: float = 1.1,
    conference: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> str:
    """Analyze the relevance or popularity of a topic at a conference.

    Use this tool to count how many papers are semantically similar to a
    topic, measuring how prevalent or important a research topic is.
    Returns relevance score, paper counts, and conference/year breakdowns.

    A conference must be specified. If not provided by the user, the
    currently selected conference from the web UI is used automatically.
    The currently selected year is also applied automatically when not
    explicitly provided.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    topic : str
        The topic or research question to analyze.
    distance_threshold : float
        Maximum distance to consider papers relevant (default: 1.1).
    conference : str, optional
        Conference name to analyze (e.g. "NeurIPS", "ICLR").
        When omitted, the default conference from the current session is used.
    years : list of int, optional
        Filter results to specific years.
        When omitted, the default year(s) from the current session are used.
    """
    kwargs: Dict[str, Any] = {"topic": topic, "distance_threshold": distance_threshold}

    # Determine conferences: explicit param > deps default
    if conference:
        kwargs["conferences"] = [conference]
    elif ctx.deps.conferences:
        kwargs["conferences"] = ctx.deps.conferences

    # Determine years: explicit param > deps default
    if years is not None:
        kwargs["years"] = years
    elif ctx.deps.years:
        kwargs["years"] = ctx.deps.years

    return _run_rag_tool(ctx, "analyze_topic_relevance", mcp_analyze_topic_relevance, kwargs)


def _tool_get_cluster_visualization(
    ctx: RunContext[RAGDeps],
    conference: Optional[str] = None,
) -> str:
    """Generate 2D visualization data for pre-computed clustered paper embeddings.

    Use this tool to produce data for plotting papers grouped by topic
    cluster. Returns points with x/y coordinates and cluster assignments.

    A conference must be specified. If not provided by the user, the
    currently selected conference from the web UI is used automatically.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    conference : str, optional
        Conference name to retrieve visualization for (e.g. "NeurIPS").
        When omitted, the default conference from the current session is used.
    """
    # Determine conferences: explicit param > deps default
    if conference:
        conferences = [conference]
    elif ctx.deps.conferences:
        conferences = ctx.deps.conferences
    else:
        conferences = None

    years = ctx.deps.years if ctx.deps.years else None

    kwargs = {"conferences": conferences, "years": years}
    return _run_rag_tool(ctx, "get_cluster_visualization", mcp_get_cluster_visualization, kwargs)


def _tool_get_paper_details(
    ctx: RunContext[RAGDeps],
    title: Optional[str] = None,
    paper_id: Optional[str] = None,
    conference: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 5,
) -> str:
    """Get detailed metadata for specific papers from the database.

    Use this tool when the user asks about: who wrote a paper, paper authors,
    where to find a paper, PDF or poster links, session or room details,
    paper awards, or any other metadata about a specific paper.

    At least one of title or paper_id must be provided. A conference filter
    is applied automatically from the current web UI selection when not
    explicitly provided.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    title : str, optional
        Title or partial title to search for (case-insensitive).
    paper_id : str, optional
        Unique paper identifier (uid or original conference/OpenReview ID).
    conference : str, optional
        Filter by conference name (e.g. "NeurIPS", "ICLR").
        When omitted, the default conference from the current session is used.
    year : int, optional
        Filter by publication year.
        When omitted, the default year from the current session is used.
    limit : int
        Maximum number of papers to return when searching by title (default: 5).
    """
    kwargs: Dict[str, Any] = {"limit": limit}
    if title:
        kwargs["title"] = title
    if paper_id:
        kwargs["paper_id"] = paper_id

    # Determine conference: explicit param > deps default
    if conference:
        kwargs["conference"] = conference
    elif ctx.deps.conferences:
        kwargs["conference"] = ctx.deps.conferences[0]

    # Determine year: explicit param > deps default
    if year is not None:
        kwargs["year"] = year
    elif ctx.deps.years:
        kwargs["year"] = ctx.deps.years[0]

    return _run_rag_tool(ctx, "get_paper_details", mcp_get_paper_details, kwargs)


def _tool_summarize_interesting_papers(
    ctx: RunContext[RAGDeps],
    conference: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> str:
    """Summarize the papers the user marked as interesting.

    Use this tool when the user asks for a summary, overview, or recap of the
    papers they starred, liked, marked, or rated as interesting. The user's
    ratings are provided automatically by the web UI, so do not ask for paper
    IDs.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    conference : str, optional
        Conference filter (e.g. "ICLR" or "ICLR 2026"). When omitted, the
        currently selected conference from the web UI is used automatically.
    years : list of int, optional
        Year filter. When omitted, the currently selected year from the web UI
        is used automatically.
    """
    priorities = ctx.deps.paper_priorities or {}
    if not priorities:
        return (
            "No papers have been marked as interesting yet. " "Rate papers with the star buttons in the web UI first."
        )

    conferences = [conference] if conference else (ctx.deps.conferences or None)
    resolved_years = years if years is not None else (ctx.deps.years or None)

    kwargs: Dict[str, Any] = {"priorities": priorities}
    if conferences:
        kwargs["conference"] = conferences[0]
    if resolved_years:
        kwargs["years"] = resolved_years

    return _run_rag_tool(ctx, "summarize_interesting_papers", mcp_summarize_interesting_papers, kwargs)


def _tool_recommend_related_papers(
    ctx: RunContext[RAGDeps],
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    n_results: int = 10,
) -> str:
    """Recommend further papers based on the user's interesting papers.

    Use this tool when the user asks for suggestions, recommendations, or
    related papers based on the papers they starred, liked, marked, or rated
    as interesting. Already-rated papers are excluded. The user's ratings are
    provided automatically by the web UI, so do not ask for paper IDs.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    conferences : list of str, optional
        Restrict recommendations to these conferences. When omitted, the
        currently selected conferences from the web UI are used automatically.
    years : list of int, optional
        Restrict recommendations to these years. When omitted, the currently
        selected years from the web UI are used automatically.
    n_results : int, optional
        Number of recommendations to return (default: 10).
    """
    priorities = ctx.deps.paper_priorities or {}
    if not priorities:
        return (
            "No papers have been marked as interesting yet. " "Rate papers with the star buttons in the web UI first."
        )

    resolved_conferences = conferences if conferences else (ctx.deps.conferences or None)
    resolved_years = years if years is not None else (ctx.deps.years or None)

    kwargs: Dict[str, Any] = {"priorities": priorities, "n_results": n_results}
    if resolved_conferences:
        kwargs["conferences"] = resolved_conferences
    if resolved_years:
        kwargs["years"] = resolved_years

    return _run_rag_tool(ctx, "recommend_related_papers", mcp_recommend_related_papers, kwargs)
