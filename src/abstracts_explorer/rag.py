"""
Retrieval Augmented Generation (RAG) for conference abstracts.

This module provides RAG functionality using Pydantic AI as the agent framework.
The agent uses tools from the MCP server to search papers, analyze topics, and
understand trends, then generates contextual responses.
"""

from datetime import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from abstracts_explorer.config import get_config
from abstracts_explorer.mcp_server import (
    search_papers as mcp_search_papers,
    get_conference_topics as mcp_get_conference_topics,
    get_topic_evolution as mcp_get_topic_evolution,
    analyze_topic_relevance as mcp_analyze_topic_relevance,
    get_cluster_visualization as mcp_get_cluster_visualization,
    get_paper_details as mcp_get_paper_details,
)
from abstracts_explorer.mcp_tools import format_tool_result_for_llm, _abbreviate_result

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for RAG-related errors."""

    pass


@dataclass
class RAGDeps:
    """
    Dependencies for RAG agent tools.

    Carries shared state for capturing tool results during agent runs,
    allowing post-processing (e.g. paper extraction) after the agent completes.

    Attributes
    ----------
    tool_results : list of dict
        Accumulated tool call results from the current agent run.
        Each entry has 'name' (tool name) and 'raw_result' (JSON string).
    conferences : list of str
        Default conferences for MCP clustering tools (from web UI selection).
    years : list of int
        Default years for MCP clustering tools (from web UI selection).
    """

    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    conferences: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool wrapper functions
# ---------------------------------------------------------------------------
# These thin wrappers call the MCP server functions and capture the raw result
# in the deps for later paper extraction.  They return a human-readable
# summary that goes back to the LLM as tool output.
# ---------------------------------------------------------------------------


def _tool_search_papers(
    ctx: RunContext[RAGDeps],
    topic_keywords: str,
    n_results: int = 5,
    years: Optional[List[int]] = None,
    conference: Optional[str] = None,
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
    n_results : int
        Number of papers to return (default: 5).
    years : list of int, optional
        Filter by publication years (e.g. [2024, 2025]).
        When omitted, the default year(s) from the current session are used.
    conference : str, optional
        Filter by conference name (e.g. "NeurIPS", "ICLR").
        When omitted, the default conference from the current session is used.
    """
    kwargs: Dict[str, Any] = {"topic_keywords": topic_keywords, "n_results": n_results}

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

    logger.info("Tool call: search_papers(%s)", kwargs)
    raw = mcp_search_papers(**kwargs)
    logger.info("Tool result: search_papers → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "search_papers", "raw_result": raw})
    return format_tool_result_for_llm("search_papers", raw)


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

    logger.info("Tool call: get_conference_topics(conferences=%s, years=%s)", conferences, years)
    raw = mcp_get_conference_topics(conferences=conferences, years=years)
    logger.info("Tool result: get_conference_topics → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "get_conference_topics", "raw_result": raw})
    return format_tool_result_for_llm("get_conference_topics", raw)


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

    logger.info("Tool call: get_topic_evolution(%s)", kwargs)
    raw = mcp_get_topic_evolution(**kwargs)
    logger.info("Tool result: get_topic_evolution → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "get_topic_evolution", "raw_result": raw})
    return format_tool_result_for_llm("get_topic_evolution", raw)


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

    logger.info("Tool call: analyze_topic_relevance(%s)", kwargs)
    raw = mcp_analyze_topic_relevance(**kwargs)
    logger.info("Tool result: analyze_topic_relevance → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "analyze_topic_relevance", "raw_result": raw})
    return format_tool_result_for_llm("analyze_topic_relevance", raw)


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

    logger.info("Tool call: get_cluster_visualization(conferences=%s, years=%s)", conferences, years)
    raw = mcp_get_cluster_visualization(conferences=conferences, years=years)
    logger.info("Tool result: get_cluster_visualization → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "get_cluster_visualization", "raw_result": raw})
    return format_tool_result_for_llm("get_cluster_visualization", raw)


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

    logger.info("Tool call: get_paper_details(%s)", kwargs)
    raw = mcp_get_paper_details(**kwargs)
    logger.info("Tool result: get_paper_details → %s", _abbreviate_result(raw))
    ctx.deps.tool_results.append({"name": "get_paper_details", "raw_result": raw})
    return format_tool_result_for_llm("get_paper_details", raw)


class RAGChat:
    """
    RAG chat interface for querying conference papers.

    Uses Pydantic AI as the agent framework to orchestrate tool calling and
    response generation via an OpenAI-compatible language model API.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Manager for embeddings and vector search.
    database : DatabaseManager
        Database instance for querying paper details. REQUIRED.
    lm_studio_url : str, optional
        URL for OpenAI-compatible API, by default from config.
    model : str, optional
        Name of the language model, by default from config.
    max_context_papers : int, optional
        Maximum number of papers to include in context, by default from config.
    temperature : float, optional
        Sampling temperature for generation, by default from config.
    enable_mcp_tools : bool, optional
        Enable MCP clustering tools for topic analysis, by default True.

    Examples
    --------
    >>> from abstracts_explorer.embeddings import EmbeddingsManager
    >>> from abstracts_explorer.database import DatabaseManager
    >>> em = EmbeddingsManager()
    >>> em.connect()
    >>> db = DatabaseManager()
    >>> db.connect()
    >>> chat = RAGChat(em, db)
    >>>
    >>> # Ask about specific papers
    >>> response = chat.query("What are the latest advances in deep learning?")
    >>> print(response)
    >>>
    >>> # Ask about conference topics (uses MCP tools automatically)
    >>> response = chat.query("What are the main research topics at NeurIPS?")
    >>> print(response)
    """

    def __init__(
        self,
        embeddings_manager,
        database,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        max_context_papers: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_mcp_tools: bool = True,
    ):
        if embeddings_manager is None:
            raise RAGError("embeddings_manager is required.")
        if database is None:
            raise RAGError("database is required. RAGChat cannot operate without database access.")

        config = get_config()
        self.embeddings_manager = embeddings_manager
        self.database = database
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.model = model or config.chat_model
        self.max_context_papers = max_context_papers or config.max_context_papers
        self.temperature = temperature or config.chat_temperature
        self.enable_mcp_tools = enable_mcp_tools

        # Message history for multi-turn conversation
        self._message_history: List[ModelMessage] = []

        # Build the Pydantic AI agent
        self.agent = self._build_agent()

    def _build_base_instructions(self) -> str:
        """
        Build the base system instructions for the RAG agent.

        Returns
        -------
        str
            Base instruction string without conference-specific context.
        """
        return (
            "You are an AI assistant helping researchers analyze conference data. "
            "Use the available tools to search for papers, analyze topics, and understand trends. "
            "Present the information in a clear, easy-to-understand format. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "When referencing specific papers, cite them using local links: "
            "<a href='#paper-1'>Paper-1</a>, <a href='#paper-2'>Paper-2</a>, etc."
        )

    def _build_instructions(
        self,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        available_conferences: Optional[List[str]] = None,
    ) -> str:
        """
        Build context-aware system instructions including conference information.

        Parameters
        ----------
        conferences : list of str, optional
            Currently selected conference(s) from the web UI.
        years : list of int, optional
            Currently selected year(s) from the web UI.
        available_conferences : list of str, optional
            All available conference names from registered plugins.

        Returns
        -------
        str
            Full instruction string with conference context appended when available.
        """
        instructions = self._build_base_instructions()

        context_parts = []
        if conferences and years:
            conf_str = ", ".join(conferences)
            year_str = ", ".join(str(y) for y in years)
            context_parts.append(
                f"Unless the user specifies otherwise, assume the default conference is "
                f"{conf_str} and the default year is {year_str}."
            )
        elif conferences:
            conf_str = ", ".join(conferences)
            context_parts.append(f"Unless the user specifies otherwise, assume the default conference is {conf_str}.")
        elif years:
            year_str = ", ".join(str(y) for y in years)
            context_parts.append(f"Unless the user specifies otherwise, assume the default year is {year_str}.")

        if available_conferences:
            avail_str = ", ".join(available_conferences)
            context_parts.append(f"The available conferences are: {avail_str}.")

        if context_parts:
            instructions += " " + " ".join(context_parts)

        return instructions

    def _build_agent(self) -> Agent[RAGDeps, str]:
        """
        Build the Pydantic AI agent with tools and model configuration.

        Returns
        -------
        Agent[RAGDeps, str]
            Configured Pydantic AI agent.
        """
        config = get_config()

        # Create OpenAI-compatible model
        provider = OpenAIProvider(
            base_url=f"{self.lm_studio_url}/v1",
            api_key=config.llm_backend_auth_token or "lm-studio-local",
        )
        ai_model = OpenAIChatModel(self.model, provider=provider)

        # Build tool list
        tools: List[Tool[RAGDeps]] = [
            Tool(_tool_search_papers, takes_ctx=True, name="search_papers"),
            Tool(_tool_get_paper_details, takes_ctx=True, name="get_paper_details"),
        ]
        if self.enable_mcp_tools:
            tools.extend(
                [
                    Tool(_tool_get_conference_topics, takes_ctx=True, name="get_conference_topics"),
                    Tool(_tool_get_topic_evolution, takes_ctx=True, name="get_topic_evolution"),
                    Tool(_tool_analyze_topic_relevance, takes_ctx=True, name="analyze_topic_relevance"),
                    Tool(_tool_get_cluster_visualization, takes_ctx=True, name="get_cluster_visualization"),
                ]
            )

        return Agent(
            ai_model,
            deps_type=RAGDeps,
            tools=tools,
            instructions=self._build_base_instructions(),
        )

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        available_conferences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Parameters
        ----------
        question : str
            User's question.
        n_results : int, optional
            Number of papers to retrieve for context.
        metadata_filter : dict, optional
            Metadata filter for paper search (currently unused with Pydantic AI agent).
        system_prompt : str, optional
            Custom system prompt for the model (overrides default instructions including
            conference context).
        conferences : list of str, optional
            Default conferences for MCP clustering tools.  Typically set from
            the web UI conference selector so that clustering tools use cached
            results for the selected conference.  Also included in the system prompt
            so the LLM knows the current conference context.
        years : list of int, optional
            Default years for MCP clustering tools.  Typically set from the
            web UI year selector.  Also included in the system prompt so the LLM
            knows the current year context.
        available_conferences : list of str, optional
            All conference names available in the system (from registered plugins).
            Included in the system prompt so the LLM can inform the user about
            which conferences can be queried.

        Returns
        -------
        dict
            Dictionary containing:
            - response: str - Generated response
            - papers: list - Retrieved papers used as context
            - visualizations: list - Visualization data from tools
            - metadata: dict - Additional metadata

        Raises
        ------
        RAGError
            If query fails.

        Examples
        --------
        >>> response = chat.query("What is attention mechanism?")
        >>> print(response["response"])
        >>> print(f"Based on {len(response['papers'])} papers")
        """
        try:
            if n_results is None:
                n_results = self.max_context_papers

            # Fresh deps for each query to capture tool results
            deps = RAGDeps(
                conferences=conferences or [],
                years=years or [],
            )

            # Model settings
            settings = ModelSettings(
                temperature=self.temperature,
                max_tokens=1000,
            )

            # Prepare run kwargs
            run_kwargs: Dict[str, Any] = {
                "deps": deps,
                "model_settings": settings,
            }

            # Pass message history for conversation context
            if self._message_history:
                run_kwargs["message_history"] = self._message_history

            # Set per-run instructions: use explicit system_prompt if provided,
            # otherwise build context-aware instructions with conference information
            if system_prompt is not None:
                run_kwargs["instructions"] = system_prompt
            else:
                run_kwargs["instructions"] = self._build_instructions(
                    conferences=conferences,
                    years=years,
                    available_conferences=available_conferences,
                )

            # Run the agent
            result = self.agent.run_sync(question, **run_kwargs)

            # Update message history for next turn
            self._message_history = result.all_messages()

            # Extract papers from tool results
            papers = self._extract_papers(deps.tool_results)

            # Extract visualization data from tool results
            visualizations = self._extract_visualizations(deps.tool_results)

            # Determine which tools were executed
            tools_executed = [tr["name"] for tr in deps.tool_results]

            return {
                "response": result.output,
                "papers": papers,
                "visualizations": visualizations,
                "metadata": {
                    "n_papers": len(papers),
                    "model": self.model,
                    "used_tools": len(tools_executed) > 0,
                    "tools_executed": tools_executed,
                },
            }

        except Exception as e:
            raise RAGError(f"Query failed: {str(e)}") from e

    def chat(
        self,
        message: str,
        use_context: bool = True,
        n_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Continue conversation with context awareness.

        Parameters
        ----------
        message : str
            User's message.
        use_context : bool, optional
            Whether to retrieve papers as context, by default True.
        n_results : int, optional
            Number of papers to retrieve.

        Returns
        -------
        dict
            Dictionary containing response and metadata.

        Examples
        --------
        >>> response = chat.chat("Tell me more about transformers")
        >>> print(response["response"])
        """
        if use_context:
            return self.query(message, n_results=n_results)
        else:
            # Use only conversation history without tool calls
            try:
                settings = ModelSettings(
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                run_kwargs: Dict[str, Any] = {
                    "deps": RAGDeps(),
                    "model_settings": settings,
                }
                if self._message_history:
                    run_kwargs["message_history"] = self._message_history

                result = self.agent.run_sync(message, **run_kwargs)
                self._message_history = result.all_messages()

                return {
                    "response": result.output,
                    "papers": [],
                    "metadata": {"n_papers": 0, "model": self.model},
                }
            except Exception as e:
                raise RAGError(f"Chat failed: {str(e)}") from e

    def reset_conversation(self):
        """
        Reset conversation history.

        Examples
        --------
        >>> chat.reset_conversation()
        """
        self._message_history = []
        logger.info("Conversation history reset")

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as a list of role/content dicts.

        This property converts Pydantic AI message objects to simple
        dicts for backwards compatibility with code that accesses
        ``conversation_history`` directly.

        Returns
        -------
        list of dict
            Messages in ``{"role": "user"|"assistant", "content": "..."}`` format.
        """
        history: List[Dict[str, str]] = []
        for msg in self._message_history:
            kind = getattr(msg, "kind", None)
            if kind == "request":
                for part in msg.parts:
                    part_kind = getattr(part, "part_kind", None)
                    if part_kind == "user-prompt":
                        history.append({"role": "user", "content": part.content})
            elif kind == "response":
                for part in msg.parts:
                    part_kind = getattr(part, "part_kind", None)
                    if part_kind == "text":
                        history.append({"role": "assistant", "content": part.content})
        return history

    def export_conversation(self, output_path: Path) -> None:
        """
        Export conversation history to JSON file.

        Parameters
        ----------
        output_path : Path
            Path to output JSON file.

        Examples
        --------
        >>> chat.export_conversation("conversation.json")
        """
        output_path = Path(output_path)
        history = self.conversation_history
        with open(output_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Conversation exported to {output_path}")

    @staticmethod
    def _extract_papers(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract papers from tool results.

        Parameters
        ----------
        tool_results : list of dict
            Tool results with 'name' and 'raw_result' keys.

        Returns
        -------
        list of dict
            Extracted papers.
        """
        papers: List[Dict[str, Any]] = []
        for tr in tool_results:
            if tr["name"] in ("search_papers", "analyze_topic_relevance"):
                try:
                    result_json = json.loads(tr["raw_result"])
                    if "papers" in result_json:
                        papers.extend(result_json.get("papers", []))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse papers from {tr['name']} result")
        return papers

    @staticmethod
    def _extract_visualizations(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract visualization data from tool results.

        Converts raw MCP tool output from ``get_topic_evolution`` and
        ``get_cluster_visualization`` into lightweight chart descriptors
        that the frontend can render with Plotly.

        Parameters
        ----------
        tool_results : list of dict
            Tool results with 'name' and 'raw_result' keys.

        Returns
        -------
        list of dict
            Visualization descriptors.  Each dict contains a ``type`` key
            (``"topic_evolution"`` or ``"cluster_visualization"``) and the
            data needed to draw the chart.
        """
        visualizations: List[Dict[str, Any]] = []
        topic_evolution_viz: Optional[Dict[str, Any]] = None
        for tr in tool_results:
            try:
                data = json.loads(tr["raw_result"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            if not isinstance(data, dict):
                continue

            if "error" in data:
                continue

            if tr["name"] == "get_topic_evolution":
                conference_data = data.get("conference_data", {})
                topic_name = data.get("topic", "")
                if conference_data and topic_name:
                    # Merge into existing topic_evolution visualization if one exists
                    if topic_evolution_viz is not None:
                        topic_evolution_viz["topics"].append(topic_name)
                        topic_evolution_viz["conference_data"][topic_name] = conference_data
                    else:
                        topic_evolution_viz = {
                            "type": "topic_evolution",
                            "topics": [topic_name],
                            "conference_data": {topic_name: conference_data},
                        }
                        visualizations.append(topic_evolution_viz)

            elif tr["name"] == "get_cluster_visualization":
                points = data.get("points", [])
                if points:
                    visualizations.append(
                        {
                            "type": "cluster_visualization",
                            "points": points,
                            "statistics": data.get("statistics", {}),
                        }
                    )

        return visualizations
