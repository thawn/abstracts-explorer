"""
Retrieval Augmented Generation (RAG) for conference abstracts.

This module provides RAG functionality using Pydantic AI as the agent framework.
The agent uses tools from the MCP server to search papers, analyze topics, and
understand trends, then generates contextual responses.
"""

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
    get_cluster_topics as mcp_get_cluster_topics,
    get_topic_evolution as mcp_get_topic_evolution,
    analyze_topic_relevance as mcp_analyze_topic_relevance,
    get_cluster_visualization as mcp_get_cluster_visualization,
)
from abstracts_explorer.mcp_tools import format_tool_result_for_llm

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
    """

    tool_results: List[Dict[str, Any]] = field(default_factory=list)


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
    conference : str, optional
        Filter by conference name (e.g. "NeurIPS", "ICLR").
    """
    kwargs: Dict[str, Any] = {"topic_keywords": topic_keywords, "n_results": n_results}
    if years is not None:
        kwargs["years"] = years
    if conference is not None:
        kwargs["conference"] = conference

    raw = mcp_search_papers(**kwargs)
    ctx.deps.tool_results.append({"name": "search_papers", "raw_result": raw})
    return format_tool_result_for_llm("search_papers", raw)


def _tool_get_cluster_topics(ctx: RunContext[RAGDeps]) -> str:
    """Get the main research topics from clustered paper embeddings.

    Use this tool to discover the most frequently mentioned topics
    and research areas across all papers. Returns clusters with
    keywords and statistics.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    """
    raw = mcp_get_cluster_topics()
    ctx.deps.tool_results.append({"name": "get_cluster_topics", "raw_result": raw})
    return format_tool_result_for_llm("get_cluster_topics", raw)


def _tool_get_topic_evolution(
    ctx: RunContext[RAGDeps],
    topic_keywords: str,
    conference: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> str:
    """Analyze how a topic has evolved over the years.

    Use this tool to understand trends and year-over-year changes
    in a research topic. Returns paper counts by year and sample papers.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    topic_keywords : str
        Keywords describing the topic to analyze (e.g. "transformers attention").
    conference : str, optional
        Filter by conference name (e.g. "NeurIPS", "ICLR").
    start_year : int, optional
        Start year for analysis (inclusive).
    end_year : int, optional
        End year for analysis (inclusive).
    """
    kwargs: Dict[str, Any] = {"topic_keywords": topic_keywords}
    if conference is not None:
        kwargs["conference"] = conference
    if start_year is not None:
        kwargs["start_year"] = start_year
    if end_year is not None:
        kwargs["end_year"] = end_year

    raw = mcp_get_topic_evolution(**kwargs)
    ctx.deps.tool_results.append({"name": "get_topic_evolution", "raw_result": raw})
    return format_tool_result_for_llm("get_topic_evolution", raw)


def _tool_analyze_topic_relevance(
    ctx: RunContext[RAGDeps],
    topic: str,
    distance_threshold: float = 1.1,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> str:
    """Analyze the relevance or popularity of a topic at a conference.

    Use this tool to count how many papers are semantically similar to a
    topic, measuring how prevalent or important a research topic is.
    Returns relevance score, paper counts, and conference/year breakdowns.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    topic : str
        The topic or research question to analyze.
    distance_threshold : float
        Maximum distance to consider papers relevant (default: 1.1).
    conferences : list of str, optional
        Filter results to specific conferences.
    years : list of int, optional
        Filter results to specific years.
    """
    kwargs: Dict[str, Any] = {"topic": topic, "distance_threshold": distance_threshold}
    if conferences is not None:
        kwargs["conferences"] = conferences
    if years is not None:
        kwargs["years"] = years

    raw = mcp_analyze_topic_relevance(**kwargs)
    ctx.deps.tool_results.append({"name": "analyze_topic_relevance", "raw_result": raw})
    return format_tool_result_for_llm("analyze_topic_relevance", raw)


def _tool_get_cluster_visualization(ctx: RunContext[RAGDeps]) -> str:
    """Generate 2D visualization data for clustered paper embeddings.

    Use this tool to produce data for plotting papers grouped by topic
    cluster. Returns points with x/y coordinates and cluster assignments.

    Parameters
    ----------
    ctx : RunContext[RAGDeps]
        Agent context with dependencies.
    """
    raw = mcp_get_cluster_visualization()
    ctx.deps.tool_results.append({"name": "get_cluster_visualization", "raw_result": raw})
    return format_tool_result_for_llm("get_cluster_visualization", raw)


# ---------------------------------------------------------------------------
# RAGChat
# ---------------------------------------------------------------------------


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
        ]
        if self.enable_mcp_tools:
            tools.extend(
                [
                    Tool(_tool_get_cluster_topics, takes_ctx=True, name="get_cluster_topics"),
                    Tool(_tool_get_topic_evolution, takes_ctx=True, name="get_topic_evolution"),
                    Tool(_tool_analyze_topic_relevance, takes_ctx=True, name="analyze_topic_relevance"),
                    Tool(_tool_get_cluster_visualization, takes_ctx=True, name="get_cluster_visualization"),
                ]
            )

        # Build instructions
        instructions = (
            "You are an AI assistant helping researchers analyze conference data. "
            "Use the available tools to search for papers, analyze topics, and understand trends. "
            "Present the information in a clear, easy-to-understand format. "
            "When referencing specific papers, cite them using local links: "
            "<a href='#paper-1'>Paper-1</a>, <a href='#paper-2'>Paper-2</a>, etc."
        )

        return Agent(
            ai_model,
            deps_type=RAGDeps,
            tools=tools,
            instructions=instructions,
        )

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
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
            Custom system prompt for the model (overrides default instructions).

        Returns
        -------
        dict
            Dictionary containing:
            - response: str - Generated response
            - papers: list - Retrieved papers used as context
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
            deps = RAGDeps()

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

            # Override instructions if custom system prompt provided
            if system_prompt is not None:
                run_kwargs["instructions"] = system_prompt

            # Run the agent
            result = self.agent.run_sync(question, **run_kwargs)

            # Update message history for next turn
            self._message_history = result.all_messages()

            # Extract papers from tool results
            papers = self._extract_papers(deps.tool_results)

            # Determine which tools were executed
            tools_executed = [tr["name"] for tr in deps.tool_results]

            return {
                "response": result.output,
                "papers": papers,
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
