"""
RAG agent construction and query execution.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx2
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from abstracts_explorer.config import get_config
from abstracts_explorer.rag.deps import RAGDeps
from abstracts_explorer.rag.errors import RAGError
from abstracts_explorer.rag.tools import (
    _tool_analyze_topic_relevance,
    _tool_get_cluster_visualization,
    _tool_get_conference_topics,
    _tool_get_paper_details,
    _tool_get_topic_evolution,
    _tool_recommend_related_papers,
    _tool_search_papers,
    _tool_summarize_interesting_papers,
)
from abstracts_explorer.utils.time_utils import format_timestamp

logger = logging.getLogger(__name__)


class AgentMixin:
    """Build the Pydantic-AI agent and run queries."""

    backend_url: str
    model: str
    enable_mcp_tools: bool
    _message_history: List[Any]
    _extract_papers: Any
    _extract_visualizations: Any
    _update_message_history_instructions: Any

    def __init__(
        self,
        embeddings_manager,
        database,
        backend_url: Optional[str] = None,
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
        self.backend_url = (backend_url or config.llm_backend_url).rstrip("/")
        self.model = model or config.chat_model
        self.max_context_papers = max_context_papers or config.max_context_papers
        self.temperature = temperature or config.chat_temperature
        self.max_tokens = config.chat_max_tokens
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
            "When the user asks about papers they marked, starred, liked, or rated as interesting, "
            "use the summarize_interesting_papers or recommend_related_papers tools; their ratings "
            "are provided automatically by the web UI, so do not ask for paper IDs. "
            f"Today's date is {format_timestamp(datetime.now(), '%Y-%m-%d')}. "
            "When referencing specific papers, cite them using local markdown links: "
            "[Paper-1](#paper-1), [Paper-2](#paper-2), etc."
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
        requests_per_minute = config.requests_per_minute

        # Build rate-limited async HTTP client when RPM > 0.
        http_client: Optional[httpx2.AsyncClient] = None
        if requests_per_minute > 0:
            from abstracts_explorer.rate_limiter import RateLimitedTransport

            transport = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=requests_per_minute)
            http_client = httpx2.AsyncClient(transport=transport)

        # Create OpenAI-compatible model
        provider = OpenAIProvider(
            base_url=f"{self.backend_url}/v1",
            api_key=config.llm_backend_auth_token or "lm-studio-local",
            http_client=http_client,
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
                    Tool(
                        _tool_summarize_interesting_papers,
                        takes_ctx=True,
                        name="summarize_interesting_papers",
                    ),
                    Tool(
                        _tool_recommend_related_papers,
                        takes_ctx=True,
                        name="recommend_related_papers",
                    ),
                ]
            )

        return Agent(
            ai_model,
            deps_type=RAGDeps,
            tools=tools,
        )

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        system_prompt: Optional[str] = None,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        available_conferences: Optional[List[str]] = None,
        paper_priorities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Parameters
        ----------
        question : str
            User's question.
        n_results : int, optional
            Number of papers to retrieve for context.
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
        paper_priorities : dict, optional
            Paper ratings from the web UI (``localStorage.paperPriorities``).
            Used by the interesting-paper summary and recommendation tools.

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
                n_results=n_results,
                paper_priorities=paper_priorities or {},
            )

            # Model settings
            settings = OpenAIChatModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_reasoning_effort="low",
            )

            # Build instructions first so we can update message history
            if system_prompt is not None:
                instructions = system_prompt
            else:
                instructions = self._build_instructions(
                    conferences=conferences,
                    years=years,
                    available_conferences=available_conferences,
                )

            # Prepare run kwargs
            run_kwargs: Dict[str, Any] = {
                "deps": deps,
                "model_settings": settings,
                "instructions": instructions,
            }

            # Pass message history for conversation context, updating stale
            # system prompt in the first request to reflect current conference.
            # Pydantic AI doesn't re-add system prompts on continuation turns,
            # so the first ModelRequest in history may contain outdated
            # conference context from a prior turn.
            if self._message_history:
                self._update_message_history_instructions(instructions)
                run_kwargs["message_history"] = self._message_history

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
