"""
Retrieval Augmented Generation (RAG) for NeurIPS abstracts.

This module provides RAG functionality to query papers and generate
contextual responses using OpenAI-compatible language model APIs.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from openai import OpenAI

from .config import get_config
from .paper_utils import format_search_results, build_context_from_papers, PaperFormattingError
from .mcp_tools import execute_mcp_tool, format_tool_result_for_llm

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for RAG-related errors."""

    pass


def parse_json_tool_call(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse JSON tool calls from LLM response.
    
    Some models return tool calls as JSON objects directly in the response content.
    This function attempts to parse the response as JSON and extract tool calls.
    
    Expected formats:
    - Single tool: {"name": "tool_name", "arguments": {...}}
    - Multiple tools: [{"name": "tool_name", "arguments": {...}}, ...]
    - OpenAI-like: {"tool_calls": [{"function": {"name": "...", "arguments": {...}}}]}
    
    Parameters
    ----------
    response_text : str
        The LLM response text to parse
        
    Returns
    -------
    list or None
        List of tool call dictionaries with 'name' and 'arguments' keys, or None if not valid JSON tool call
        
    Examples
    --------
    >>> text = '{"name": "analyze_topic_relevance", "arguments": {"query": "transformers"}}'
    >>> calls = parse_json_tool_call(text)
    >>> calls[0]['name']
    'analyze_topic_relevance'
    """
    if response_text is None or not isinstance(response_text, str) or not response_text.strip():
        return None
    
    try:
        # Try to parse as JSON
        parsed = json.loads(response_text.strip())
        
        # Handle different JSON formats
        tool_calls = []
        
        # Format 1: Single tool call object {"name": "...", "arguments": {...}}
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            tool_calls.append({
                "name": parsed["name"],
                "arguments": parsed.get("arguments", {})
            })
        
        # Format 2: Array of tool calls [{"name": "...", "arguments": {...}}, ...]
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    tool_calls.append({
                        "name": item["name"],
                        "arguments": item.get("arguments", {})
                    })
        
        # Format 3: OpenAI-like format {"tool_calls": [...]}
        elif isinstance(parsed, dict) and "tool_calls" in parsed:
            for tc in parsed["tool_calls"]:
                if isinstance(tc, dict):
                    # Handle {"function": {"name": "...", "arguments": {...}}}
                    if "function" in tc:
                        func = tc["function"]
                        if "name" in func:
                            tool_calls.append({
                                "name": func["name"],
                                "arguments": func.get("arguments", {})
                            })
                    # Handle direct format {"name": "...", "arguments": {...}}
                    elif "name" in tc:
                        tool_calls.append({
                            "name": tc["name"],
                            "arguments": tc.get("arguments", {})
                        })
        
        # Format 4: Function call format {"function": {"name": "...", "arguments": {...}}}
        elif isinstance(parsed, dict) and "function" in parsed:
            func = parsed["function"]
            if isinstance(func, dict) and "name" in func:
                tool_calls.append({
                    "name": func["name"],
                    "arguments": func.get("arguments", {})
                })
        
        # Return tool calls if any were found
        if tool_calls:
            logger.info(f"Parsed {len(tool_calls)} JSON tool call(s) from response")
            return tool_calls
        
        return None
    
    except (json.JSONDecodeError, ValueError, TypeError):
        # Not valid JSON or doesn't match expected format
        return None


class RAGChat:
    """
    RAG chat interface for querying NeurIPS papers.

    Uses embeddings for semantic search and OpenAI-compatible API for response generation.
    Optionally integrates with MCP clustering tools to answer questions about conference
    topics, trends, and developments.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Manager for embeddings and vector search.
    database : DatabaseManager
        Database instance for querying paper details. REQUIRED.
    lm_studio_url : str, optional
        URL for OpenAI-compatible API, by default "http://localhost:1234"
    model : str, optional
        Name of the language model, by default "auto"
    max_context_papers : int, optional
        Maximum number of papers to include in context, by default 5
    temperature : float, optional
        Sampling temperature for generation, by default 0.7
    enable_mcp_tools : bool, optional
        Enable MCP clustering tools for topic analysis, by default True

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
        """
        Initialize RAG chat.

        Parameters are optional and will use values from environment/config if not provided.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Manager for embeddings and vector search.
        database : DatabaseManager
            Database instance for querying paper details. REQUIRED - no fallback allowed.
        lm_studio_url : str, optional
            URL for OpenAI-compatible API. If None, uses config value.
        model : str, optional
            Name of the language model. If None, uses config value.
        max_context_papers : int, optional
            Maximum number of papers to include in context. If None, uses config value.
        temperature : float, optional
            Sampling temperature for generation. If None, uses config value.
        enable_mcp_tools : bool, optional
            Whether to enable MCP clustering tools for the LLM. Default is True.

        Raises
        ------
        RAGError
            If required parameters are missing or invalid.
        """
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
        self.enable_query_rewriting = config.enable_query_rewriting
        self.query_similarity_threshold = config.query_similarity_threshold
        self.enable_mcp_tools = enable_mcp_tools
        self.conversation_history: List[Dict[str, str]] = []
        self.last_search_query: Optional[str] = None

        # OpenAI client - lazy loaded on first use to avoid API calls during test collection
        self._openai_client: Optional[OpenAI] = None
        self._llm_backend_auth_token = config.llm_backend_auth_token

    @property
    def openai_client(self) -> OpenAI:
        """
        Get the OpenAI client, creating it lazily on first access.
        
        This lazy loading prevents API calls during test collection.
        
        Returns
        -------
        OpenAI
            Initialized OpenAI client instance.
        """
        if self._openai_client is None:
            self._openai_client = OpenAI(
                base_url=f"{self.lm_studio_url}/v1",
                api_key=self._llm_backend_auth_token or "lm-studio-local"
            )
        return self._openai_client

    def _analyze_and_route_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze query and decide whether to use tools or search papers.

        Uses the LLM to determine if the query requires MCP tools (for clustering,
        trends, topic analysis) or if it should be answered with paper retrieval.
        If tools are needed, returns tool call information. Otherwise, returns
        a rewritten query optimized for semantic search.

        Parameters
        ----------
        user_query : str
            Original user query or question.

        Returns
        -------
        dict
            Dictionary containing:
            - use_tools: bool - Whether to use MCP tools
            - tool_calls: list - Tool calls if use_tools is True (can be empty if not detected)
            - rewritten_query: str - Rewritten query if use_tools is False
            - original_query: str - The original query for reference

        Examples
        --------
        >>> chat = RAGChat(em, db)
        >>> result = chat._analyze_and_route_query("What are the main topics?")
        >>> if result['use_tools']:
        ...     print("Using tools:", result['tool_calls'])
        >>> else:
        ...     print("Searching with:", result['rewritten_query'])
        """
        # Build system prompt that decides routing
        if self.enable_mcp_tools:
            system_prompt = (
                "You are a query analysis assistant. Determine if the user's question requires:\n"
                "1. MCP TOOLS (clustering analysis) - for questions about:\n"
                "   - Overall topics/themes ('what are the main topics', 'research areas covered')\n"
                "   - Trends over time ('how has X evolved', 'topic evolution')\n"
                "   - Recent developments ('latest research on', 'recent papers about')\n"
                "   - Counting papers by topic ('how many papers about X')\n"
                "   - Visualization requests ('show me', 'visualize', 'plot')\n\n"
                "2. PAPER SEARCH - for specific questions about:\n"
                "   - Particular concepts/techniques\n"
                "   - Detailed explanations\n"
                "   - Specific papers or authors\n\n"
                "If MCP TOOLS are needed, respond with ONLY a valid JSON tool call in one of these formats:\n"
                "- Single: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
                "- Array: [{\"name\": \"tool1\", \"arguments\": {...}}, ...]\n\n"
                "If PAPER SEARCH is needed, respond with ONLY a rewritten search query (5-15 words).\n"
                "For follow-up questions, incorporate context from previous conversation."
            )
        else:
            # If MCP tools disabled, just do query rewriting
            system_prompt = (
                "You are a query rewriting assistant. Your task is to rewrite user questions "
                "into effective search queries for finding relevant research papers. "
                "Convert conversational questions into keyword-based search queries. "
                "For follow-up questions, incorporate context from previous conversation. "
                "Output ONLY the rewritten query, nothing else. "
                "Keep queries concise (5-15 words) and focused on key concepts."
            )

        # Build messages with conversation history for context
        messages = [{"role": "system", "content": system_prompt}]

        # Include recent conversation history for context (last 4 messages)
        if self.conversation_history:
            context_history = self.conversation_history[-4:]
            messages.extend(context_history)

        # Add current query
        if self.enable_mcp_tools:
            messages.append({
                "role": "user",
                "content": f"Analyze this query and respond appropriately: {user_query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Rewrite this query: {user_query}"
            })

        try:
            # Don't pass tools here - we want the model to decide via content
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent decisions
                max_tokens=200,  # Allow for tool calls or rewritten queries
                timeout=30,  # Shorter timeout for quick analysis
            )
            response_content = response.choices[0].message.content.strip()

            # Check if response is a tool call (JSON format)
            if self.enable_mcp_tools:
                tool_calls = parse_json_tool_call(response_content)
                if tool_calls:
                    logger.info(f"Query analysis: Using {len(tool_calls)} MCP tool(s)")
                    return {
                        "use_tools": True,
                        "tool_calls": tool_calls,
                        "rewritten_query": None,
                        "original_query": user_query
                    }

            # Otherwise, treat as rewritten query
            rewritten = response_content.strip("\"'")
            logger.info(f"Query analysis: Rewrote query '{user_query}' -> '{rewritten}'")
            return {
                "use_tools": False,
                "tool_calls": [],
                "rewritten_query": rewritten,
                "original_query": user_query
            }

        except Exception as e:
            logger.warning(f"Query analysis failed: {e}, using original query for search")
            return {
                "use_tools": False,
                "tool_calls": [],
                "rewritten_query": user_query,
                "original_query": user_query
            }

    def _should_retrieve_papers(self, rewritten_query: str) -> bool:
        """
        Determine if new papers should be retrieved based on query similarity.

        Compares the rewritten query with the last search query to avoid
        redundant retrievals for similar follow-up questions.

        Parameters
        ----------
        rewritten_query : str
            The rewritten search query.

        Returns
        -------
        bool
            True if papers should be retrieved, False to reuse previous context.

        Examples
        --------
        >>> chat = RAGChat(em, db)
        >>> chat._should_retrieve_papers("deep learning networks")
        True
        """
        # Always retrieve if this is the first query
        if self.last_search_query is None:
            return True

        # Simple similarity check: count common words
        # For better similarity, could use embeddings, but this is fast and effective
        query_words = set(rewritten_query.lower().split())
        last_words = set(self.last_search_query.lower().split())

        # Calculate Jaccard similarity
        if not query_words or not last_words:
            return True

        intersection = query_words & last_words
        union = query_words | last_words
        similarity = len(intersection) / len(union)

        # Retrieve if queries are less similar than threshold
        should_retrieve = similarity < self.query_similarity_threshold

        logger.info(
            f"Query similarity: {similarity:.2f} (threshold: {self.query_similarity_threshold}). "
            f"{'Retrieving new papers' if should_retrieve else 'Reusing previous context'}"
        )

        return should_retrieve

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
            Metadata filter for paper search.
        system_prompt : str, optional
            Custom system prompt for the model.

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

            # Analyze query and decide routing (tools vs paper search)
            if self.enable_query_rewriting:
                route_info = self._analyze_and_route_query(question)
            else:
                # If query rewriting disabled, just use original query for search
                route_info = {
                    "use_tools": False,
                    "tool_calls": [],
                    "rewritten_query": question,
                    "original_query": question
                }
                logger.info("Query rewriting disabled, using original query")

            # ROUTE 1: Use MCP Tools
            if route_info["use_tools"] and route_info["tool_calls"]:
                logger.info(f"Executing {len(route_info['tool_calls'])} MCP tool(s)")
                
                # Execute tools
                tool_results = []
                for tool_call in route_info["tool_calls"]:
                    function_name = tool_call['name']
                    function_args = tool_call['arguments']
                    
                    logger.info(f"Executing tool: {function_name} with args: {function_args}")
                    tool_result = execute_mcp_tool(function_name, function_args)
                    formatted_result = format_tool_result_for_llm(function_name, tool_result)
                    
                    tool_results.append({
                        'name': function_name,
                        'result': formatted_result
                    })
                
                # Build context from tool results
                tool_context = "\n\n".join([
                    f"Tool: {tr['name']}\nResult:\n{tr['result']}"
                    for tr in tool_results
                ])
                
                # Generate response based on tool results
                logger.info("Generating response from tool results")
                response_text = self._generate_response_from_context(
                    question, tool_context, system_prompt, is_tool_result=True
                )
                
                # Store in conversation history
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return {
                    "response": response_text,
                    "papers": [],  # No papers used for tool-based queries
                    "metadata": {
                        "n_papers": 0,
                        "model": self.model,
                        "rewritten_query": None,
                        "used_tools": True,
                        "tools_executed": [tc['name'] for tc in route_info['tool_calls']],
                        "retrieved_new_papers": False,
                    },
                }

            # ROUTE 2: Use Paper Search
            rewritten_query = route_info["rewritten_query"]
            
            # Check if we should retrieve new papers or reuse previous context
            should_retrieve = self._should_retrieve_papers(rewritten_query) if self.enable_query_rewriting else True

            if should_retrieve:
                # Search for relevant papers using rewritten query
                logger.info(f"Searching for papers with rewritten query: {rewritten_query}")
                search_results = self.embeddings_manager.search_similar(
                    rewritten_query, n_results=n_results, where=metadata_filter
                )

                # Store rewritten query for next comparison
                self.last_search_query = rewritten_query

                if not search_results["ids"][0]:
                    logger.warning("No relevant papers found")
                    return {
                        "response": "I couldn't find any relevant papers to answer your question.",
                        "papers": [],
                        "metadata": {"n_papers": 0, "rewritten_query": rewritten_query, "used_tools": False},
                    }

                # Format context from papers using shared utility
                papers = format_search_results(search_results, self.database, include_documents=True)
                context = build_context_from_papers(papers)

                # Cache papers for potential reuse
                self._cached_papers = papers
                self._cached_context = context
            else:
                # Reuse cached papers and context from previous query
                logger.info("Reusing cached papers from previous query")
                papers = getattr(self, "_cached_papers", [])
                context = getattr(self, "_cached_context", "")

                if not papers:
                    # Fallback: retrieve papers if cache is empty
                    logger.warning("Cache empty, retrieving papers")
                    search_results = self.embeddings_manager.search_similar(
                        rewritten_query, n_results=n_results, where=metadata_filter
                    )
                    self.last_search_query = rewritten_query

                    if not search_results["ids"][0]:
                        logger.warning("No relevant papers found")
                        return {
                            "response": "I couldn't find any relevant papers to answer your question.",
                            "papers": [],
                            "metadata": {"n_papers": 0, "rewritten_query": rewritten_query, "used_tools": False},
                        }

                    # Format context from papers using shared utility
                    papers = format_search_results(search_results, self.database, include_documents=True)
                    context = build_context_from_papers(papers)

                    # Cache papers for potential reuse
                    self._cached_papers = papers
                    self._cached_context = context

            # Generate response from paper context (no tool decision here)
            logger.info(f"Generating response using {len(papers)} papers as context")
            response_text = self._generate_response_from_context(question, context, system_prompt, is_tool_result=False)

            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return {
                "response": response_text,
                "papers": papers,
                "metadata": {
                    "n_papers": len(papers),
                    "model": self.model,
                    "rewritten_query": rewritten_query,
                    "retrieved_new_papers": should_retrieve,
                    "used_tools": False,
                },
            }

        except PaperFormattingError as e:
            raise RAGError(f"Failed to format papers: {str(e)}") from e
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
            Whether to retrieve papers as context, by default True
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
            # Use only conversation history without paper context
            response_text = self._generate_response_from_context(message, "", None, is_tool_result=False)
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return {
                "response": response_text,
                "papers": [],
                "metadata": {"n_papers": 0, "model": self.model},
            }

    def reset_conversation(self):
        """
        Reset conversation history and cached context.

        Examples
        --------
        >>> chat.reset_conversation()
        """
        self.conversation_history = []
        self.last_search_query = None
        self._cached_papers = []
        self._cached_context = ""
        logger.info("Conversation history and cache reset")

    def _generate_response_from_context(
        self, 
        question: str, 
        context: str, 
        system_prompt: Optional[str] = None,
        is_tool_result: bool = False
    ) -> str:
        """
        Generate response from provided context (papers or tool results).

        This method no longer makes tool call decisions. Tool calls are decided
        earlier in the flow during query analysis.

        Parameters
        ----------
        question : str
            User's question.
        context : str
            Context from papers or tool results.
        system_prompt : str, optional
            Custom system prompt.
        is_tool_result : bool
            Whether context comes from tool results (vs papers).

        Returns
        -------
        str
            Generated response.

        Raises
        ------
        RAGError
            If generation fails.
        """
        # Default system prompt
        if system_prompt is None:
            if is_tool_result:
                system_prompt = (
                    "You are an AI assistant helping researchers analyze conference data. "
                    "You have been provided with results from clustering and analysis tools. "
                    "Use these results to answer the user's question accurately and concisely. "
                    "Present the information in a clear, easy-to-understand format."
                )
            else:
                system_prompt = (
                    "You are an AI assistant helping researchers find relevant NeurIPS abstracts. "
                    "Use the provided paper abstracts to answer questions accurately and concisely. "
                    "If the papers don't contain enough information to answer the question, suggest a query that might return more relevant results. "
                    "Always cite which papers you're referencing (e.g., 'Paper 1', 'Paper 2'), using local links: <a href='#paper-1'>Paper-1</a>, <a href='#paper-2'>Paper-2</a> etc."
                )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 10 messages)
        if self.conversation_history:
            messages.extend(self.conversation_history[-10:])

        # Add current question with context
        if context:
            if is_tool_result:
                user_message = f"Tool Results:\n\n{context}\n\nQuestion: {question}\n\nPlease provide a clear answer based on the tool results above."
            else:
                user_message = f"Context from relevant papers:\n\n{context}\n\nQuestion: {question}"
        else:
            user_message = question

        messages.append({"role": "user", "content": user_message})

        # Prepare API call parameters - NO TOOLS passed here
        # Tool decisions are made earlier in query analysis
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1000,
            "timeout": 180,
        }

        # Call OpenAI-compatible API using OpenAI client
        try:
            response = self.openai_client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            raise RAGError(f"Failed to generate response: {str(e)}")
    
    
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
        with open(output_path, "w") as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"Conversation exported to {output_path}")
