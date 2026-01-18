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
from .mcp_tools import get_mcp_tools_schema, execute_mcp_tool, format_tool_result_for_llm

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for RAG-related errors."""

    pass


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

    def _rewrite_query(self, user_query: str) -> str:
        """
        Rewrite user query into an effective search query.

        Uses the LLM to transform conversational questions into optimized
        search queries, considering conversation history for follow-up questions.

        Parameters
        ----------
        user_query : str
            Original user query or question.

        Returns
        -------
        str
            Rewritten query optimized for semantic search.

        Raises
        ------
        RAGError
            If query rewriting fails.

        Examples
        --------
        >>> chat = RAGChat(em, db)
        >>> rewritten = chat._rewrite_query("What about transformers?")
        >>> print(rewritten)
        "transformer architecture attention mechanism neural networks"
        """
        # Build system prompt for query rewriting
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
        messages.append(
            {
                "role": "user",
                "content": f"Rewrite this query: {user_query}",
            }
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent rewrites
                max_tokens=100,  # Short rewritten queries
                timeout=30,  # Shorter timeout for quick rewriting
            )
            rewritten = response.choices[0].message.content.strip()

            # Remove any quotes or extra formatting
            rewritten = rewritten.strip("\"'")

            logger.info(f"Rewrote query: '{user_query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original query")
            return user_query

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

            # Rewrite query for better semantic search (if enabled)
            if self.enable_query_rewriting:
                rewritten_query = self._rewrite_query(question)
            else:
                rewritten_query = question
                logger.info("Query rewriting disabled, using original query")

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
                        "metadata": {"n_papers": 0, "rewritten_query": rewritten_query},
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
                            "metadata": {"n_papers": 0, "rewritten_query": rewritten_query},
                        }

                    # Format context from papers using shared utility
                    papers = format_search_results(search_results, self.database, include_documents=True)
                    context = build_context_from_papers(papers)

                    # Cache papers for potential reuse
                    self._cached_papers = papers
                    self._cached_context = context

            # Generate response using OpenAI API (uses original question, not rewritten query)
            logger.info(f"Generating response using {len(papers)} papers as context")
            response_text = self._generate_response(question, context, system_prompt)

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
            response_text = self._generate_response(message, "", None)
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

    def _generate_response(self, question: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response using OpenAI-compatible API with optional MCP tool support.

        Parameters
        ----------
        question : str
            User's question.
        context : str
            Context from papers.
        system_prompt : str, optional
            Custom system prompt.

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
            system_prompt = (
                "You are an AI assistant helping researchers find relevant NeurIPS abstracts. "
                "Use the provided paper abstracts to answer questions accurately and concisely. "
                "If the papers don't contain enough information to answer the question, suggest a query that might return more relevant results. "
                "Always cite which papers you're referencing (e.g., 'Paper 1', 'Paper 2'), using local links: <a href='#paper-1'>Paper-1</a>, <a href='#paper-2'>Paper-2</a> etc."
            )
            
            # Enhance system prompt when MCP tools are enabled
            if self.enable_mcp_tools:
                system_prompt += (
                    "\n\nYou have access to clustering analysis tools that can help answer questions about "
                    "overall conference topics, trends over time, and recent developments. Use these tools when appropriate "
                    "for questions about general themes, topic evolution, or recent research in specific areas."
                )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 10 messages)
        if self.conversation_history:
            messages.extend(self.conversation_history[-10:])

        # Add current question with context
        if context:
            user_message = f"Context from relevant papers:\n\n{context}\n\nQuestion: {question}"
        else:
            user_message = question

        messages.append({"role": "user", "content": user_message})

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1000,
            "timeout": 180,
        }
        
        # Add MCP tools if enabled
        if self.enable_mcp_tools:
            api_params["tools"] = get_mcp_tools_schema()
            api_params["tool_choice"] = "auto"  # Let the model decide when to use tools

        # Call OpenAI-compatible API using OpenAI client
        try:
            response = self.openai_client.chat.completions.create(**api_params)
            
            # Check if the model wants to use tools
            if self.enable_mcp_tools and response.choices[0].message.tool_calls:
                return self._handle_tool_calls(response, messages)
            
            # No tool calls, return direct response
            return response.choices[0].message.content

        except Exception as e:
            raise RAGError(f"Failed to generate response: {str(e)}")
    
    def _handle_tool_calls(self, initial_response, messages: List[Dict[str, Any]]) -> str:
        """
        Handle tool calls from the LLM response.
        
        This method executes requested MCP tools and sends the results back to the LLM
        to generate a final response.
        
        Parameters
        ----------
        initial_response
            The initial API response containing tool calls
        messages : list
            The message history to continue the conversation
            
        Returns
        -------
        str
            Final response from the LLM after tool execution
        """
        assistant_message = initial_response.choices[0].message
        tool_calls = assistant_message.tool_calls
        
        # Add the assistant's tool call message to history
        tool_calls_data: List[Dict[str, Any]] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in tool_calls
        ]
        
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": tool_calls_data
        })
        
        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"LLM requested tool: {function_name} with args: {function_args}")
            
            # Execute the tool
            tool_result = execute_mcp_tool(function_name, function_args)
            
            # Format the result for better LLM consumption
            formatted_result = format_tool_result_for_llm(function_name, tool_result)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": formatted_result
            })
        
        # Get final response from LLM with tool results
        try:
            final_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000,
                timeout=180,
            )
            return final_response.choices[0].message.content
        except Exception as e:
            raise RAGError(f"Failed to generate response after tool calls: {str(e)}")

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
