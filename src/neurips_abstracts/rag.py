"""
Retrieval Augmented Generation (RAG) for NeurIPS abstracts.

This module provides RAG functionality to query papers and generate
contextual responses using LM Studio's language models.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests

from .config import get_config

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for RAG-related errors."""

    pass


class RAGChat:
    """
    RAG chat interface for querying NeurIPS papers.

    Uses embeddings for semantic search and LM Studio for response generation.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Manager for embeddings and vector search.
    lm_studio_url : str, optional
        URL for LM Studio API, by default "http://localhost:1234"
    model : str, optional
        Name of the language model, by default "auto"
    max_context_papers : int, optional
        Maximum number of papers to include in context, by default 5
    temperature : float, optional
        Sampling temperature for generation, by default 0.7

    Examples
    --------
    >>> from neurips_abstracts.embeddings import EmbeddingsManager
    >>> em = EmbeddingsManager("chroma_db")
    >>> em.connect()
    >>> chat = RAGChat(em)
    >>> response = chat.query("What are the latest advances in deep learning?")
    >>> print(response)
    """

    def __init__(
        self,
        embeddings_manager,
        database=None,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        max_context_papers: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize RAG chat.

        Parameters are optional and will use values from environment/config if not provided.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Manager for embeddings and vector search.
        database : NeurIPSDatabase, optional
            Database instance for querying paper details.
        lm_studio_url : str, optional
            URL for LM Studio API. If None, uses config value.
        model : str, optional
            Name of the language model. If None, uses config value.
        max_context_papers : int, optional
            Maximum number of papers to include in context. If None, uses config value.
        temperature : float, optional
            Sampling temperature for generation. If None, uses config value.
        """
        config = get_config()
        self.embeddings_manager = embeddings_manager
        self.database = database
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.model = model or config.chat_model
        self.max_context_papers = max_context_papers or config.max_context_papers
        self.temperature = temperature or config.chat_temperature
        self.conversation_history: List[Dict[str, str]] = []

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

            # Search for relevant papers
            logger.info(f"Searching for papers related to: {question}")
            search_results = self.embeddings_manager.search_similar(
                question, n_results=n_results, where=metadata_filter
            )

            if not search_results["ids"][0]:
                logger.warning("No relevant papers found")
                return {
                    "response": "I couldn't find any relevant papers to answer your question.",
                    "papers": [],
                    "metadata": {"n_papers": 0},
                }

            # Format context from papers
            papers = self._format_papers(search_results)
            context = self._build_context(papers)

            # Generate response using LM Studio
            logger.info(f"Generating response using {len(papers)} papers as context")
            response_text = self._generate_response(question, context, system_prompt)

            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return {
                "response": response_text,
                "papers": papers,
                "metadata": {"n_papers": len(papers), "model": self.model},
            }

        except requests.exceptions.RequestException as e:
            raise RAGError(f"API request failed: {str(e)}") from e
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
        Reset conversation history.

        Examples
        --------
        >>> chat.reset_conversation()
        """
        self.conversation_history = []
        logger.info("Conversation history reset")

    def _format_papers(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format search results into structured paper data.

        Parameters
        ----------
        search_results : dict
            Results from embeddings search.

        Returns
        -------
        list
            List of formatted paper dictionaries.
        """
        papers = []
        for i in range(len(search_results["ids"][0])):
            paper_id = search_results["ids"][0][i]
            metadata = search_results["metadatas"][0][i]

            # Get full paper details from database if available
            if self.database is not None:
                try:
                    # Get full paper record from database (includes session, poster_position, etc.)
                    paper_rows = self.database.query("SELECT * FROM papers WHERE id = ?", (paper_id,))
                    if paper_rows:
                        paper = dict(paper_rows[0])

                        # Get authors from authors table
                        authors_rows = self.database.get_paper_authors(paper_id)
                        paper["authors"] = [a["fullname"] for a in authors_rows]

                        # Add distance/similarity scores
                        if "distances" in search_results and search_results["distances"][0]:
                            distance = search_results["distances"][0][i]
                            paper["distance"] = distance
                            paper["similarity"] = 1 - distance if distance <= 1 else 0

                        # Add aliases for consistency (database uses 'name', web UI expects 'title' too)
                        if "name" in paper:
                            paper["title"] = paper["name"]

                        # Add abstract from search results if not in database
                        if not paper.get("abstract") and "documents" in search_results:
                            paper["abstract"] = search_results["documents"][0][i]

                        papers.append(paper)
                        continue
                except Exception as e:
                    logger.debug(f"Failed to get full paper details for {paper_id}: {e}")
                    # Fall through to metadata-based approach

            # Fallback: use metadata from ChromaDB (less complete)
            authors_str = metadata.get("authors", "N/A")
            authors = [authors_str] if isinstance(authors_str, str) else []

            paper = {
                "id": paper_id,
                "title": metadata.get("title", "N/A"),
                "name": metadata.get("title", "N/A"),
                "authors": authors,
                "abstract": search_results["documents"][0][i] if "documents" in search_results else "",
                "decision": metadata.get("decision", "N/A"),
                "topic": metadata.get("topic", "N/A"),
                "session": metadata.get("session", None),
                "poster_position": metadata.get("poster_position", None),
                "paper_url": metadata.get("paper_url", None),
                "distance": search_results["distances"][0][i] if "distances" in search_results else None,
                "similarity": 1 - search_results["distances"][0][i] if search_results["distances"][0][i] <= 1 else 0,
            }
            papers.append(paper)
        return papers

    def _build_context(self, papers: List[Dict[str, Any]]) -> str:
        """
        Build context string from papers.

        Parameters
        ----------
        papers : list
            List of paper dictionaries.

        Returns
        -------
        str
            Formatted context string.
        """
        context_parts = []
        for i, paper in enumerate(papers, 1):
            context_parts.append(f"Paper {i}:")

            # Handle title (could be 'title' or 'name')
            title = paper.get("title") or paper.get("name", "N/A")
            context_parts.append(f"Title: {title}")

            # Handle authors (could be list or string)
            authors = paper.get("authors", "N/A")
            if isinstance(authors, list):
                authors = ", ".join(authors) if authors else "N/A"
            context_parts.append(f"Authors: {authors}")

            # Handle optional fields
            topic = paper.get("topic", "N/A")
            context_parts.append(f"Topic: {topic}")

            decision = paper.get("decision", "N/A")
            context_parts.append(f"Decision: {decision}")

            abstract = paper.get("abstract", "N/A")
            context_parts.append(f"Abstract: {abstract}")

            context_parts.append("")  # Empty line between papers

        return "\n".join(context_parts)

    def _generate_response(self, question: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response using LM Studio.

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
                "You are an AI assistant helping researchers understand NeurIPS papers. "
                "Use the provided paper abstracts to answer questions accurately and concisely. "
                "If the papers don't contain enough information to answer the question, say so. "
                "Always cite which papers you're referencing (e.g., 'Paper 1', 'Paper 2')."
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

        # Call LM Studio API
        try:
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 1000,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise RAGError("Request to LM Studio timed out")
        except requests.exceptions.HTTPError as e:
            raise RAGError(f"LM Studio API error: {e.response.status_code}")
        except KeyError:
            raise RAGError("Invalid response from LM Studio API")
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
