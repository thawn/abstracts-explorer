"""
Shared request-scoped dependencies for the web UI.

Database connections, the embeddings manager, and per-session RAG chat
instances live here so routes and tests use one set of helpers.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from flask import g, session

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat

logger = logging.getLogger(__name__)

# Initialize components (lazy loading)
embeddings_manager = None
_rag_chat_cache: Dict[str, RAGChat] = {}
_RAG_CHAT_CACHE_MAX = 1024


def get_database():
    """
    Get or create database connection (thread-local using Flask g).

    Returns
    -------
    DatabaseManager
        Database instance
    """
    if "db" not in g:
        g.db = DatabaseManager()
        g.db.create_tables()
    return g.db


def get_embeddings_manager():
    """
    Get or create embeddings manager.

    Returns
    -------
    EmbeddingsManager
        Embeddings manager instance
    """
    global embeddings_manager
    if embeddings_manager is None:
        config = get_config()
        embeddings_manager = EmbeddingsManager(
            backend_url=config.llm_backend_url,
            model_name=config.embedding_model,
            collection_name=config.collection_name,
        )
    return embeddings_manager


def _get_session_id() -> str:
    """
    Get or create a unique session identifier.

    Generates a UUID4 on first request and persists it in the Flask
    session cookie so it survives page reloads within the same tab.
    Closing the tab (session cookie expiry) generates a new ID on
    the next request.
    """
    if "_session_id" not in session:
        session["_session_id"] = uuid.uuid4().hex
    return session["_session_id"]


def get_rag_chat():
    """
    Get or create a RAG chat instance scoped to the current browser session.

    Each unique browser session gets its own ``RAGChat`` instance so that
    conversation history is isolated per browser tab. A new page reload
    (new session cookie) starts with a fresh conversation.

    Returns
    -------
    RAGChat
        RAG chat instance for the current session.
    """
    database = get_database()
    session_id = _get_session_id()

    if session_id not in _rag_chat_cache:
        if len(_rag_chat_cache) >= _RAG_CHAT_CACHE_MAX:
            oldest_key = next(iter(_rag_chat_cache))
            del _rag_chat_cache[oldest_key]

        config = get_config()
        em = get_embeddings_manager()
        _rag_chat_cache[session_id] = RAGChat(
            embeddings_manager=em,
            database=database,
            backend_url=config.llm_backend_url,
            model=config.chat_model,
        )

    return _rag_chat_cache[session_id]


def _reset_rag_chat_local():
    """
    Reset the RAG chat cache.

    When called inside a request context, only clears the entry for the
    current session.  When called outside a request context (e.g., by
    tests during setup/teardown), clears the entire cache.
    """
    try:
        session_id = _get_session_id()
        if session_id in _rag_chat_cache:
            del _rag_chat_cache[session_id]
    except (RuntimeError, KeyError):
        _rag_chat_cache.clear()


def teardown_db(exception) -> None:
    """Close database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


# Known LLM backend definitions: each entry maps a URL fragment to backend metadata.
# Order matters – more specific patterns should come first.
_LLM_BACKENDS: List[Dict[str, Any]] = [
    {
        "url_fragments": ["blablador.fz-juelich.de"],
        "name": "BLABLADOR",
        "homepage": "https://helmholtz-blablador.fz-juelich.de",
        "logo": "blablador-logo.png",
    },
    {
        "url_fragments": ["chat.fz-rossendorf.de"],
        "name": "chat.fz-rossendorf.de",
        "homepage": "https://chat.fz-rossendorf.de",
        "logo": None,
    },
    {
        "url_fragments": ["localhost:1234", "127.0.0.1:1234"],
        "name": "LM Studio",
        "homepage": "https://lmstudio.ai",
        "logo": None,
    },
]


def get_llm_backend_info(backend_url: str) -> Dict[str, Optional[str]]:
    """
    Detect the LLM backend from its URL and return display metadata.

    Parameters
    ----------
    backend_url : str
        The URL of the configured LLM backend (e.g. from ``config.llm_backend_url``).

    Returns
    -------
    dict
        A dictionary with keys ``name``, ``homepage``, and ``logo`` (each
        ``None`` when unknown).
    """
    for backend in _LLM_BACKENDS:
        if any(fragment in backend_url for fragment in backend["url_fragments"]):
            return {
                "name": backend["name"],
                "homepage": backend["homepage"],
                "logo": backend["logo"],
            }
    return {"name": None, "homepage": None, "logo": None}
