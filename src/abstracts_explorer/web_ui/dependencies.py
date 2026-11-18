"""
Shared request-scoped dependencies for the web UI.

Database connections, the embeddings manager, and per-session RAG chat
instances live here so routes and tests use one set of helpers.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from flask import g, session

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat

logger = logging.getLogger(__name__)

# Initialize components (lazy loading)
embeddings_manager = None
_rag_chat_cache: "OrderedDict[str, RAGChat]" = OrderedDict()
_RAG_CHAT_CACHE_MAX = 1024
_RAG_CHAT_LOCK = threading.Lock()
_SCHEMA_INIT_LOCK = threading.Lock()
_INITIALIZED_DB_URLS: set[str] = set()


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
        with _SCHEMA_INIT_LOCK:
            if g.db.database_url not in _INITIALIZED_DB_URLS:
                g.db.create_tables()
                _INITIALIZED_DB_URLS.add(g.db.database_url)
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

    with _RAG_CHAT_LOCK:
        if session_id in _rag_chat_cache:
            _rag_chat_cache.move_to_end(session_id)
            return _rag_chat_cache[session_id]

        while len(_rag_chat_cache) >= _RAG_CHAT_CACHE_MAX:
            _evicted_id, evicted = _rag_chat_cache.popitem(last=False)
            try:
                evicted.close()
            except Exception as exc:
                logger.debug("Failed to close evicted RAG chat %s: %s", _evicted_id, exc)

        config = get_config()
        em = get_embeddings_manager()
        chat = RAGChat(
            embeddings_manager=em,
            database=database,
            backend_url=config.llm_backend_url,
            model=config.chat_model,
        )
        _rag_chat_cache[session_id] = chat
        return chat


def _reset_rag_chat_local():
    """
    Reset the RAG chat cache.

    When called inside a request context, only clears the entry for the
    current session.  When called outside a request context (e.g., by
    tests during setup/teardown), clears the entire cache.
    """
    try:
        session_id = _get_session_id()
        with _RAG_CHAT_LOCK:
            chat = _rag_chat_cache.pop(session_id, None)
        if chat is not None:
            chat.close()
    except (RuntimeError, KeyError):
        with _RAG_CHAT_LOCK:
            chats = list(_rag_chat_cache.values())
            _rag_chat_cache.clear()
        for chat in chats:
            try:
                chat.close()
            except Exception as exc:
                logger.debug("Failed to close RAG chat: %s", exc)


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
        "name": "HZDR Chat",
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
