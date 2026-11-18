"""
Shared request-scoped services for the web UI.

Database connections, the embeddings manager, and per-session chat history
live here so routes and tests use one set of helpers.  RAG chat instances are
created per request so their async HTTP clients are never reused across
Waitress worker threads.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from flask import g, session

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat

logger = logging.getLogger(__name__)

# Initialize components (lazy loading)
embeddings_manager = None


@dataclass
class _ChatSessionState:
    """Conversation state retained between requests for one browser session."""

    history: List[Any] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)


_chat_sessions: "OrderedDict[str, _ChatSessionState]" = OrderedDict()
_CHAT_SESSION_CACHE_MAX = 1024
_CHAT_SESSION_CACHE_LOCK = threading.Lock()
_CHAT_HISTORY_SWEEP_INTERVAL_SECONDS = 3600
_DEFAULT_CHAT_HISTORY_TTL_SECONDS = 86400
_chat_history_cleanup_lock = threading.Lock()
_chat_history_cleanup_thread: Optional[threading.Thread] = None
_chat_history_cleanup_stop: Optional[threading.Event] = None
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


def _chat_history_ttl() -> int:
    """Return the configured chat-history idle TTL in seconds."""
    ttl = getattr(get_config(), "chat_history_ttl", _DEFAULT_CHAT_HISTORY_TTL_SECONDS)
    if isinstance(ttl, bool) or not isinstance(ttl, int):
        return _DEFAULT_CHAT_HISTORY_TTL_SECONDS
    return max(0, ttl)


def _get_chat_session_state(session_id: str) -> _ChatSessionState:
    """Return the chat state for *session_id*, creating it when needed."""
    now = time.monotonic()
    ttl = _chat_history_ttl()

    with _CHAT_SESSION_CACHE_LOCK:
        state = _chat_sessions.get(session_id)
        if state is not None:
            last_used = getattr(state, "last_used", now)
            if ttl > 0 and now - last_used > ttl:
                state = _ChatSessionState(last_used=now)
                _chat_sessions[session_id] = state
            else:
                state.last_used = now
                _chat_sessions.move_to_end(session_id)
            return state

        while len(_chat_sessions) >= _CHAT_SESSION_CACHE_MAX:
            _chat_sessions.popitem(last=False)

        state = _ChatSessionState(last_used=now)
        _chat_sessions[session_id] = state
        return state


def _purge_expired_chat_sessions(
    *,
    now: Optional[float] = None,
    ttl_seconds: Optional[int] = None,
) -> int:
    """
    Remove chat session states that have been idle longer than the TTL.

    Parameters
    ----------
    now : float, optional
        Monotonic timestamp to evaluate against. Defaults to the current time.
    ttl_seconds : int, optional
        Override the configured TTL for testing or explicit sweeps.

    Returns
    -------
    int
        Number of expired sessions removed.
    """
    ttl = _chat_history_ttl() if ttl_seconds is None else ttl_seconds
    if ttl <= 0:
        return 0

    current = time.monotonic() if now is None else now
    with _CHAT_SESSION_CACHE_LOCK:
        expired = [
            session_id
            for session_id, state in _chat_sessions.items()
            if current - getattr(state, "last_used", current) > ttl
        ]
        for session_id in expired:
            del _chat_sessions[session_id]

    if expired:
        logger.info("Expired %d stale chat session(s)", len(expired))
    return len(expired)


def _chat_history_cleanup_loop(stop_event: threading.Event) -> None:
    """Periodically purge stale chat sessions until *stop_event* is set."""
    while not stop_event.wait(_CHAT_HISTORY_SWEEP_INTERVAL_SECONDS):
        try:
            _purge_expired_chat_sessions()
        except Exception:
            logger.exception("Failed to purge expired chat histories")


def start_chat_history_cleanup() -> None:
    """Start the background chat-history cleanup thread, if not already running."""
    global _chat_history_cleanup_thread, _chat_history_cleanup_stop
    with _chat_history_cleanup_lock:
        if _chat_history_cleanup_thread is not None and _chat_history_cleanup_thread.is_alive():
            return

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_chat_history_cleanup_loop,
            args=(stop_event,),
            name="chat-history-cleanup",
            daemon=True,
        )
        _chat_history_cleanup_stop = stop_event
        _chat_history_cleanup_thread = thread
        thread.start()


def stop_chat_history_cleanup() -> None:
    """Stop the background chat-history cleanup thread."""
    global _chat_history_cleanup_thread, _chat_history_cleanup_stop
    with _chat_history_cleanup_lock:
        stop_event = _chat_history_cleanup_stop
        thread = _chat_history_cleanup_thread
        _chat_history_cleanup_stop = None
        _chat_history_cleanup_thread = None

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)


def get_rag_chat() -> RAGChat:
    """
    Create a RAG chat instance for the current browser session.

    A fresh ``RAGChat`` is created for every request so its async HTTP client
    is only used on the current worker thread's event loop.  The session's
    message history is restored so multi-turn conversations keep working.

    Returns
    -------
    RAGChat
        RAG chat instance seeded with the current session's history.
    """
    state = _get_chat_session_state(_get_session_id())
    config = get_config()
    return RAGChat(
        embeddings_manager=get_embeddings_manager(),
        database=get_database(),
        backend_url=config.llm_backend_url,
        model=config.chat_model,
        message_history=state.history,
    )


@contextmanager
def chat_session(reset: bool = False) -> Iterator[RAGChat]:
    """
    Yield a per-request RAG chat and persist its conversation history.

    Parameters
    ----------
    reset : bool, optional
        Clear the session's conversation history before the query.

    Yields
    ------
    RAGChat
        Chat instance scoped to the current request.
    """
    state = _get_chat_session_state(_get_session_id())
    with state.lock:
        if reset:
            state.history = []

        chat = get_rag_chat()
        try:
            yield chat
        finally:
            history = getattr(chat, "message_history", None)
            if isinstance(history, list):
                state.history = history
            state.last_used = time.monotonic()
            try:
                chat.close()
            except Exception as exc:
                logger.debug("Failed to close RAG chat: %s", exc)


def reset_chat_history() -> None:
    """Clear the current browser session's conversation history."""
    state = _get_chat_session_state(_get_session_id())
    with state.lock:
        state.history = []


def _reset_rag_chat_local():
    """
    Reset the chat session cache.

    When called inside a request context, only clears the entry for the
    current session.  When called outside a request context (e.g., by
    tests during setup/teardown), clears the entire cache.
    """
    try:
        session_id = _get_session_id()
        with _CHAT_SESSION_CACHE_LOCK:
            _chat_sessions.pop(session_id, None)
    except (RuntimeError, KeyError):
        with _CHAT_SESSION_CACHE_LOCK:
            _chat_sessions.clear()


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
