"""
Opt-in web request profiling.

Enabled with ``WEB_PROFILE=1``.  Records one JSONL line per request containing
wall time, SQL fingerprints, Chroma/embedding/LLM/tool spans, and a compact
summary in the application log.
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import Optional

from flask import Flask, Response, g, request

from abstracts_explorer.config import get_config
from abstracts_explorer.profiling.spans import ProfilingSession

logger = logging.getLogger(__name__)

_EXTENSION_KEY = "abstracts_profiling"
_SESSION: Optional[ProfilingSession] = None
_SKIP_PREFIXES = ("/static/",)
_SKIP_PATHS = {"/health", "/favicon.ico"}


def _truthy(value: Optional[str]) -> bool:
    """Return whether an environment value enables a flag."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_output_path() -> Path:
    """Return the configured profiling output path."""
    configured = os.environ.get("PROFILE_OUTPUT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = Path(get_config().data_dir) / path
        return path
    return Path(get_config().data_dir) / "profiles" / f"web-profile-{os.getpid()}.jsonl"


def init_profiling(app: Flask) -> Optional[ProfilingSession]:
    """
    Enable request profiling on *app* when ``WEB_PROFILE`` is set.

    Parameters
    ----------
    app : Flask
        Flask application instance.

    Returns
    -------
    ProfilingSession or None
        The active profiling session, or ``None`` when disabled.
    """
    global _SESSION

    if not _truthy(os.environ.get("WEB_PROFILE")):
        return None
    if _EXTENSION_KEY in app.extensions:
        return _SESSION

    session = ProfilingSession(_resolve_output_path())
    session.attach()
    _SESSION = session
    app.extensions[_EXTENSION_KEY] = session
    atexit.register(session.detach)
    logger.info("Web request profiling enabled; writing to %s", session.output_path)

    @app.before_request
    def _profiling_before_request() -> None:
        if request.path in _SKIP_PATHS or request.path.startswith(_SKIP_PREFIXES):
            return
        g._abstracts_profile = session.start_request(
            method=request.method,
            path=request.path,
            query=request.query_string.decode("utf-8", "replace"),
            user_agent=request.headers.get("User-Agent", ""),
        )

    @app.after_request
    def _profiling_after_request(response: Response) -> Response:
        profile = getattr(g, "_abstracts_profile", None)
        if profile is None:
            return response
        profile.response_bytes = response.calculate_content_length() or 0
        response.headers["X-Request-ID"] = profile.request_id
        session.finish_request(profile, response.status_code)
        return response

    return session
