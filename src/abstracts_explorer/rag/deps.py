"""
RAG agent dependencies.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    paper_priorities : dict
        Paper ratings from the web UI (``localStorage.paperPriorities``).
        Keys are paper UIDs; values are either star ratings or dicts with
        ``priority`` and ``searchTerm`` fields.
    tool_cache : dict
        Raw results of tool calls already executed during this query, keyed by
        tool name and canonical arguments.  Used to coalesce duplicate calls.
    tool_cache_lock : threading.Lock
        Guards ``tool_cache`` when tool calls run in parallel.
    """

    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    conferences: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    n_results: Optional[int] = None
    paper_priorities: Dict[str, Any] = field(default_factory=dict)
    tool_cache: Dict[str, str] = field(default_factory=dict)
    tool_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
