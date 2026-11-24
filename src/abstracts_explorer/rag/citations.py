"""
Stable citation numbering for papers returned by RAG tools.
"""

import threading
from typing import Any, Dict, List

from abstracts_explorer.paper_utils import paper_citation_key

_CITATION_PAPER_FIELDS: Dict[str, tuple[str, ...]] = {
    "search_papers": ("papers",),
    "get_paper_details": ("papers",),
    "recommend_related_papers": ("papers",),
    "summarize_interesting_papers": ("papers",),
    "analyze_topic_relevance": ("sample_papers", "papers"),
}


class PaperCitationRegistry:
    """
    Assign one stable ``Paper-N`` number to each paper in a conversation.

    Tools may return overlapping papers across multiple search calls.  The
    registry keeps the first-seen order, deduplicates by :func:`paper_citation_key`,
    and exposes the same numbering to both the LLM tool output and the web UI.
    """

    def __init__(self) -> None:
        self._papers: List[Dict[str, Any]] = []
        self._numbers: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def papers(self) -> List[Dict[str, Any]]:
        """
        Return a snapshot of the registered papers in citation order.

        Returns
        -------
        list of dict
            Paper records, each including its ``citation_number``.
        """
        with self._lock:
            return [dict(paper) for paper in self._papers]

    def register(self, tool_name: str, result: Any) -> Dict[str, int]:
        """
        Register papers from one tool result and return their citation numbers.

        Parameters
        ----------
        tool_name : str
            Name of the tool that produced *result*.
        result : Any
            Parsed tool result.  Non-dict values are ignored.

        Returns
        -------
        dict
            Mapping from :func:`paper_citation_key` to global citation number
            for every paper in the result.
        """
        if not isinstance(result, dict):
            return {}

        mapping: Dict[str, int] = {}
        with self._lock:
            for field in _CITATION_PAPER_FIELDS.get(tool_name, ()):
                papers = result.get(field, [])
                if not isinstance(papers, list):
                    continue
                for paper in papers:
                    if not isinstance(paper, dict):
                        continue
                    key = paper_citation_key(paper)
                    if not key:
                        continue
                    number = self._numbers.get(key)
                    if number is None:
                        number = len(self._papers) + 1
                        stored = dict(paper)
                        stored["citation_number"] = number
                        self._papers.append(stored)
                        self._numbers[key] = number
                    mapping[key] = number
        return mapping

    def clear(self) -> None:
        """Remove all registered papers and their numbers."""
        with self._lock:
            self._papers.clear()
            self._numbers.clear()
