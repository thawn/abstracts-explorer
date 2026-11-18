"""
Markdown generation for interesting-paper exports.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote, urlsplit

from abstracts_explorer.utils.time_utils import format_timestamp
from abstracts_explorer.utils.url_utils import paper_pdf_url

logger = logging.getLogger(__name__)

_MAX_FILENAME_LENGTH = 50
_DEFAULT_FILENAME = "unknown"


def _coerce_priority(value: Any) -> int:
    """Return a priority clamped to the supported 0-5 range."""
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(priority, 5))


def _escape_inline(value: Any) -> str:
    """Escape user-provided text for inline Markdown and table cells."""
    text = str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\r", " ").replace("\n", " ")


def _escape_block(value: Any) -> str:
    """Neutralize HTML in block text while preserving paragraph newlines."""
    text = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _safe_http_url(value: Any) -> Optional[str]:
    """Return an http(s) URL safe for Markdown, or ``None`` when invalid."""
    url = str(value).strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    return quote(url, safe=":/?#[]@!$&'*+,;=%")


def _sanitize_filename_base(value: Any, max_length: int = _MAX_FILENAME_LENGTH) -> str:
    """Sanitize a search term into a deterministic filename base."""
    safe_name = re.sub(r"[^\w\s-]", "", str(value)).strip().replace(" ", "_")
    safe_name = re.sub(r"[\r\n\t]+", "_", safe_name)
    safe_name = safe_name[:max_length]
    return safe_name or _DEFAULT_FILENAME


def build_search_term_filename_map(
    search_terms: Iterable[str], reserved: Iterable[str] = ("readme", "all_papers")
) -> Dict[str, str]:
    """
    Build collision-free Markdown filenames for search terms.

    Parameters
    ----------
    search_terms : iterable of str
        Search terms that need their own Markdown file.
    reserved : iterable of str, optional
        Filename stems that are already taken by other archive members.

    Returns
    -------
    dict
        Mapping from search term to its ``.md`` filename.
    """
    used = {str(name).lower() for name in reserved}
    filenames: Dict[str, str] = {}
    for search_term in sorted({str(term) for term in search_terms}):
        base = _sanitize_filename_base(search_term)
        candidate = base
        counter = 2
        while candidate.lower() in used:
            suffix = f"_{counter}"
            candidate = f"{base[: max(0, _MAX_FILENAME_LENGTH - len(suffix))]}{suffix}"
            counter += 1
        used.add(candidate.lower())
        filenames[search_term] = f"{candidate}.md"
    return filenames


def natural_sort_key(s: str) -> List:
    """
    Generate a sort key for natural sorting of strings with numbers.

    Parameters
    ----------
    s : str
        String to generate sort key for

    Returns
    -------
    list
        Sort key that enables natural number sorting

    Examples
    --------
    >>> sorted(["A10", "A2", "A1"], key=natural_sort_key)
    ['A1', 'A2', 'A10']
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r"(\d+)", s)]


def get_poster_url(paper: Dict[str, Any]) -> Optional[str]:
    """
    Get poster image URL from paper object.

    Parameters
    ----------
    paper : dict
        Paper object containing poster_image_url and original_id fields

    Returns
    -------
    str or None
        Poster image URL if found, None otherwise

    Examples
    --------
    >>> paper = {"poster_image_url": "https://example.com/poster.png"}
    >>> get_poster_url(paper)
    'https://example.com/poster.png'
    """
    # Use poster_image_url from database if available
    poster_image_url = paper.get("poster_image_url")
    if poster_image_url:
        return _safe_http_url(poster_image_url)

    # Fallback: construct poster URL from original_id
    original_id = paper.get("original_id")
    if original_id:
        url = f"https://{paper.get('conference', 'neurips').lower()}.cc/media/PosterPDFs/{paper.get('conference', 'NeurIPS')}%20{paper.get('year', '2025')}/{original_id}.png"
        return _safe_http_url(url)

    return None


def _group_papers_by_session(papers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group papers by their session name.

    Parameters
    ----------
    papers : list of dict
        Paper dictionaries to group.

    Returns
    -------
    dict
        Mapping of session name to papers. Papers without a session are placed
        under ``"No Session"``.
    """
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        sessions.setdefault(session, []).append(paper)
    return sessions


def _format_paper_block(paper: Dict[str, Any]) -> str:
    """
    Format a single paper as a markdown block.

    Parameters
    ----------
    paper : dict
        Paper dictionary to format.

    Returns
    -------
    str
        Markdown block for the paper, terminated by a horizontal rule.
    """
    markdown = ""
    priority = _coerce_priority(paper.get("priority", 0))
    stars = "⭐" * priority
    markdown += f"### {_escape_inline(paper.get('title', 'Untitled'))}\n\n"
    markdown += f"**Rating:** {stars} ({priority}/5)\n\n"

    if paper.get("searchTerm"):
        markdown += f"**Search Term:** {_escape_inline(paper.get('searchTerm'))}\n\n"

    if paper.get("authors"):
        authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
        markdown += f"**Authors:** {_escape_inline(authors)}\n\n"

    if paper.get("poster_position"):
        markdown += f"**Poster:** {_escape_inline(paper['poster_position'])}\n\n"

    pdf_url = _safe_http_url(paper_pdf_url(paper))
    if pdf_url:
        markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

    paper_url = _safe_http_url(paper.get("paper_url"))
    if paper_url:
        markdown += f"**Paper URL:** {paper_url}\n\n"

    source_url = _safe_http_url(paper.get("url"))
    if source_url:
        markdown += f"**Source URL:** {source_url}\n\n"

    if paper.get("abstract"):
        markdown += f"**Abstract:**\n\n{_escape_block(paper['abstract'])}\n\n"

    poster_url = get_poster_url(paper)
    if poster_url:
        markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

    markdown += "---\n\n"
    return markdown


def generate_all_papers_markdown(papers: List[Dict[str, Any]], title: str) -> str:
    """
    Generate markdown for all papers in a single file.

    Parameters
    ----------
    papers : list
        List of paper dictionaries
    title : str
        Title for the markdown file

    Returns
    -------
    str
        Markdown content

    Examples
    --------
    >>> papers = [{"title": "My Paper", "session": "Oral"}]
    >>> md = generate_all_papers_markdown(papers, "All Papers")
    >>> "# All Papers" in md
    True
    """
    markdown = f"# {_escape_inline(title)}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    sessions = _group_papers_by_session(papers)

    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {_escape_inline(session)}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            markdown += _format_paper_block(paper)

    return markdown


def generate_search_term_markdown(search_term: str, papers: List[Dict[str, Any]]) -> str:
    """
    Generate markdown for a single search term with all its papers.

    Parameters
    ----------
    search_term : str
        The search term
    papers : list
        List of papers for this search term

    Returns
    -------
    str
        Markdown content

    Examples
    --------
    >>> papers = [{"title": "My Paper", "session": "Oral"}]
    >>> md = generate_search_term_markdown("transformers", papers)
    >>> "# transformers" in md
    True
    """
    markdown = f"# {_escape_inline(search_term)}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    sessions = _group_papers_by_session(papers)

    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {_escape_inline(session)}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            markdown += _format_paper_block(paper)

    return markdown


def generate_main_readme(
    papers: List[Dict[str, Any]],
    sort_order: str = "search-rating-poster",
    search_term_files: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Generate main README.md with conference overview and links to search term files.

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')
    search_term_files : mapping, optional
        Mapping of search term to the Markdown filename written to the archive.
        When omitted, filenames are derived from the search terms.

    Returns
    -------
    str
        Markdown content for main README

    Examples
    --------
    >>> papers = [{"title": "Paper", "searchTerm": "AI", "priority": 3}]
    >>> readme = generate_main_readme(papers, "search-rating-poster")
    >>> "Interesting Papers" in readme
    True
    """
    conferences = sorted({str(p["conference"]) for p in papers if p.get("conference")})
    years = sorted({int(p["year"]) for p in papers if p.get("year")})
    if len(conferences) == 1 and len(years) == 1:
        title = f"# {_escape_inline(conferences[0])} {years[0]} - Interesting Papers"
    elif conferences:
        title = f"# Interesting Papers - {_escape_inline(', '.join(str(c) for c in conferences))}"
    else:
        title = "# Interesting Papers"
    markdown = title + "\n\n"

    markdown += "Generated by [Abstracts Explorer](https://github.com/thawn/abstracts-explorer)\n\n"

    markdown += "## Conference Information\n\n"

    if conferences:
        markdown += f"**Conference:** {_escape_inline(', '.join(str(c) for c in conferences))}\n\n"
    else:
        markdown += "**Conference:** Unknown\n\n"
    if years:
        markdown += f"**Year:** {_escape_inline(', '.join(str(y) for y in years))}\n\n"

    markdown += "---\n\n"

    # Export metadata
    markdown += "## Export Information\n\n"
    markdown += f"**Generated:** {format_timestamp(datetime.now(), '%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"**Total Papers:** {len(papers)}\n\n"

    # Document sort order
    sort_order_descriptions = {
        "search-rating-poster": "Search Term → Rating → Poster #",
        "rating-poster-search": "Rating → Poster # → Search Term",
        "poster-search-rating": "Poster # → Search Term → Rating",
    }
    sort_desc = sort_order_descriptions.get(sort_order, sort_order)
    markdown += f"**Sort Order:** {sort_desc}\n\n"

    markdown += "---\n\n"

    # Group papers by search term and session
    search_terms: Dict[str, Dict[str, Any]] = {}
    for paper in papers:
        search_term = str(paper.get("searchTerm") or "Unknown")
        if search_term not in search_terms:
            search_terms[search_term] = {"count": 0, "sessions": set(), "avg_priority": 0, "priorities": []}
        search_terms[search_term]["count"] += 1
        search_terms[search_term]["priorities"].append(_coerce_priority(paper.get("priority", 0)))
        if paper.get("session"):
            search_terms[search_term]["sessions"].add(str(paper["session"]))

    # Calculate averages
    for term_data in search_terms.values():
        if term_data["priorities"]:
            term_data["avg_priority"] = sum(term_data["priorities"]) / len(term_data["priorities"])

    # Generate table of contents based on sort order
    if sort_order == "poster-search-rating":
        # Single file with all papers
        markdown += "## Papers\n\n"
        markdown += "All papers are organized in a single file: [View All Papers](all_papers.md)\n\n"
        markdown += f"**Total Papers:** {len(papers)}\n\n"

        # Still show search terms summary
        markdown += "### Search Terms Summary\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating |\n"
        markdown += "|-------------|--------|----------|------------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * _coerce_priority(round(term_data["avg_priority"]))
            markdown += f"| {_escape_inline(search_term)} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    elif sort_order == "rating-poster-search":
        # Files organized by rating
        markdown += "## Papers by Rating\n\n"

        # Group by priority
        priority_groups: Dict[int, Dict[str, Any]] = {}
        for paper in papers:
            priority = _coerce_priority(paper.get("priority", 0))
            if priority not in priority_groups:
                priority_groups[priority] = {"count": 0, "search_terms": set()}
            priority_groups[priority]["count"] += 1
            priority_groups[priority]["search_terms"].add(str(paper.get("searchTerm") or "Unknown"))

        markdown += "| Rating | Papers | Search Terms | File |\n"
        markdown += "|--------|--------|--------------|------|\n"

        for priority in sorted(priority_groups.keys(), reverse=True):
            priority_data = priority_groups[priority]
            priority_stars = "⭐" * priority
            priority_name = f"{priority}_stars" if priority > 0 else "0_stars"
            search_terms_str = ", ".join(sorted(priority_data["search_terms"]))
            if len(search_terms_str) > 50:
                search_terms_str = search_terms_str[:50] + "..."
            markdown += f"| {priority_stars} {priority}/5 | {priority_data['count']} | {_escape_inline(search_terms_str)} | [{priority_name}.md]({priority_name}.md) |\n"

        markdown += "\n### Search Terms Summary\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating |\n"
        markdown += "|-------------|--------|----------|------------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * _coerce_priority(round(term_data["avg_priority"]))
            markdown += f"| {_escape_inline(search_term)} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    else:  # search-rating-poster
        # Files organized by search term
        if search_term_files is None:
            search_term_files = build_search_term_filename_map(search_terms.keys())
        markdown += "## Papers by Search Term\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating | File |\n"
        markdown += "|-------------|--------|----------|------------|------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            filename = search_term_files[search_term]

            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * _coerce_priority(round(term_data["avg_priority"]))

            markdown += f"| [{_escape_inline(search_term)}]({filename}) | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) | {filename} |\n"

    markdown += "\n---\n\n"

    # Session overview
    sessions: Dict[str, Dict[str, Any]] = {}
    for paper in papers:
        session = str(paper.get("session") or "No Session")
        if session not in sessions:
            sessions[session] = {"count": 0, "search_terms": set()}
        sessions[session]["count"] += 1
        sessions[session]["search_terms"].add(str(paper.get("searchTerm") or "Unknown"))

    markdown += "## Sessions Overview\n\n"
    for session in sorted(sessions.keys()):
        session_data = sessions[session]
        markdown += f"### {_escape_inline(session)}\n\n"
        markdown += f"- **Papers:** {session_data['count']}\n"
        search_terms_list = ", ".join(_escape_inline(term) for term in sorted(session_data["search_terms"]))
        markdown += f"- **Search Terms:** {search_terms_list}\n\n"

    return markdown
