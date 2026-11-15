"""
Markdown generation for interesting-paper exports.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from abstracts_explorer.export.conference_info import fetch_conference_info
from abstracts_explorer.utils.time_utils import format_timestamp
from abstracts_explorer.utils.url_utils import paper_pdf_url

logger = logging.getLogger(__name__)


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
        return poster_image_url

    # Fallback: construct poster URL from original_id
    original_id = paper.get("original_id")
    if original_id:
        return f"https://{paper.get('conference', 'neurips').lower()}.cc/media/PosterPDFs/{paper.get('conference', 'NeurIPS')}%20{paper.get('year', '2025')}/{original_id}.png"

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
    stars = "⭐" * paper.get("priority", 0)
    markdown += f"### {paper.get('title', 'Untitled')}\n\n"
    markdown += f"**Rating:** {stars} ({paper.get('priority', 0)}/5)\n\n"

    if paper.get("searchTerm"):
        markdown += f"**Search Term:** {paper.get('searchTerm')}\n\n"

    if paper.get("authors"):
        authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
        markdown += f"**Authors:** {authors}\n\n"

    if paper.get("poster_position"):
        markdown += f"**Poster:** {paper['poster_position']}\n\n"

    pdf_url = paper_pdf_url(paper)
    if pdf_url:
        markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

    if paper.get("paper_url"):
        markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

    if paper.get("url"):
        markdown += f"**Source URL:** {paper['url']}\n\n"

    if paper.get("abstract"):
        markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

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
    markdown = f"# {title}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    sessions = _group_papers_by_session(papers)

    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
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
    markdown = f"# {search_term}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    sessions = _group_papers_by_session(papers)

    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            markdown += _format_paper_block(paper)

    return markdown


def generate_main_readme(papers: List[Dict[str, Any]], sort_order: str = "search-rating-poster") -> str:
    """
    Generate main README.md with conference overview and links to search term files.

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    str
        Markdown content for main README

    Examples
    --------
    >>> papers = [{"title": "Paper", "searchTerm": "AI", "priority": 3}]
    >>> readme = generate_main_readme(papers, "search-rating-poster")
    >>> "NeurIPS 2025" in readme
    True
    """
    conferences = sorted({str(p["conference"]) for p in papers if p.get("conference")})
    years = sorted({int(p["year"]) for p in papers if p.get("year")})
    if len(conferences) == 1 and len(years) == 1:
        title = f"# {conferences[0]} {years[0]} - Interesting Papers"
    elif conferences:
        title = f"# Interesting Papers - {', '.join(str(c) for c in conferences)}"
    else:
        title = "# Interesting Papers"
    markdown = title + "\n\n"

    markdown += "Generated by [Abstracts Explorer](https://github.com/thawn/abstracts-explorer)\n\n"

    markdown += "## Conference Information\n\n"

    if len(conferences) == 1 and conferences[0].lower() == "neurips":
        conf_info = fetch_conference_info()
        if conf_info:
            markdown += f"**Conference:** {conf_info.get('name', conferences[0])}\n\n"
            if conf_info.get("dates"):
                markdown += f"**Dates:** {conf_info['dates']}\n\n"
            if conf_info.get("location"):
                markdown += f"**Location:** {conf_info['location']}\n\n"
            markdown += "**Website:** [https://neurips.cc/](https://neurips.cc/)\n\n"
            if conf_info.get("description"):
                markdown += f"**About:** {conf_info['description']}\n\n"
        else:
            markdown += f"**Conference:** {conferences[0]}\n\n"
    elif conferences:
        markdown += f"**Conference:** {', '.join(str(c) for c in conferences)}\n\n"
    else:
        markdown += "**Conference:** Unknown\n\n"

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
        search_term = paper.get("searchTerm") or "Unknown"
        if search_term not in search_terms:
            search_terms[search_term] = {"count": 0, "sessions": set(), "avg_priority": 0, "priorities": []}
        search_terms[search_term]["count"] += 1
        search_terms[search_term]["priorities"].append(paper.get("priority", 0))
        if paper.get("session"):
            search_terms[search_term]["sessions"].add(paper["session"])

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
            avg_stars = "⭐" * round(term_data["avg_priority"])
            markdown += f"| {search_term} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    elif sort_order == "rating-poster-search":
        # Files organized by rating
        markdown += "## Papers by Rating\n\n"

        # Group by priority
        priority_groups: Dict[int, Dict[str, Any]] = {}
        for paper in papers:
            priority = paper.get("priority", 0)
            if priority not in priority_groups:
                priority_groups[priority] = {"count": 0, "search_terms": set()}
            priority_groups[priority]["count"] += 1
            priority_groups[priority]["search_terms"].add(paper.get("searchTerm") or "Unknown")

        markdown += "| Rating | Papers | Search Terms | File |\n"
        markdown += "|--------|--------|--------------|------|\n"

        for priority in sorted(priority_groups.keys(), reverse=True):
            priority_data = priority_groups[priority]
            priority_stars = "⭐" * priority
            priority_name = f"{priority}_stars" if priority > 0 else "0_stars"
            search_terms_str = ", ".join(sorted(priority_data["search_terms"]))
            if len(search_terms_str) > 50:
                search_terms_str = search_terms_str[:50] + "..."
            markdown += f"| {priority_stars} {priority}/5 | {priority_data['count']} | {search_terms_str} | [{priority_name}.md]({priority_name}.md) |\n"

        markdown += "\n### Search Terms Summary\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating |\n"
        markdown += "|-------------|--------|----------|------------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * round(term_data["avg_priority"])
            markdown += f"| {search_term} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    else:  # search-rating-poster
        # Files organized by search term
        markdown += "## Papers by Search Term\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating | File |\n"
        markdown += "|-------------|--------|----------|------------|------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            safe_name = re.sub(r"[^\w\s-]", "", search_term).strip().replace(" ", "_")
            safe_name = safe_name[:50]
            if not safe_name:
                safe_name = "unknown"

            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * round(term_data["avg_priority"])

            markdown += f"| [{search_term}]({safe_name}.md) | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) | {safe_name}.md |\n"

    markdown += "\n---\n\n"

    # Session overview
    sessions: Dict[str, Dict[str, Any]] = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = {"count": 0, "search_terms": set()}
        sessions[session]["count"] += 1
        sessions[session]["search_terms"].add(paper.get("searchTerm") or "Unknown")

    markdown += "## Sessions Overview\n\n"
    for session in sorted(sessions.keys()):
        session_data = sessions[session]
        markdown += f"### {session}\n\n"
        markdown += f"- **Papers:** {session_data['count']}\n"
        markdown += f"- **Search Terms:** {', '.join(sorted(session_data['search_terms']))}\n\n"

    return markdown
