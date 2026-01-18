"""
Export utilities for generating markdown and zip files from papers.

This module provides functions for:
- Natural sorting of strings with numbers
- Fetching conference information from websites
- Generating markdown files for papers
- Creating zip archives with organized paper exports
"""

import re
import logging
import zipfile
from io import BytesIO
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

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


def fetch_conference_info() -> Optional[Dict[str, Any]]:
    """
    Fetch conference information from NeurIPS website.

    Returns
    -------
    dict or None
        Conference information dictionary with keys: name, dates, location, description
        Returns None if fetching fails

    Examples
    --------
    >>> conf_info = fetch_conference_info()
    >>> if conf_info:
    ...     print(conf_info['name'], conf_info['dates'])
    """
    try:
        # Try to import BeautifulSoup
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("BeautifulSoup not installed, using fallback conference info")
            return None

        conference_url = "https://neurips.cc/Conferences/2025"
        response = requests.get(conference_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        conf_info = {
            "name": "38th Conference on Neural Information Processing Systems (NeurIPS 2025)",
            "dates": None,
            "location": None,
            "description": None,
        }

        # Try to find conference title
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if "NeurIPS" in title_text or "Neural Information Processing Systems" in title_text:
                conf_info["name"] = title_text

        # Try to find dates - look for patterns like "December 9-15, 2025"
        page_text = soup.get_text()
        date_patterns = [
            r"(December\s+\d+[-–]\d+,\s+\d{4})",
            r"(Dec\s+\d+[-–]\d+,\s+\d{4})",
            r"(\d{1,2}[-–]\d{1,2}\s+December\s+\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["dates"] = match.group(1)
                break

        # Try to find location - look for "Vancouver" mentions
        location_patterns = [
            r"(Vancouver\s+Convention\s+Centre(?:,\s+(?:Vancouver|BC|British\s+Columbia))?(?:,\s+Canada)?)",
            r"(Vancouver(?:,\s+(?:BC|British\s+Columbia))?(?:,\s+Canada)?)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["location"] = match.group(1)
                break

        # Try to find conference description/about text
        about_section = soup.find(["div", "p"], class_=re.compile(r"about|description", re.I))
        if about_section:
            desc_text = about_section.get_text(strip=True)
            # Limit description length
            if len(desc_text) > 500:
                desc_text = desc_text[:500] + "..."
            conf_info["description"] = desc_text

        logger.info(f"Fetched conference info: {conf_info}")
        return conf_info

    except Exception as e:
        logger.warning(f"Failed to fetch conference info from website: {e}")
        return None


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

    # Group by session
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(paper)

    # Write each session
    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
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

            # Link to PDF on OpenReview
            pdf_url = paper.get("paper_pdf_url")
            if not pdf_url and paper.get("paper_url"):
                pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
            if pdf_url:
                markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

            if paper.get("paper_url"):
                markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

            if paper.get("url"):
                markdown += f"**Source URL:** {paper['url']}\n\n"

            if paper.get("abstract"):
                markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

            # Link to poster image
            poster_url = get_poster_url(paper)
            if poster_url:
                markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

            markdown += "---\n\n"

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

    # Group by session
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(paper)

    # Write each session
    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            stars = "⭐" * paper.get("priority", 0)
            markdown += f"### {paper.get('title', 'Untitled')}\n\n"
            markdown += f"**Rating:** {stars} ({paper.get('priority', 0)}/5)\n\n"

            if paper.get("authors"):
                authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
                markdown += f"**Authors:** {authors}\n\n"

            if paper.get("poster_position"):
                markdown += f"**Poster:** {paper['poster_position']}\n\n"

            # Link to PDF on OpenReview
            pdf_url = paper.get("paper_pdf_url")
            if not pdf_url and paper.get("paper_url"):
                pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
            if pdf_url:
                markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

            if paper.get("paper_url"):
                markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

            if paper.get("url"):
                markdown += f"**Source URL:** {paper['url']}\n\n"

            if paper.get("abstract"):
                markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

            # Link to poster image
            poster_url = get_poster_url(paper)
            if poster_url:
                markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

            markdown += "---\n\n"

    return markdown


def generate_main_readme(
    papers: List[Dict[str, Any]], search_query: str, sort_order: str = "search-rating-poster"
) -> str:
    """
    Generate main README.md with conference overview and links to search term files.

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    search_query : str
        Search query context
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    str
        Markdown content for main README

    Examples
    --------
    >>> papers = [{"title": "Paper", "searchTerm": "AI", "priority": 3}]
    >>> readme = generate_main_readme(papers, "AI search", "search-rating-poster")
    >>> "NeurIPS 2025" in readme
    True
    """
    markdown = "# NeurIPS 2025 - Interesting Papers\n\n"

    markdown += "Generated by [Abstracts Explorer](https://github.com/thawn/neurips-abstracts)\n\n"

    # Conference information - fetch from website
    markdown += "## Conference Information\n\n"

    # Try to fetch conference info from neurips.cc
    conf_info = fetch_conference_info()

    if conf_info:
        markdown += f"**Conference:** {conf_info.get('name', '38th Conference on Neural Information Processing Systems (NeurIPS 2025)')}\n\n"
        if conf_info.get("dates"):
            markdown += f"**Dates:** {conf_info['dates']}\n\n"
        if conf_info.get("location"):
            markdown += f"**Location:** {conf_info['location']}\n\n"
        markdown += "**Website:** [https://neurips.cc/](https://neurips.cc/)\n\n"
        if conf_info.get("description"):
            markdown += f"**About:** {conf_info['description']}\n\n"
    else:
        # Fallback to static information if scraping fails
        markdown += "**Conference:** 38th Conference on Neural Information Processing Systems (NeurIPS 2025)\n\n"
        markdown += "**Dates:** December 9-15, 2025\n\n"
        markdown += "**Location:** Vancouver Convention Centre, Vancouver, Canada\n\n"
        markdown += "**Website:** [https://neurips.cc/](https://neurips.cc/)\n\n"
        markdown += (
            "**Topic:** Neural Information Processing Systems - Machine Learning and Artificial Intelligence\n\n"
        )

    markdown += "---\n\n"

    # Export metadata
    markdown += "## Export Information\n\n"
    markdown += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
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


def generate_folder_structure_export(
    papers: List[Dict[str, Any]], search_query: str, sort_order: str = "search-rating-poster"
) -> BytesIO:
    """
    Generate a zip file with folder structure respecting the sort order.

    File organization based on first sort priority:
    - search-rating-poster: Separate files per search term
    - rating-poster-search: Separate files per rating level
    - poster-search-rating: Single file with all papers (poster # is first)

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    search_query : str
        Search query context
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    BytesIO
        Buffer containing zip file

    Examples
    --------
    >>> papers = [{"title": "Paper", "searchTerm": "AI", "priority": 3}]
    >>> zip_buffer = generate_folder_structure_export(papers, "AI", "search-rating-poster")
    >>> zip_buffer.tell() > 0
    True
    """
    # Create in-memory zip file
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Generate main README.md with conference information
        readme_content = generate_main_readme(papers, search_query, sort_order)
        zipf.writestr("README.md", readme_content)

        if sort_order == "poster-search-rating":
            # Poster number is first priority - all papers in one single file
            all_papers_markdown = generate_all_papers_markdown(papers, "All Papers (by Poster #)")
            zipf.writestr("all_papers.md", all_papers_markdown)

        elif sort_order == "rating-poster-search":
            # Rating is first priority - split by rating
            priority_groups: Dict[int, List[Dict[str, Any]]] = {}
            for paper in papers:
                priority = paper.get("priority", 0)
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append(paper)

            # Create a markdown file for each priority rating
            for priority in sorted(priority_groups.keys(), reverse=True):  # Higher priorities first
                priority_papers = priority_groups[priority]
                priority_stars = "⭐" * priority
                priority_name = f"{priority}_stars" if priority > 0 else "0_stars"

                # Generate markdown for this priority level
                priority_markdown = generate_all_papers_markdown(
                    priority_papers, f"{priority_stars} {priority} Stars ({len(priority_papers)} papers)"
                )

                # Write to file named by priority
                zipf.writestr(f"{priority_name}.md", priority_markdown)

        else:  # sort_order == "search-rating-poster"
            # Search term is first priority - split by search term
            search_terms: Dict[str, List[Dict[str, Any]]] = {}
            for paper in papers:
                search_term = paper.get("searchTerm") or "Unknown"
                if search_term not in search_terms:
                    search_terms[search_term] = []
                search_terms[search_term].append(paper)

            # Create a markdown file for each search term
            for search_term, term_papers in search_terms.items():
                # Sanitize search term for filename
                safe_name = re.sub(r"[^\w\s-]", "", search_term).strip().replace(" ", "_")
                safe_name = safe_name[:50]  # Limit length

                if not safe_name:
                    safe_name = "unknown"

                # Generate markdown for this search term
                term_markdown = generate_search_term_markdown(search_term, term_papers)

                # Write to file named by search term
                zipf.writestr(f"{safe_name}.md", term_markdown)

    zip_buffer.seek(0)
    return zip_buffer
