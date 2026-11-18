"""
ZIP archive generation for interesting-paper exports.
"""

import logging
import zipfile
from io import BytesIO
from typing import Any, Dict, List

from abstracts_explorer.export.markdown import (
    _coerce_priority,
    build_search_term_filename_map,
    generate_all_papers_markdown,
    generate_main_readme,
    generate_search_term_markdown,
    natural_sort_key,
)

logger = logging.getLogger(__name__)


def generate_folder_structure_export(
    papers: List[Dict[str, Any]], sort_order: str = "search-rating-poster"
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
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    BytesIO
        Buffer containing zip file

    Examples
    --------
    >>> papers = [{"title": "Paper", "searchTerm": "AI", "priority": 3}]
    >>> zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")
    >>> zip_buffer.tell() > 0
    True
    """
    # Create in-memory zip file
    zip_buffer = BytesIO()

    search_terms: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers:
        search_term = str(paper.get("searchTerm") or "Unknown")
        search_terms.setdefault(search_term, []).append(paper)

    search_term_files = build_search_term_filename_map(search_terms.keys())

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Generate main README.md with conference information
        readme_content = generate_main_readme(papers, sort_order, search_term_files=search_term_files)
        zipf.writestr("README.md", readme_content)

        if sort_order == "poster-search-rating":
            # Poster number is first priority - all papers in one single file
            all_papers_markdown = generate_all_papers_markdown(papers, "All Papers (by Poster #)")
            zipf.writestr("all_papers.md", all_papers_markdown)

        elif sort_order == "rating-poster-search":
            # Rating is first priority - split by rating
            priority_groups: Dict[int, List[Dict[str, Any]]] = {}
            for paper in papers:
                priority = _coerce_priority(paper.get("priority", 0))
                priority_groups.setdefault(priority, []).append(paper)

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
            for search_term in sorted(search_terms.keys()):
                term_papers = search_terms[search_term]

                # Generate markdown for this search term
                term_markdown = generate_search_term_markdown(search_term, term_papers)

                # Write to file named by search term
                zipf.writestr(search_term_files[search_term], term_markdown)

    zip_buffer.seek(0)
    return zip_buffer


def export_papers_to_zip(papers: List[Dict[str, Any]], sort_order: str = "search-rating-poster") -> BytesIO:
    """
    Export papers with priorities to a ZIP file, handling sorting and formatting.

    This is a convenience function that handles the complete export workflow:
    - Sorts papers based on the specified sort order
    - Generates the ZIP file structure
    - Returns the buffer ready for sending

    Parameters
    ----------
    papers : list
        List of paper dictionaries with priority and searchTerm fields
    sort_order : str, optional
        Sort order: 'search-rating-poster', 'rating-poster-search', or 'poster-search-rating'

    Returns
    -------
    BytesIO
        Buffer containing the generated ZIP file

    Examples
    --------
    >>> papers = [{"title": "Paper 1", "priority": 5, "searchTerm": "AI"}]
    >>> zip_buffer = export_papers_to_zip(papers, "search-rating-poster")
    """
    # Sort papers based on the selected sort order without mutating the caller's list
    if sort_order == "search-rating-poster":
        # Search term, then priority, then poster position
        papers = sorted(
            papers,
            key=lambda p: (
                str(p.get("searchTerm") or ""),
                -_coerce_priority(p.get("priority", 0)),  # Descending priority
                natural_sort_key(p.get("poster_position") or ""),
            ),
        )
    elif sort_order == "rating-poster-search":
        # Priority, then poster position, then search term
        papers = sorted(
            papers,
            key=lambda p: (
                -_coerce_priority(p.get("priority", 0)),  # Descending priority
                natural_sort_key(p.get("poster_position") or ""),
                str(p.get("searchTerm") or ""),
            ),
        )
    elif sort_order == "poster-search-rating":
        # Poster position, then search term, then priority
        papers = sorted(
            papers,
            key=lambda p: (
                natural_sort_key(p.get("poster_position") or ""),
                str(p.get("searchTerm") or ""),
                -_coerce_priority(p.get("priority", 0)),  # Descending priority
            ),
        )
    else:
        # Default: search term, priority, poster position
        papers = sorted(
            papers,
            key=lambda p: (
                str(p.get("searchTerm") or ""),
                -_coerce_priority(p.get("priority", 0)),  # Descending priority
                natural_sort_key(p.get("poster_position") or ""),
            ),
        )

    # Generate zip file with folder structure
    return generate_folder_structure_export(papers, sort_order)
