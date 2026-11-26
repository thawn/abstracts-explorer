"""
Schema conversion utilities.

Converts papers in the full/EventHosts schema to the lightweight schema.
"""

from typing import Any, Dict, List

from abstracts_explorer.plugin.models import (
    deserialize_authors_from_string,
    deserialize_keywords_from_string,
    sanitize_author_names,
)


def convert_to_lightweight_schema(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert full NeurIPS schema to lightweight paper format.

    This function extracts only the fields needed for the lightweight schema
    from papers in the full NeurIPS format, making it easier to work with
    simplified data structures.

    Parameters
    ----------
    papers : list
        List of papers in full NeurIPS format with fields like:
        - id (int)
        - title or name (str) - will use 'title', fallback to 'name'
        - authors (list of dict with 'fullname' or list of str)
        - abstract (str)
        - session (str)
        - poster_position (str)
        - paper_pdf_url (str, optional)
        - poster_image_url (str, optional)
        - url (str, optional)
        - room_name (str, optional)
        - keywords (list or str, optional)
        - starttime (str, optional)
        - endtime (str, optional)
        - award or decision (str, optional)
        - year (int, optional)
        - conference (str, optional)

    Returns
    -------
    list of dict
        Papers in lightweight format with only essential fields.
        Authors are returned as lists of strings.

    Examples
    --------
    >>> papers = [
    ...     {
    ...         'id': 123,
    ...         'title': 'Deep Learning',
    ...         'authors': [
    ...             {'id': 1, 'fullname': 'John Doe', 'institution': 'MIT'},
    ...             {'id': 2, 'fullname': 'Jane Smith', 'institution': 'Stanford'}
    ...         ],
    ...         'abstract': 'A paper about deep learning',
    ...         'session': 'Session A',
    ...         'poster_position': 'A-42',
    ...         'paper_pdf_url': 'https://example.com/paper.pdf',
    ...         'year': 2025,
    ...         'conference': 'NeurIPS'
    ...     }
    ... ]
    >>> lightweight = convert_to_lightweight_schema(papers)
    >>> lightweight[0]['authors']
    ['John Doe', 'Jane Smith']

    Notes
    -----
    - Author objects are converted to lists of name strings
    - Extra NeurIPS-specific fields are dropped
    - 'name' field is converted to 'title' if needed
    - Keywords are converted from string to list if needed
    """
    lightweight_papers = []

    for paper in papers:
        title = paper.get("title") or paper.get("name", "")
        if not title:
            continue

        authors_data = paper.get("authors", [])
        authors = []

        if isinstance(authors_data, list):
            for author in authors_data:
                if isinstance(author, dict):
                    name = author.get("fullname") or author.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)
        elif isinstance(authors_data, str):
            authors = deserialize_authors_from_string(authors_data)

        authors = sanitize_author_names(authors)

        lightweight_paper = {
            "title": title,
            "authors": authors,
            "abstract": paper.get("abstract") or "",
            "session": paper.get("session") or "No session",
            "poster_position": paper.get("poster_position") or "",
            "year": paper.get("year") or 0,
            "conference": paper.get("conference") or "",
        }

        if "id" in paper:
            lightweight_paper["original_id"] = paper["id"]

        if paper.get("paper_pdf_url"):
            lightweight_paper["paper_pdf_url"] = paper["paper_pdf_url"]
        elif paper.get("paper_url"):
            lightweight_paper["paper_pdf_url"] = paper["paper_url"]

        if paper.get("poster_image_url"):
            lightweight_paper["poster_image_url"] = paper["poster_image_url"]

        if paper.get("url"):
            lightweight_paper["url"] = paper["url"]

        if paper.get("room_name"):
            lightweight_paper["room_name"] = paper["room_name"]

        if paper.get("location"):
            lightweight_paper["location"] = paper["location"]

        keywords = paper.get("keywords")
        if keywords:
            if isinstance(keywords, str):
                lightweight_paper["keywords"] = deserialize_keywords_from_string(keywords)
            elif isinstance(keywords, list):
                lightweight_paper["keywords"] = keywords

        if paper.get("starttime"):
            lightweight_paper["starttime"] = paper["starttime"]

        if paper.get("endtime"):
            lightweight_paper["endtime"] = paper["endtime"]

        decision = paper.get("decision") or ""
        award = paper.get("award") or (paper.get("decision") if "award" in decision.lower() else None)
        if award:
            lightweight_paper["award"] = award

        lightweight_papers.append(lightweight_paper)

    return lightweight_papers
