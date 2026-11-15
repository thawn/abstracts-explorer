"""
URL helpers shared across the package.
"""

from typing import Any, Mapping, Optional


def normalize_doi_url(value: str) -> str:
    """
    Normalize a DOI value or DOI URL into a canonical ``https://doi.org/`` URL.

    Parameters
    ----------
    value : str
        Bare DOI (``10.1234/foo``), a ``doi.org/``-prefixed path, or an
        ``http(s)://`` DOI URL.

    Returns
    -------
    str
        Canonical ``https://doi.org/...`` URL. Values that already start with
        ``https://`` are returned unchanged; ``http://`` is upgraded to
        ``https://``.

    Examples
    --------
    >>> normalize_doi_url("10.1234/abc")
    'https://doi.org/10.1234/abc'
    >>> normalize_doi_url("doi.org/10.1234/abc")
    'https://doi.org/10.1234/abc'
    >>> normalize_doi_url("http://doi.org/10.1234/abc")
    'https://doi.org/10.1234/abc'
    >>> normalize_doi_url("https://example.com/x")
    'https://example.com/x'
    """
    value = value.strip()
    if value.startswith("https://"):
        return value
    if value.startswith("http://"):
        return value.replace("http://", "https://", 1)
    prefix = "doi.org/"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    return f"https://doi.org/{value}"


def paper_pdf_url(paper: Mapping[str, Any]) -> Optional[str]:
    """
    Return the PDF URL for a paper, falling back to its OpenReview paper URL.

    Parameters
    ----------
    paper : mapping
        Paper record with optional ``paper_pdf_url`` and ``paper_url`` keys.

    Returns
    -------
    str or None
        ``paper_pdf_url`` when present; otherwise ``paper_url`` converted
        from an OpenReview forum URL to a PDF URL; ``None`` when neither is
        available.

    Examples
    --------
    >>> paper_pdf_url({"paper_url": "https://openreview.net/forum?id=abc"})
    'https://openreview.net/pdf?id=abc'
    >>> paper_pdf_url({"paper_pdf_url": "https://example.com/p.pdf"})
    'https://example.com/p.pdf'
    >>> paper_pdf_url({}) is None
    True
    """
    pdf_url = paper.get("paper_pdf_url")
    if not pdf_url and paper.get("paper_url"):
        pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
    return pdf_url or None
