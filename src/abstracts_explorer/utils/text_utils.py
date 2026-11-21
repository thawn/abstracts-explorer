"""
Text-normalization helpers shared across the package.
"""

from typing import Any

from bs4 import BeautifulSoup


def html_to_text(value: Any) -> str:
    """
    Strip HTML tags, unescape entities, and normalize whitespace.

    Parameters
    ----------
    value : Any
        Text or HTML fragment.  ``None`` is treated as an empty string.

    Returns
    -------
    str
        Plain text with tags removed, HTML entities decoded, and repeated
        whitespace collapsed.

    Examples
    --------
    >>> html_to_text("<b>Data Management</b> Makes &amp; ML Easier")
    'Data Management Makes & ML Easier'
    >>> html_to_text(None)
    ''
    """
    if value is None:
        return ""

    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())
