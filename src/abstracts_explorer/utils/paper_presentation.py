"""
Presentation helpers for paper text in API, CLI, and export output.
"""


def truncate_abstract(text: str, max_length: int) -> str:
    """
    Truncate text with an ellipsis when it exceeds *max_length*.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_length : int
        Maximum number of characters kept before the ellipsis.

    Returns
    -------
    str
        *text* itself when it fits within *max_length* characters,
        otherwise the first *max_length* characters followed by ``"..."``.

    Examples
    --------
    >>> truncate_abstract("short", 10)
    'short'
    >>> truncate_abstract("a" * 12, 10)
    'aaaaaaaaaa...'
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
