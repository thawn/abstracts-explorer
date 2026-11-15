"""
Conference name and year parsing helpers shared across the package.
"""

import re
from typing import Optional, Tuple


def parse_conference_year(conference: str) -> Tuple[str, Optional[int]]:
    """
    Split a trailing year from a conference name.

    Handles the combined ``"conference year"`` forms LLMs and users
    typically produce (e.g. ``"NeurIPS 2025"`` or ``"ICLR'26"``) so the
    conference name matches database and cache entries, which store name
    and year separately.

    Parameters
    ----------
    conference : str
        Conference name, possibly with a trailing 4- or 2-digit year.

    Returns
    -------
    tuple of (str, int or None)
        ``(conference_name, year)`` — *year* is ``None`` when no trailing
        year was found; 2-digit years are expanded to the 2000s.

    Examples
    --------
    >>> parse_conference_year("NeurIPS 2025")
    ('NeurIPS', 2025)
    >>> parse_conference_year("ICLR'26")
    ('ICLR', 2026)
    >>> parse_conference_year("ICLR")
    ('ICLR', None)
    """
    match = re.match(r"^(.+?)[\s']+(\d{4}|\d{2})$", conference.strip())
    if not match:
        return conference.strip(), None

    name = match.group(1).strip()
    year = int(match.group(2))
    if year < 100:
        year += 2000
    return name, year
