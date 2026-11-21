"""
ORCID-based search-term suggestion tool.
"""

import json
import logging
from typing import Optional

from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.mcp.server import mcp
from abstracts_explorer.orcid import OrcidError
from abstracts_explorer.orcid import suggest_search_terms_from_orcid as _suggest_search_terms_from_orcid

logger = logging.getLogger(__name__)


@mcp.tool()
def suggest_search_terms_from_orcid(
    orcid: str,
    max_terms: int = 15,
    min_year: Optional[int] = None,
    max_works: int = 100,
) -> str:
    """
    Suggest paper search terms based on a researcher's public ORCID record.

    Use this tool when the user provides their ORCID iD (or ORCID profile URL)
    and asks what to search for at a conference.  The tool reads the public
    ORCID works and author keywords, then asks the configured chat model for
    concise semantic-search terms.  If the model is unavailable, it falls back
    to ORCID keywords and phrases extracted from work titles.

    The ORCID record must be public; private works are not visible to the API.

    Parameters
    ----------
    orcid : str
        ORCID iD or ORCID profile URL (e.g. ``"0000-0002-1825-0097"`` or
        ``"https://orcid.org/0000-0002-1825-0097"``).
    max_terms : int, optional
        Maximum number of search terms to return (default: 15, capped at 50).
    min_year : int, optional
        Only consider works published in or after this year.
    max_works : int, optional
        Maximum number of works to send to the LLM (default: 100, capped at 500).

    Returns
    -------
    str
        JSON string containing:

        - ``orcid`` – normalized ORCID iD
        - ``name`` – display name from the public record
        - ``works_found`` – total public works on the record
        - ``works_used`` – works considered after filtering and capping
        - ``keywords`` – ORCID author keywords
        - ``suggested_terms`` – list of dicts with ``term``, ``reason``, and ``source``
        - ``recent_titles`` – up to ten recent work titles
        - ``source`` – ``"llm"``, ``"fallback"``, or ``"none"``
        - ``warning`` – optional explanation when no terms could be generated

    Examples
    --------
    >>> result = suggest_search_terms_from_orcid("0000-0002-1825-0097")
    >>> json.loads(result)["suggested_terms"][0]["term"]
    'psychoceramics'
    """
    try:
        result = _suggest_search_terms_from_orcid(
            orcid,
            max_terms=max_terms,
            min_year=min_year,
            max_works=max_works,
        )
        return json.dumps(result, indent=2, default=str)
    except OrcidError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to suggest search terms from ORCID: %s", exc)
        raise ToolError("ORCID search-term suggestion failed; check server logs for details.") from exc
