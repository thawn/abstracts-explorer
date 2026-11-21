"""
ORCID public-record integration.

Fetches a researcher's public ORCID works, keywords, and display name, then
asks the configured chat model for concise paper search terms.  When the LLM is
unavailable or returns unusable output, the service falls back to ORCID author
keywords and TF-IDF phrases extracted from work titles.
"""

import html
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

import requests

from abstracts_explorer.config import get_config
from abstracts_explorer.embeddings.models import ModelMixin
from abstracts_explorer.llm_utils import chat_completion_text
from abstracts_explorer.metadata_cache import MetadataCache
from abstracts_explorer.paper_utils import extract_top_keywords

logger = logging.getLogger(__name__)

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
ORCID_ID_PATTERN = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
ORCID_USER_AGENT = "abstracts-explorer ORCID client"
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

_ORCID_CACHE = MetadataCache()

_LLM_SYSTEM_PROMPT = """You are an expert academic research assistant.
Suggest concise paper search terms for a researcher based on their public ORCID record.
Return only a JSON object with this shape:
{"search_terms": [{"term": "short search query", "reason": "why it fits"}]}

Rules:
- Each term must be 1-4 words and suitable as a semantic paper search query.
- Prefer research topics, methods, and application areas.
- Do not include author names, venue names, or generic words such as "paper".
- Return at most {max_terms} terms.
"""


class OrcidError(Exception):
    """Exception raised for ORCID lookup and parsing failures."""


def _clean_text(value: Any) -> str:
    """Unescape HTML entities, strip tags, and normalize whitespace."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _HTML_TAG_PATTERN.sub("", text)
    return " ".join(text.split())


class _OrcidLlmProvider(ModelMixin):
    """Minimal ``ModelMixin`` adapter that supplies a chat LLM client."""

    def __init__(self) -> None:
        config = get_config()
        self._openai_client = None
        self._init_lock = threading.Lock()
        self.backend_url = config.llm_backend_url.rstrip("/")
        self.llm_backend_auth_token = config.llm_backend_auth_token
        self.model_name = config.chat_model
        self.requests_per_minute = config.requests_per_minute
        self.request_timeout = config.llm_request_timeout
        self.connect_timeout = config.llm_connect_timeout


def normalize_orcid_id(value: str) -> str:
    """
    Normalize an ORCID iD or ORCID URL to the bare ``0000-...`` form.

    Parameters
    ----------
    value : str
        Bare ORCID iD, ``orcid.org/...`` path, or ``http(s)://orcid.org/...`` URL.

    Returns
    -------
    str
        Normalized ORCID iD with an uppercase check digit.

    Raises
    ------
    OrcidError
        If *value* is not a valid ORCID iD.

    Examples
    --------
    >>> normalize_orcid_id("https://orcid.org/0000-0002-1825-0097")
    '0000-0002-1825-0097'
    >>> normalize_orcid_id("0000-0002-1825-009x")
    '0000-0002-1825-009X'
    """
    if not isinstance(value, str):
        raise OrcidError("An ORCID iD must be provided as a string.")

    candidate = value.strip()
    if not candidate:
        raise OrcidError("An ORCID iD must be provided.")

    if "orcid.org/" in candidate:
        candidate = candidate.split("orcid.org/", 1)[1].split("/", 1)[0]

    candidate = candidate.upper()
    if not ORCID_ID_PATTERN.match(candidate):
        raise OrcidError(f"Invalid ORCID iD: {value!r}. Expected a form like 0000-0002-1825-0097.")

    return candidate


def _get_json(url: str, *, timeout: float) -> Dict[str, Any]:
    """Fetch one ORCID API endpoint and return its JSON object."""
    headers = {
        "Accept": "application/json",
        "User-Agent": ORCID_USER_AGENT,
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise OrcidError(f"Could not reach the ORCID API at {url}: {exc}") from exc

    if response.status_code == 404:
        raise OrcidError(f"ORCID record not found at {url}.")
    if response.status_code != 200:
        raise OrcidError(f"ORCID API returned HTTP {response.status_code} for {url}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise OrcidError(f"ORCID API returned malformed JSON for {url}.") from exc

    if not isinstance(payload, dict):
        raise OrcidError(f"ORCID API returned unexpected JSON for {url}.")
    return payload


def _extract_doi(summary: Dict[str, Any]) -> Optional[str]:
    """Return the first DOI from a work summary, if present."""
    external_ids = (summary.get("external-ids") or {}).get("external-id") or []
    for item in external_ids:
        if item.get("external-id-type") == "doi":
            value = item.get("external-id-value")
            if value:
                return str(value)
    return None


def _parse_works(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse an ORCID ``/works`` response into compact work records."""
    works: List[Dict[str, Any]] = []
    seen: set[tuple[str, Optional[int]]] = set()

    for group in payload.get("group") or []:
        for summary in group.get("work-summary") or []:
            title = _clean_text(((summary.get("title") or {}).get("title") or {}).get("value"))
            if not title:
                continue

            raw_year = ((summary.get("publication-date") or {}).get("year") or {}).get("value")
            try:
                year = int(raw_year) if raw_year is not None else None
            except (TypeError, ValueError):
                year = None

            key = (title.casefold(), year)
            if key in seen:
                continue
            seen.add(key)

            works.append(
                {
                    "title": title,
                    "year": year,
                    "type": summary.get("type"),
                    "doi": _extract_doi(summary),
                    "journal": _clean_text(summary.get("journal-title")) or None,
                }
            )

    return works


def _parse_keywords(payload: Dict[str, Any]) -> List[str]:
    """Parse an ORCID ``/keywords`` response into a deduplicated list."""
    keywords: List[str] = []
    seen: set[str] = set()
    for item in payload.get("keyword") or []:
        content = _clean_text(item.get("content"))
        key = content.casefold()
        if content and key not in seen:
            seen.add(key)
            keywords.append(content)
    return keywords


def _parse_name(payload: Dict[str, Any]) -> Optional[str]:
    """Parse an ORCID ``/person`` response into a display name."""
    name = payload.get("name") or {}
    credit_name = _clean_text((name.get("credit-name") or {}).get("value"))
    if credit_name:
        return credit_name

    given = _clean_text((name.get("given-names") or {}).get("value"))
    family = _clean_text((name.get("family-name") or {}).get("value"))
    return " ".join(part for part in (given, family) if part) or None


def fetch_orcid_record(orcid: str, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Fetch and parse a public ORCID record.

    Parameters
    ----------
    orcid : str
        ORCID iD or ORCID URL.
    timeout : float, optional
        Per-request HTTP timeout in seconds (default: 15.0).

    Returns
    -------
    dict
        Parsed record with ``orcid``, ``name``, ``works``, and ``keywords``.

    Raises
    ------
    OrcidError
        If the iD is invalid or the ORCID API request fails.
    """
    orcid_id = normalize_orcid_id(orcid)
    config = get_config()
    cache_key = f"orcid:{orcid_id}"

    cached = _ORCID_CACHE.get(cache_key, config.metadata_cache_ttl)
    if cached is not None:
        return cached

    works_payload = _get_json(f"{ORCID_API_BASE}/{orcid_id}/works", timeout=timeout)
    keywords_payload = _get_json(f"{ORCID_API_BASE}/{orcid_id}/keywords", timeout=timeout)
    person_payload = _get_json(f"{ORCID_API_BASE}/{orcid_id}/person", timeout=timeout)

    record = {
        "orcid": orcid_id,
        "name": _parse_name(person_payload),
        "works": _parse_works(works_payload),
        "keywords": _parse_keywords(keywords_payload),
    }
    _ORCID_CACHE.set(cache_key, record, config.metadata_cache_ttl)
    return record


def _build_user_prompt(record: Dict[str, Any], works: List[Dict[str, Any]]) -> str:
    """Build the LLM user prompt for one ORCID record."""
    lines = [
        f"Researcher: {record.get('name') or 'Unknown'}",
        f"ORCID: {record.get('orcid')}",
    ]
    keywords = record.get("keywords") or []
    if keywords:
        lines.append("Author keywords: " + ", ".join(keywords))

    lines.append(f"Works ({len(works)}):")
    for work in works:
        year = work.get("year") or "n.d."
        lines.append(f"- {year}: {work.get('title')}")

    return "\n".join(lines)


def _parse_llm_terms(raw: str, max_terms: int) -> List[Dict[str, str]]:
    """Parse an LLM JSON response into validated search-term records."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty response")

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]

    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        raise ValueError("LLM response did not contain JSON")

    try:
        payload, _ = json.JSONDecoder().raw_decode(text, min(starts))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response contained invalid JSON: {exc}") from exc

    if isinstance(payload, dict):
        items = payload.get("search_terms") or payload.get("terms") or payload.get("suggested_terms") or []
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("LLM response JSON had an unexpected shape")

    if not isinstance(items, list):
        raise ValueError("LLM response did not contain a search-term list")

    terms: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            term = item
            reason = ""
        elif isinstance(item, dict):
            term = item.get("term") or item.get("query") or ""
            reason = item.get("reason") or ""
        else:
            continue

        term = " ".join(str(term).split()).strip("\"' ")
        reason = " ".join(str(reason).split()).strip()
        if not term or len(term) > 80:
            continue

        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append({"term": term, "reason": reason, "source": "llm"})
        if len(terms) >= max_terms:
            break

    if not terms:
        raise ValueError("LLM returned no usable search terms")
    return terms


def _fallback_terms(
    record: Dict[str, Any],
    works: List[Dict[str, Any]],
    max_terms: int,
) -> List[Dict[str, str]]:
    """Derive search terms from ORCID keywords and work titles."""
    terms: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add(term: str, reason: str) -> None:
        cleaned = " ".join(str(term).split()).strip("\"' ")
        if not cleaned or len(cleaned) > 80:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        terms.append({"term": cleaned, "reason": reason, "source": "fallback"})

    for keyword in record.get("keywords") or []:
        add(keyword, "ORCID author keyword")

    if len(terms) < max_terms:
        phrases = extract_top_keywords(
            [{"title": work.get("title")} for work in works],
            n_keywords=max_terms,
            ngram_range=(1, 2),
        )
        for phrase in phrases:
            add(phrase, "Frequent phrase in ORCID work titles")

    if not terms:
        for work in works:
            add(str(work.get("title") or ""), "ORCID work title")

    return terms[:max_terms]


def _generate_llm_terms(
    record: Dict[str, Any],
    works: List[Dict[str, Any]],
    max_terms: int,
    *,
    client: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Ask the configured chat model for search terms."""
    config = get_config()
    owns_client = client is None

    if client is None:
        provider = _OrcidLlmProvider()
        client = provider.openai_client

    try:
        raw = chat_completion_text(
            client=client,
            model=config.chat_model,
            system_prompt=_LLM_SYSTEM_PROMPT.replace("{max_terms}", str(max_terms)),
            user_prompt=_build_user_prompt(record, works),
            temperature=0.3,
            max_tokens=800,
            timeout=config.llm_request_timeout,
            reasoning_effort="low",
        )
    finally:
        if owns_client and client is not None:
            try:
                client.close()
            except Exception as exc:
                logger.debug("Failed to close ORCID LLM client: %s", exc)

    return _parse_llm_terms(raw, max_terms)


def suggest_search_terms_from_orcid(
    orcid: str,
    max_terms: int = 15,
    min_year: Optional[int] = None,
    max_works: int = 100,
    *,
    timeout: float = 15.0,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Suggest paper search terms from a researcher's public ORCID record.

    Parameters
    ----------
    orcid : str
        ORCID iD or ORCID URL.
    max_terms : int, optional
        Maximum number of search terms to return (default: 15, capped at 50).
    min_year : int, optional
        Only consider works published in or after this year.
    max_works : int, optional
        Maximum number of works to send to the LLM (default: 100, capped at 500).
    timeout : float, optional
        Per-request ORCID API timeout in seconds (default: 15.0).
    client : optional
        Pre-built OpenAI-compatible client.  Mainly used for testing.

    Returns
    -------
    dict
        Suggested search terms plus the ORCID name, keywords, work counts,
        recent titles, source (``llm``, ``fallback``, or ``none``), and an
        optional warning.
    """
    try:
        max_terms = max(1, min(int(max_terms), 50))
    except (TypeError, ValueError):
        max_terms = 15
    try:
        max_works = max(1, min(int(max_works), 500))
    except (TypeError, ValueError):
        max_works = 100

    record = fetch_orcid_record(orcid, timeout=timeout)

    works = list(record.get("works") or [])
    if min_year is not None:
        try:
            year_threshold = int(min_year)
        except (TypeError, ValueError):
            year_threshold = None
        if year_threshold is not None:
            works = [work for work in works if work.get("year") is None or int(work["year"]) >= year_threshold]

    works.sort(key=lambda work: work.get("year") or 0, reverse=True)
    works = works[:max_works]

    result: Dict[str, Any] = {
        "orcid": record.get("orcid"),
        "name": record.get("name"),
        "works_found": len(record.get("works") or []),
        "works_used": len(works),
        "keywords": list(record.get("keywords") or []),
        "suggested_terms": [],
        "recent_titles": [work.get("title") for work in works[:10] if work.get("title")],
        "source": "none",
        "warning": None,
    }

    if not works and not result["keywords"]:
        result["warning"] = "The public ORCID record has no works or keywords to derive search terms from."
        return result

    try:
        result["suggested_terms"] = _generate_llm_terms(
            record,
            works,
            max_terms,
            client=client,
        )
        result["source"] = "llm"
    except Exception as exc:
        logger.warning("LLM ORCID term generation failed: %s", exc)
        result["suggested_terms"] = _fallback_terms(record, works, max_terms)
        result["source"] = "fallback"
        result["warning"] = f"LLM term generation failed ({exc}); used ORCID keywords and work titles instead."

    return result
