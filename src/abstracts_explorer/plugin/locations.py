"""
EventHosts location mapping helpers.

Derives a mapping from session titles and room names to physical conference
locations by reading the per-location calendars of an EventHosts virtual site.
Only a small, fixed number of pages are fetched per conference year.
"""

from __future__ import annotations

import html
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_LOCATION_TAB_RE = re.compile(r'<li class="nav-item" role="presentation">')
_LOCATION_NAME_RE = re.compile(r'alt="([^"]+) graphic"')
_LOCATION_SLUG_RE = re.compile(r"/media/Locations/[^\"']*?-([a-z0-9\-]+)\.svg")
_SESSION_TITLE_TMPL = r'/virtual/%d/session/\d+">\s*([^<\n]+)'
_ROOM_OPTION_RE = re.compile(r'<option class="filter-select" value="([^"]+)"')

FetchText = Callable[[str], str]


@dataclass
class LocationMap:
    """
    Mapping of sessions and rooms to physical conference locations.

    Attributes
    ----------
    locations : list of str
        Location display names in site order (first entry is the main site).
    session_locations : dict
        Mapping from session title to the set of locations that list it.
    room_locations : dict
        Mapping from room display name to the set of locations that list it.
    """

    locations: List[str] = field(default_factory=list)
    session_locations: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    room_locations: Dict[str, set] = field(default_factory=lambda: defaultdict(set))

    @property
    def main_location(self) -> Optional[str]:
        """Return the main (first) location, if any."""
        return self.locations[0] if self.locations else None

    @property
    def is_multi_location(self) -> bool:
        """Return whether more than one location was discovered."""
        return len(self.locations) > 1

    def assign(
        self, session: Optional[str], room_name: Optional[str], event_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Return the location for a presentation, if it can be determined.

        Resolution order is: explicit city in ``event_type``, a session title
        listed by exactly one location, room/session intersection, and finally
        the main location for non-empty sessions not listed anywhere.

        Parameters
        ----------
        session : str or None
            Session title from the conference data.
        room_name : str or None
            Room display name from the conference data.
        event_type : str or None
            Event type string, which may contain a location for orals.

        Returns
        -------
        str or None
            Location display name, or ``None`` when it cannot be determined.
        """
        session = (session or "").strip()
        room_name = (room_name or "").strip()

        if event_type:
            for location in self.locations:
                if event_type.strip().lower() == f"{location.lower()} oral":
                    return location

        candidates = set(self.session_locations.get(session, set()))
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            room_candidates = self.room_locations.get(room_name, set()) & candidates
            if len(room_candidates) == 1:
                return next(iter(room_candidates))

        if not session or session.lower() == "no session":
            return None
        return self.main_location


def discover_locations(year: int, base_url: str, fetch_text: FetchText) -> List[Tuple[str, str]]:
    """
    Parse the EventHosts landing page for location tabs.

    Parameters
    ----------
    year : int
        Conference year.
    base_url : str
        Site base URL (e.g. ``https://neurips.cc``).
    fetch_text : callable
        Function fetching a URL and returning its text.

    Returns
    -------
    list of tuple
        ``(display_name, slug)`` pairs in tab order. The first entry is the
        main site.
    """
    page = fetch_text(f"{base_url}/virtual/{year}/index.html")
    locations: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for block in _LOCATION_TAB_RE.split(page):
        name_match = _LOCATION_NAME_RE.search(block)
        slug_match = _LOCATION_SLUG_RE.search(block)
        if not name_match or not slug_match:
            continue
        name = html.unescape(name_match.group(1)).strip()
        slug = slug_match.group(1)
        if slug in seen:
            continue
        seen.add(slug)
        locations.append((name, slug))
    return locations


def parse_location_calendar(year: int, slug: str, base_url: str, fetch_text: FetchText) -> Tuple[set, set]:
    """
    Parse a location calendar for session titles and room names.

    Parameters
    ----------
    year : int
        Conference year.
    slug : str
        Location slug.
    base_url : str
        Site base URL.
    fetch_text : callable
        Function fetching a URL and returning its text.

    Returns
    -------
    tuple of set
        ``(session_titles, room_names)``.
    """
    page = fetch_text(f"{base_url}/virtual/{year}/loc/{slug}/calendar")
    title_re = re.compile(_SESSION_TITLE_TMPL % year)
    sessions = {html.unescape(value.strip()) for value in title_re.findall(page)}
    rooms = {html.unescape(value.strip()) for value in _ROOM_OPTION_RE.findall(page)}
    return sessions, rooms


def build_location_map(year: int, base_url: str, fetch_text: FetchText) -> LocationMap:
    """
    Build a session/room to location mapping for a conference year.

    Parameters
    ----------
    year : int
        Conference year.
    base_url : str
        Site base URL (e.g. ``https://neurips.cc``).
    fetch_text : callable
        Function fetching a URL and returning its text.

    Returns
    -------
    LocationMap
        Mapping populated from the per-location calendars.

    Raises
    ------
    RuntimeError
        If no locations can be discovered.
    """
    discovered = discover_locations(year, base_url, fetch_text)
    if not discovered:
        raise RuntimeError(f"No EventHosts locations found for {year} at {base_url}")

    result = LocationMap(locations=[name for name, _ in discovered])
    for name, slug in discovered:
        sessions, rooms = parse_location_calendar(year, slug, base_url, fetch_text)
        for session in sessions:
            result.session_locations[session].add(name)
        for room in rooms:
            result.room_locations[room].add(name)
    return result


def assign_locations(
    papers: Sequence,
    location_map: LocationMap,
) -> None:
    """
    Assign a location to each lightweight paper in place.

    Parameters
    ----------
    papers : sequence
        Objects or dictionaries with ``session``, ``room_name``, and
        ``event_type``/``location`` attributes.
    location_map : LocationMap
        Mapping built by :func:`build_location_map`.
    """
    for paper in papers:
        if isinstance(paper, dict):
            session = paper.get("session")
            room = paper.get("room_name")
            event_type = paper.get("event_type")
            paper["location"] = location_map.assign(session, room, event_type)
        else:
            session = getattr(paper, "session", None)
            room = getattr(paper, "room_name", None)
            event_type = getattr(paper, "event_type", None)
            paper.location = location_map.assign(session, room, event_type)


def build_neurips_location_map(
    year: int,
    fetch_text: FetchText,
    base_url: str = "https://neurips.cc",
) -> LocationMap:
    """
    Build a location map for NeurIPS.

    Parameters
    ----------
    year : int
        Conference year.
    fetch_text : callable
        Function fetching a URL and returning its text.
    base_url : str, optional
        NeurIPS base URL.

    Returns
    -------
    LocationMap
        Session/room to location mapping.
    """
    return build_location_map(year, base_url, fetch_text)
