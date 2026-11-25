"""
Shared location mapping data structure.

The ``LocationMap`` dataclass is generic; conference-specific code that builds
and applies it lives in the respective downloader plugin.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
