"""Tests for location derivation and content-hash grouping."""

from __future__ import annotations

from abstracts_explorer.embeddings.hashing import paper_content_hash
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.plugin.locations import LocationMap
from abstracts_explorer.plugins.neurips_downloader import (
    assign_locations,
    build_location_map,
    discover_locations,
)
from abstracts_explorer.utils.hashing import hash_text


def _paper(
    title: str,
    session: str,
    room_name: str = "",
    location: str | None = None,
    original_id: str | None = None,
    year: int = 2026,
) -> LightweightPaper:
    return LightweightPaper(
        original_id=original_id or f"{title}::{session}",
        title=title,
        authors=["Author"],
        abstract=f"Abstract for {title}",
        session=session,
        poster_position="1",
        year=year,
        conference="NeurIPS",
        room_name=room_name or None,
        location=location,
    )


def _index_html(*locations: tuple[str, str]) -> str:
    tabs = []
    for name, slug in locations:
        tabs.append(
            f'<li class="nav-item" role="presentation">'
            f'<img alt="{name} graphic" src="/media/Locations/1-{slug}.svg">'
            f"</li>"
        )
    return "<html>" + "".join(tabs) + "</html>"


def _calendar_html(sessions: list[str], rooms: list[str] | None = None) -> str:
    session_html = "".join(f'<a href="/virtual/2026/session/{i}">{session}</a>' for i, session in enumerate(sessions))
    room_html = "".join(f'<option class="filter-select" value="{room}">{room}</option>' for room in rooms or [])
    return f"<html>{session_html}{room_html}</html>"


def test_discover_locations_preserves_order():
    pages = {"https://example.org/virtual/2026/index.html": _index_html(("Sydney", "sydney"), ("Paris", "paris"))}
    locations = discover_locations(2026, "https://example.org", pages.__getitem__)
    assert locations == [("Sydney", "sydney"), ("Paris", "paris")]


def test_assign_prefers_unique_session_then_room_intersection():
    pages = {
        "https://example.org/virtual/2026/index.html": _index_html(
            ("Sydney", "sydney"), ("Paris", "paris"), ("Atlanta", "atlanta")
        ),
        "https://example.org/virtual/2026/loc/sydney/calendar": _calendar_html(
            ["Sydney Poster Session 1", "Orals 1A Shared"], ["Hall 1", "Main Hall"]
        ),
        "https://example.org/virtual/2026/loc/paris/calendar": _calendar_html(
            ["Paris Poster Session 1", "Orals 1A Shared"], ["Paris Hall"]
        ),
        "https://example.org/virtual/2026/loc/atlanta/calendar": _calendar_html(
            ["Atlanta Poster Session 1"], ["Hall C1"]
        ),
    }
    mapping = build_location_map(2026, "https://example.org", pages.__getitem__)

    assert mapping.main_location == "Sydney"
    assert mapping.is_multi_location
    assert mapping.assign("Paris Poster Session 1", "Paris Hall") == "Paris"
    assert mapping.assign("Orals 1A Shared", "Main Hall") == "Sydney"
    assert mapping.assign("Orals 1A Shared", "Paris Hall") == "Paris"
    assert mapping.assign("Unlisted Session", "", None) == "Sydney"
    assert mapping.assign("No session", "", None) is None


def test_assign_uses_event_type_city():
    mapping = LocationMap(locations=["San Diego", "Mexico City"])
    assert mapping.assign("Oral 1A", "Don Alberto 1", event_type="Mexico City Oral") == "Mexico City"


def test_assign_locations_sets_attribute_on_models():
    mapping = LocationMap(locations=["Sydney"])
    mapping.session_locations["Sydney Poster Session 1"].add("Sydney")
    papers = [_paper("Paper A", "Sydney Poster Session 1")]
    assign_locations(papers, mapping)
    assert papers[0].location == "Sydney"


def test_paper_content_hash_matches_chroma_hash_text():
    assert paper_content_hash("Title", "Abstract") == hash_text("Title\n\nAbstract")
    assert paper_content_hash(None, None) == hash_text("")


def test_database_location_and_hash_queries(connected_db):
    first = _paper("Shared paper", "Sydney Poster Session 1", "Hall 1", "Sydney")
    second = _paper("Shared paper", "Paris Poster Session 1", "Paris Hall", "Paris")
    third = _paper("Unique paper", "Sydney Poster Session 2", "Hall 1", "Sydney")

    connected_db.add_papers([first, second, third])

    assert connected_db.get_locations(conference="NeurIPS", year=2026) == ["Paris", "Sydney"]
    assert connected_db.get_conference_year_locations()["NeurIPS"][2026] == ["Paris", "Sydney"]
    assert set(connected_db.resolve_paper_uids(locations=["Paris"])) == {
        connected_db.compute_uid(second.title, second.original_id, "NeurIPS", 2026)
    }

    stats = connected_db.get_stats(year=2026, conference="NeurIPS")
    assert stats["total_papers"] == 3
    assert stats["unique_papers"] == 2

    hashes = connected_db.get_content_hashes([])
    assert hashes == {}
    grouped = connected_db.get_locations_by_content_hashes(
        [paper_content_hash("Shared paper", "Abstract for Shared paper")]
    )
    assert grouped[paper_content_hash("Shared paper", "Abstract for Shared paper")] == ["Paris", "Sydney"]


def test_group_papers_by_content_does_not_merge_across_years(connected_db):
    connected_db.add_papers(
        [
            _paper("Recurring paper", "Sydney Poster Session 1", "Hall 1", "Sydney", year=2025),
            _paper("Recurring paper", "Sydney Poster Session 1", "Hall 1", "Sydney", year=2026),
        ]
    )
    results = connected_db.group_papers_by_content(connected_db.get_paper())
    recurring = [paper for paper in results if paper["title"] == "Recurring paper"]
    assert len(recurring) == 2


def test_group_papers_by_content_merges_presentations(connected_db):
    connected_db.add_papers(
        [
            _paper("Shared paper", "Sydney Poster Session 1", "Hall 1", "Sydney"),
            _paper("Shared paper", "Paris Poster Session 1", "Paris Hall", "Paris"),
        ]
    )
    results = connected_db.group_papers_by_content(connected_db.get_paper())
    shared = [paper for paper in results if paper["title"] == "Shared paper"]
    assert len(shared) == 1
    assert shared[0]["locations"] == ["Sydney", "Paris"]
    assert len(shared[0]["presentations"]) == 2
