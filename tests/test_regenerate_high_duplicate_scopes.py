"""Tests for the high-duplicate scope selection helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from abstracts_explorer.plugin import LightweightPaper

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_high_duplicate_scopes.py"
SPEC = importlib.util.spec_from_file_location("regenerate_high_duplicate_scopes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
regen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regen)


def _paper(title: str, year: int, session: str) -> LightweightPaper:
    return LightweightPaper(
        title=title,
        authors=["Author"],
        abstract=f"Abstract for {title}",
        session=session,
        poster_position="1",
        year=year,
        conference="ICML",
        original_id=f"{title}:{session}",
    )


def test_high_duplicate_scopes_selects_and_sorts(connected_db):
    connected_db.add_papers(
        [
            _paper("A", 2022, "Poster"),
            _paper("A", 2022, "Talk"),
            _paper("B", 2022, "Poster"),
            _paper("B", 2022, "Talk"),
            _paper("C", 2022, "Poster"),
            _paper("D", 2023, "Poster"),
            _paper("D", 2023, "Talk"),
        ]
    )

    scopes = regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=2)

    assert scopes == [("ICML", 2022, 5, 2)]
    assert regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=1) == [
        ("ICML", 2022, 5, 2),
        ("ICML", 2023, 2, 1),
    ]
    assert regen.high_duplicate_scopes(connected_db, conference="ICML", year=2023, threshold=1) == [
        ("ICML", 2023, 2, 1)
    ]
    assert regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=99) == []
