"""
Tests for shared text utilities.
"""

import pytest

from abstracts_explorer.utils.text_utils import (
    html_to_text,
    sanitize_json_strings,
    strip_nul_bytes,
    strip_nul_bytes_optional,
)


class TestHtmlToText:
    """Tests for the shared HTML-to-text helper."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                "<b>Data Management</b> Makes &amp; Machine Learning Easier",
                "Data Management Makes & Machine Learning Easier",
            ),
            ("<p><em>Nested</em> <strong>markup</strong></p>", "Nested markup"),
            ('<a href="https://example.com">A link</a>', "A link"),
            ("<highlight>Remote</highlight> <highlight>Sensing</highlight>", "Remote Sensing"),
            ("No tags here", "No tags here"),
            ("  Collapse   repeated\twhitespace  ", "Collapse repeated whitespace"),
            ("Use x &lt; y and y &gt; z", "Use x < y and y > z"),
            ("<b>Malformed </i>markup", "Malformed markup"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_converts_html_fragments(self, value, expected):
        assert html_to_text(value) == expected


class TestStripNulBytes:
    """Tests for removing NUL bytes before PostgreSQL writes."""

    def test_removes_nul_bytes(self):
        assert strip_nul_bytes("alpha\x00beta\x00") == "alphabeta"

    def test_none_becomes_empty_string(self):
        assert strip_nul_bytes(None) == ""

    def test_non_string_is_coerced(self):
        assert strip_nul_bytes(12) == "12"

    def test_optional_preserves_none(self):
        assert strip_nul_bytes_optional(None) is None
        assert strip_nul_bytes_optional("a\x00b") == "ab"


class TestSanitizeJsonStrings:
    """Tests for recursive JSON NUL sanitization."""

    def test_nested_structures(self):
        value = {"label": "a\x00b", "items": ["c\x00d", 3, None]}
        assert sanitize_json_strings(value) == {"label": "ab", "items": ["cd", 3, None]}

    def test_dict_keys(self):
        assert sanitize_json_strings({"a\x00b": "v"}) == {"ab": "v"}
