"""
Tests for shared text utilities.
"""

import pytest

from abstracts_explorer.utils.text_utils import html_to_text


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
