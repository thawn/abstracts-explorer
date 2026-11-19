"""
Tests for paper presentation helpers.
"""

from abstracts_explorer.utils.paper_presentation import truncate_abstract


class TestTruncateAbstract:
    """Tests for truncate_abstract."""

    def test_short_text_returned_unchanged(self):
        assert truncate_abstract("short", 10) == "short"

    def test_exact_length_returned_unchanged(self):
        assert truncate_abstract("a" * 10, 10) == "a" * 10

    def test_long_text_truncated_with_ellipsis(self):
        assert truncate_abstract("a" * 12, 10) == "a" * 10 + "..."

    def test_empty_text(self):
        assert truncate_abstract("", 10) == ""

    def test_custom_max_length(self):
        assert truncate_abstract("abcdef", 3) == "abc..."
