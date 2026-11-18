"""
Tests for conference name/year parsing helpers.
"""

from abstracts_explorer.utils.conference_utils import parse_conference_year


class TestParseConferenceYear:
    """Tests for parse_conference_year."""

    def test_four_digit_year(self):
        assert parse_conference_year("NeurIPS 2025") == ("NeurIPS", 2025)

    def test_multi_word_conference(self):
        assert parse_conference_year("IEEE VIS 2024") == ("IEEE VIS", 2024)

    def test_apostrophe_two_digit_year(self):
        assert parse_conference_year("ICLR'26") == ("ICLR", 2026)

    def test_space_two_digit_year(self):
        assert parse_conference_year("ICLR 26") == ("ICLR", 2026)

    def test_no_year(self):
        assert parse_conference_year("ICLR") == ("ICLR", None)

    def test_whitespace_is_stripped(self):
        assert parse_conference_year("  NeurIPS 2025  ") == ("NeurIPS", 2025)

    def test_non_year_trailing_number_is_kept(self):
        assert parse_conference_year("NeurIPS 123") == ("NeurIPS 123", None)

    def test_empty_string(self):
        assert parse_conference_year("") == ("", None)
