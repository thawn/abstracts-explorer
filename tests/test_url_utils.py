"""
Tests for URL utility helpers.
"""

from abstracts_explorer.utils.url_utils import normalize_doi_url, paper_pdf_url


class TestNormalizeDoiUrl:
    """Tests for normalize_doi_url."""

    def test_bare_doi(self):
        assert normalize_doi_url("10.1234/abc") == "https://doi.org/10.1234/abc"

    def test_doi_org_prefixed_path(self):
        assert normalize_doi_url("doi.org/10.1234/abc") == "https://doi.org/10.1234/abc"

    def test_http_url_upgraded_to_https(self):
        assert normalize_doi_url("http://doi.org/10.1234/abc") == "https://doi.org/10.1234/abc"

    def test_https_url_returned_unchanged(self):
        assert normalize_doi_url("https://doi.org/10.1234/abc") == "https://doi.org/10.1234/abc"

    def test_https_non_doi_url_returned_unchanged(self):
        assert normalize_doi_url("https://example.com/x") == "https://example.com/x"

    def test_whitespace_stripped(self):
        assert normalize_doi_url("  10.1234/abc  ") == "https://doi.org/10.1234/abc"


class TestPaperPdfUrl:
    """Tests for paper_pdf_url."""

    def test_prefers_explicit_pdf_url(self):
        paper = {"paper_pdf_url": "https://example.com/p.pdf", "paper_url": "https://openreview.net/forum?id=x"}
        assert paper_pdf_url(paper) == "https://example.com/p.pdf"

    def test_falls_back_to_openreview_pdf_url(self):
        paper = {"paper_url": "https://openreview.net/forum?id=abc"}
        assert paper_pdf_url(paper) == "https://openreview.net/pdf?id=abc"

    def test_plain_paper_url_passed_through(self):
        paper = {"paper_url": "https://example.com/paper"}
        assert paper_pdf_url(paper) == "https://example.com/paper"

    def test_missing_fields_return_none(self):
        assert paper_pdf_url({}) is None

    def test_empty_pdf_url_falls_back(self):
        paper = {"paper_pdf_url": "", "paper_url": "https://openreview.net/forum?id=abc"}
        assert paper_pdf_url(paper) == "https://openreview.net/pdf?id=abc"
