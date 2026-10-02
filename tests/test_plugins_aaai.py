"""Mocked regression tests for the official AAAI proceedings downloader."""

import json
from datetime import datetime
from unittest.mock import Mock

import pytest
import requests
from bs4 import BeautifulSoup

from abstracts_explorer.plugin import LightweightDownloaderPlugin
from abstracts_explorer.plugins import AAAIDownloaderPlugin, get_plugin

BASE = "https://ojs.aaai.org/index.php/AAAI"

# Minimal synthetic OJS pages; no downloaded website snapshots are needed.
AAAI_PAGES = {
    "/issue/archive": """
<div class="obj_issue_summary">
  <h2><a class="title" href="/index.php/AAAI/issue/view/683">
    AAAI-26 Technical Tracks 1
  </a></h2>
  <div class="series">Vol. 40 No. 1</div>
</div>
<div class="obj_issue_summary">
  <a class="title" href="/index.php/AAAI/issue/view/683">
    AAAI-26 Technical Tracks 1
  </a>
  <div class="series">Vol. 40 No. 1</div>
</div>
<div class="cmp_pagination">
  <span class="current">1-2 of 6</span>
  <a class="next" href="/index.php/AAAI/issue/archive/2">Next</a>
</div>
""",
    "/issue/archive/2": """
<div class="obj_issue_summary">
  <a class="title" href="/index.php/AAAI/issue/view/729">
    AAAI-26 Journal Track, IAAI-26 and EAAI-26
  </a>
  <div class="series">Vol. 40 No. 47</div>
</div>
<div class="obj_issue_summary">
  <a class="title" href="/index.php/AAAI/issue/view/309">
    Twenty-Fourth AAAI Conference on Artificial Intelligence
  </a>
  <div class="series">Vol. 24 No. 1 (2010)</div>
</div>
<div class="obj_issue_summary">
  <a class="title" href="/index.php/AAAI/issue/view/467">
    Twenty-Second Innovative Applications of Artificial Intelligence
  </a>
  <div class="series">Vol. 24 No. 2 (2010)</div>
</div>
<div class="obj_issue_summary">
  <a class="title" href="/index.php/AAAI/issue/view/468">
    First Symposium on Education Advances in Artificial Intelligence
  </a>
  <div class="series">Vol. 24 No. 3 (2010)</div>
</div>
<div class="cmp_pagination"><span class="current">3-6 of 6</span></div>
""",
    "/issue/view/683": """
<div class="section">
  <h2>Frontmatter</h2>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/900">Editorial</a>
  </h3></div>
</div>
<div class="section">
  <h2>AAAI Technical Track on Machine Learning</h2>
  <div class="obj_article_summary">
    <h3 class="title">
      <a href="/index.php/AAAI/article/view/36958">Learning &amp; Reasoning</a>
    </h3>
    <a class="obj_galley_link pdf"
       href="/index.php/AAAI/article/view/36958/40920">PDF</a>
  </div>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/36959">No abstract</a>
  </h3></div>
</div>
""",
    "/issue/view/729": """
<div class="section">
  <h2>AAAI Journal Track</h2>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/36958">Duplicate</a>
  </h3></div>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/36960">Reliable Planning</a>
  </h3></div>
</div>
<div class="section">
  <h2>IAAI Technical Track on Deployed Applications</h2>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/901">Co-located IAAI paper</a>
  </h3></div>
</div>
<div class="section">
  <h2>EAAI Symposium: Main Track</h2>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/902">Co-located EAAI paper</a>
  </h3></div>
</div>
""",
    "/issue/view/309": """
<div class="section">
  <h2>Constraints, Satisfiability, and Search</h2>
  <div class="obj_article_summary"><h3 class="title">
    <a href="/index.php/AAAI/article/view/7545">Weighted Search</a>
  </h3></div>
</div>
""",
    "/article/view/36958": """
<meta name="citation_title" content="Learning &amp; Reasoning">
<meta name="citation_author" content="Ada Researcher">
<meta name="citation_author" content="Lin; Scientist">
<meta name="citation_author" content="   ">
<meta name="DC.Description"
    content="&lt;p&gt;We study &lt;em&gt;learning&lt;/em&gt; and reasoning.&lt;/p&gt;">
<meta name="citation_pdf_url" content="/index.php/AAAI/article/download/36958/40920">
<meta name="DC.Subject" content="Machine learning">
<meta name="DC.Subject" content="Reasoning">
<meta name="citation_date" content="2025/12/01">
""",
    "/article/view/36959": """
<meta name="citation_title" content="No abstract">
<meta name="citation_author" content="Ada Researcher">
""",
    "/article/view/36960": """
<meta name="citation_title" content="Reliable Planning">
<meta name="citation_author" content="Lin Scientist">
<section class="item abstract">
  <h2 class="label">Abstract</h2>
  <p>We study <em>reliable</em> planning.</p>
  <p>Experiments support the approach.</p>
</section>
""",
    "/article/view/7545": """
<meta name="DC.Title" content="Weighted Search">
<meta name="DC.Creator.PersonalName" content="Ada Researcher">
<meta name="DC.Description" content="An algorithm for weighted search.">
<meta name="citation_pdf_url"
      content="https://ojs.aaai.org/index.php/AAAI/article/download/7545/7406">
""",
}


@pytest.fixture
def aaai_site(monkeypatch):
    """Provide fresh synthetic OJS pages and a strictly mocked HTTP session."""
    pages = {BASE + path: html for path, html in AAAI_PAGES.items()}

    def respond(url, **kwargs):
        return Mock(text=pages[url])

    plugin = AAAIDownloaderPlugin()
    get = Mock(side_effect=respond)
    monkeypatch.setattr(plugin._session, "get", get)
    monkeypatch.setattr("abstracts_explorer.plugins.aaai_downloader.time.sleep", Mock())
    return plugin, pages, get


def test_registration():
    """Importing the package registers the standalone lightweight plugin."""
    plugin = get_plugin("aaai")
    assert isinstance(plugin, AAAIDownloaderPlugin)
    assert isinstance(plugin, LightweightDownloaderPlugin)
    assert plugin.conference_name == "AAAI"


def test_metadata_and_supported_years(aaai_site):
    """Metadata exposes parameters and only probes actual current-year issues."""
    plugin, _, get = aaai_site
    metadata = plugin.get_metadata()
    assert metadata["name"] == "aaai"
    assert metadata["conference_name"] == "AAAI"
    assert metadata["supported_years"][0] == 2010
    assert "request_delay" in metadata["parameters"]
    plugin.get_metadata()
    assert get.call_count == 1
    assert plugin.get_url(2010) == plugin.get_url(2026) == BASE + "/issue/archive"


@pytest.mark.parametrize("year,available", [(2026, True), (2027, False)])
def test_current_year_availability(aaai_site, year, available):
    """An HTTP 200 for a shared archive is not proof of a year's availability."""
    plugin, _, _ = aaai_site
    assert plugin._check_current_year_available(year) is available


def test_current_year_network_failure(aaai_site):
    """Metadata probing tolerates an unavailable website."""
    plugin, _, get = aaai_site
    get.side_effect = requests.Timeout("offline")
    assert plugin._check_current_year_available(2026) is False


def test_colocated_issue_not_current_year_proof(aaai_site):
    """An IAAI-only issue mentioning AAAI in its description is not enough."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/archive"] = (
        '<div class="obj_issue_summary"><a class="title">IAAI-26</a>'
        '<div class="series">Vol. 40 No. 47</div><p>Held with AAAI</p></div>'
    )
    assert plugin._check_current_year_available(2026) is False


@pytest.mark.parametrize(
    "series,year", [("Vol. 24 No. 1 (2010)", 2010), ("Vol. 40 No. 1", 2026), ("", None)]
)
def test_issue_year(series, year):
    """Old explicit years and modern volume numbers map to conference years."""
    summary = BeautifulSoup(f'<div class="series">{series}</div>', "html.parser")
    assert AAAIDownloaderPlugin._issue_year(summary) == year


def test_multiple_issues_and_metadata(aaai_site, caplog):
    """Follow pagination, deduplicate papers, and preserve available metadata."""
    plugin, _, get = aaai_site
    papers = plugin.download(2026, request_delay=0)
    assert [paper.original_id for paper in papers] == [36958, 36960]
    paper = papers[0]
    assert paper.title == "Learning & Reasoning"
    assert paper.authors == ["Ada Researcher", "Lin Scientist"]
    assert paper.abstract == "We study learning and reasoning."
    assert paper.year == 2026  # Not the citation publication date of 2025.
    assert paper.conference == "AAAI"
    assert paper.session == "AAAI Technical Track on Machine Learning"
    assert paper.poster_position == ""
    assert paper.url == BASE + "/article/view/36958"
    assert paper.paper_pdf_url == BASE + "/article/download/36958/40920"
    assert paper.keywords == ["Machine learning", "Reasoning"]
    assert papers[1].session == "AAAI Journal Track"
    assert papers[1].abstract == (
        "We study reliable planning. Experiments support the approach."
    )
    assert papers[1].paper_pdf_url is None
    assert "Skipping AAAI article" in caplog.text
    fetched = [call.args[0] for call in get.call_args_list]
    assert fetched.count(BASE + "/issue/view/683") == 1
    assert fetched.count(BASE + "/article/view/36958") == 1
    assert not any(url.endswith(("/900", "/901", "/902", "/40920")) for url in fetched)


def test_old_proceedings(aaai_site):
    """Download historical abstracts using Dublin Core metadata fallbacks."""
    plugin, _, get = aaai_site
    (paper,) = plugin.download(2010, request_delay=0)
    assert paper.title == "Weighted Search"
    assert paper.authors == ["Ada Researcher"]
    assert paper.abstract == "An algorithm for weighted search."
    assert paper.year == 2010
    assert paper.session == "Constraints, Satisfiability, and Search"
    assert paper.original_id == 7545
    assert not any(
        call.args[0].endswith(("/467", "/468")) for call in get.call_args_list
    )


def test_default_year(aaai_site):
    """Omitting the year uses the latest year advertised by the plugin."""
    plugin, _, _ = aaai_site
    plugin.supported_years = [2010, 2026]
    assert all(paper.year == 2026 for paper in plugin.download(request_delay=0))


def test_transport_options_and_delay(aaai_site):
    """Forward HTTP options and delay every proceedings request."""
    plugin, _, get = aaai_site
    plugin.download(2026, timeout=7, verify_ssl=False, request_delay=0.25)
    for call in get.call_args_list:
        assert call.kwargs == {"timeout": 7, "verify": False}
    from abstracts_explorer.plugins.aaai_downloader import time

    assert time.sleep.call_count == get.call_count
    time.sleep.assert_called_with(0.25)


def test_explicit_download_after_failed_availability_probe(aaai_site, monkeypatch):
    """A negative metadata probe must not block a configured explicit download."""
    plugin, _, get = aaai_site
    clock = Mock()
    clock.now.return_value = datetime(2026, 1, 1)
    monkeypatch.setattr("abstracts_explorer.plugin.datetime", clock)
    monkeypatch.setattr("abstracts_explorer.plugins.aaai_downloader.datetime", clock)
    respond = get.side_effect
    get.side_effect = requests.Timeout("availability probe failed")
    assert 2026 not in plugin.get_metadata()["supported_years"]
    get.side_effect = respond
    get.reset_mock()
    papers = plugin.download(2026, timeout=60, verify_ssl=False, request_delay=0)
    assert [paper.original_id for paper in papers] == [36958, 36960]
    for call in get.call_args_list:
        assert call.kwargs == {"timeout": 60, "verify": False}


def test_no_probe_for_explicit_download(aaai_site, monkeypatch):
    """An explicit download goes straight to discovery with caller options."""
    plugin, _, _ = aaai_site
    probe = Mock(side_effect=AssertionError("Do not probe during explicit download"))
    monkeypatch.setattr(plugin, "_check_current_year_available", probe)
    assert len(plugin.download(2026, request_delay=0)) == 2
    probe.assert_not_called()


def test_article_identity_deduplication(aaai_site):
    """Trailing slashes and leading-zero IDs must not duplicate an article."""
    plugin, pages, get = aaai_site
    pages[BASE + "/issue/view/729"] = pages[BASE + "/issue/view/729"].replace(
        '/article/view/36958"', '/article/view/036958/"'
    )
    papers = plugin.download(2026, request_delay=0)
    assert [paper.original_id for paper in papers] == [36958, 36960]
    assert sum("/article/view/" in call.args[0] for call in get.call_args_list) == 3


@pytest.mark.parametrize("change", ["title", "summary", "url"])
def test_partial_issue_layout_preserves_cache(aaai_site, tmp_path, change):
    """A malformed later issue must not save an earlier issue's partial result."""
    plugin, pages, _ = aaai_site
    url = BASE + "/issue/view/729"
    if change == "title":
        pages[url] = pages[url].replace('class="title"', 'class="paper-title"')
    elif change == "summary":
        pages[url] = pages[url].replace(
            'class="obj_article_summary"', 'class="article-summary"'
        )
    else:
        pages[url] = pages[url].replace("/article/view/36960", "/about")
    path = tmp_path / "aaai.json"
    path.write_text("existing cache", encoding="utf-8")
    with pytest.raises(RuntimeError, match="AAAI.*(?:section|article)"):
        plugin.download(2026, str(path), force_download=True, request_delay=0)
    assert path.read_text(encoding="utf-8") == "existing cache"


def test_one_missing_article_summary_preserves_cache(aaai_site, tmp_path):
    """Cross-check list entries so one malformed row cannot silently disappear."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/view/729"] = (
        '<div class="section"><h2>AAAI Journal Track</h2><ul class="articles">'
        '<li><div class="obj_article_summary"><h3 class="title">'
        '<a href="/index.php/AAAI/article/view/36958">Duplicate</a>'
        '</h3></div></li><li><div class="article-summary"><h3 class="title">'
        '<a href="/index.php/AAAI/article/view/36960">Paper</a>'
        "</h3></div></li></ul></div>"
    )
    path = tmp_path / "aaai.json"
    path.write_text("existing cache", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Inconsistent AAAI article listing"):
        plugin.download(2026, str(path), force_download=True, request_delay=0)
    assert path.read_text(encoding="utf-8") == "existing cache"


@pytest.mark.parametrize("change", ["next", "counter", "title", "range", "total"])
def test_partial_archive_layout_preserves_cache(aaai_site, tmp_path, change):
    """Validate archive pagination and entries before accepting a partial year."""
    plugin, pages, _ = aaai_site
    url = BASE + "/issue/archive"
    if change == "next":
        pages[url] = pages[url].replace('class="next"', 'class="next-page"')
    elif change == "counter":
        pages[url] = pages[url].replace('class="current"', 'class="range"')
    elif change == "title":
        pages[url] = pages[url].replace('class="title"', 'class="issue-title"')
    elif change == "range":
        pages[BASE + "/issue/archive/2"] = pages[BASE + "/issue/archive/2"].replace(
            "3-6 of 6", "4-7 of 7"
        )
    else:
        pages[BASE + "/issue/archive/2"] = pages[BASE + "/issue/archive/2"].replace(
            "3-6 of 6", "3-6 of 7"
        )
    path = tmp_path / "aaai.json"
    with pytest.raises(RuntimeError, match="AAAI archive"):
        plugin.download(2026, str(path), request_delay=0)
    assert not path.exists()


def test_smoke_cache_uses_dedicated_directory():
    """The guide's output directory isolates the CLI's derived cache filename."""
    from types import SimpleNamespace
    from pathlib import Path
    from abstracts_explorer.cli import _download_single

    plugin = AAAIDownloaderPlugin()
    plugin.download = Mock(return_value=[])
    args = SimpleNamespace(max_workers=20, input_file=None)
    _download_single(
        plugin, 2010, Path("data/aaai-pr-smoke/abstracts.json"), False, args
    )
    assert plugin.download.call_args.kwargs["output_path"] == str(
        Path("data/aaai-pr-smoke/aaai_2010.json")
    )


def test_json_cache_round_trip(aaai_site, tmp_path):
    """Saved JSON can be loaded without any HTTP requests."""
    plugin, _, get = aaai_site
    path = tmp_path / "nested" / "aaai.json"
    papers = plugin.download(2026, str(path), request_delay=0)
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2
    get.reset_mock()
    get.side_effect = AssertionError("Cache loading must be offline")
    assert plugin.download(2026, str(path)) == papers
    get.assert_not_called()


@pytest.mark.parametrize("replacement", ["year", "conference", "empty"])
def test_mismatched_or_empty_cache(aaai_site, tmp_path, replacement):
    """A cache for a different conference/year or no papers must be refreshed."""
    plugin, _, get = aaai_site
    path = tmp_path / "aaai.json"
    papers = plugin.download(2026, request_delay=0)
    data = [paper.model_dump() for paper in papers]
    if replacement == "year":
        data[0]["year"] = 2010
    elif replacement == "conference":
        data[0]["conference"] = "Other"
    else:
        data = []
    path.write_text(json.dumps(data), encoding="utf-8")
    get.reset_mock()
    assert plugin.download(2026, str(path), request_delay=0) == papers
    assert get.called


def test_unsupported_year_with_cache(aaai_site, tmp_path):
    """An existing cache must not bypass basic year bounds."""
    plugin, _, _ = aaai_site
    path = tmp_path / "aaai.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not supported"):
        plugin.download(2009, str(path))


def test_database_insertion(aaai_site, connected_db):
    """Downloaded lightweight papers fit the existing database insertion path."""
    plugin, _, _ = aaai_site
    papers = plugin.download(2026, request_delay=0)
    assert connected_db.add_papers(papers) == 2


def test_section_fallback_and_non_article_link(aaai_site):
    """Handle an unlabeled section without following unrelated title links."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/view/683"] = (
        '<div class="section"><div class="obj_article_summary">'
        '<h3 class="title"><a href="/index.php/AAAI/article/view/36958">Paper</a>'
        '<a href="/about">About</a></h3></div></div>'
    )
    assert plugin.download(2026, request_delay=0)[0].session == "AAAI proceedings"


@pytest.mark.parametrize("force", [False, True])
def test_cache_fallback_and_force(aaai_site, tmp_path, force):
    """Corrupt caches fall back to fetching; force ignores even valid JSON."""
    plugin, _, get = aaai_site
    path = tmp_path / "aaai.json"
    path.write_text("[]" if force else "not JSON", encoding="utf-8")
    assert len(plugin.download(2026, str(path), force, request_delay=0)) == 2
    assert get.called


@pytest.mark.parametrize("year", [2009, datetime.now().year + 1])
def test_unsupported_year(aaai_site, year):
    """Reject years outside the supported proceedings range."""
    plugin, _, _ = aaai_site
    with pytest.raises(ValueError, match="not supported"):
        plugin.download(year, request_delay=0)


@pytest.mark.parametrize(
    "failed_path", ["/issue/archive/2", "/issue/view/729", "/article/view/36960"]
)
def test_network_failure_does_not_replace_cache(aaai_site, tmp_path, failed_path):
    """Failures at any stage abort and leave an existing cache unchanged."""
    plugin, _, get = aaai_site
    path = tmp_path / "aaai.json"
    original = "[]"
    path.write_text(original, encoding="utf-8")
    respond = get.side_effect

    def fail(url, **kwargs):
        if url == BASE + failed_path:
            raise requests.Timeout("offline")
        return respond(url, **kwargs)

    get.side_effect = fail
    with pytest.raises(RuntimeError, match="Failed to fetch AAAI"):
        plugin.download(2026, str(path), force_download=True, request_delay=0)
    assert path.read_text(encoding="utf-8") == original


def test_http_error(aaai_site):
    """HTTP error statuses produce the same contextual failure as timeouts."""
    plugin, _, get = aaai_site
    response = Mock(text="Unavailable")
    response.raise_for_status.side_effect = requests.HTTPError("503")
    get.side_effect = None
    get.return_value = response
    with pytest.raises(RuntimeError, match="Failed to fetch AAAI"):
        plugin._fetch_page(BASE, 7, True, 0)


@pytest.mark.parametrize(
    "replacement,message",
    [
        ("<html>Changed layout</html>", "No issue summaries"),
        (
            '<div class="obj_issue_summary"><a class="title" href="/x">AAAI</a></div>',
            "Could not determine year",
        ),
    ],
)
def test_invalid_archive(aaai_site, replacement, message):
    """Do not quietly return partial results when archive parsing fails."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/archive/2"] = replacement
    with pytest.raises(RuntimeError, match=message):
        plugin.download(2026, request_delay=0)


def test_archive_pagination_cycle(aaai_site):
    """Fail clearly rather than loop or save partial data on cyclic pagination."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/archive/2"] += (
        '<div class="cmp_pagination"><a class="next" '
        'href="/index.php/AAAI/issue/archive">Next</a></div>'
    )
    with pytest.raises(RuntimeError, match="Repeated AAAI archive"):
        plugin.download(2026, request_delay=0)


def test_missing_year_issues(aaai_site):
    """Report a missing historical year's issues explicitly."""
    plugin, _, _ = aaai_site
    with pytest.raises(RuntimeError, match="No AAAI proceedings issues found"):
        plugin.download(2011, request_delay=0)


def test_invalid_issue(aaai_site):
    """A changed issue layout must not be treated as an empty successful issue."""
    plugin, pages, _ = aaai_site
    pages[BASE + "/issue/view/729"] = "<html>Changed layout</html>"
    with pytest.raises(RuntimeError, match="No proceedings sections"):
        plugin.download(2026, request_delay=0)


def test_no_valid_papers_does_not_write_cache(aaai_site, tmp_path):
    """No valid abstracts should produce an error, not an empty cache."""
    plugin, pages, _ = aaai_site
    for article_id in (36958, 36959, 36960):
        pages[BASE + f"/article/view/{article_id}"] = "<html>No metadata</html>"
    path = tmp_path / "aaai.json"
    with pytest.raises(RuntimeError, match="No valid AAAI papers"):
        plugin.download(2026, str(path), request_delay=0)
    assert not path.exists()


@pytest.mark.parametrize(
    "missing", ["citation_title", "citation_author", "DC.Description"]
)
def test_missing_required_metadata(aaai_site, missing):
    """Invalid title, author, or abstract data is skipped after validation."""
    plugin, pages, _ = aaai_site
    soup = BeautifulSoup(pages[BASE + "/article/view/36958"], "html.parser")
    for tag in soup.select(f'meta[name="{missing}"]'):
        tag.decompose()
    assert (
        plugin._parse_article(soup, BASE + "/article/view/36958", "Track", 2026) is None
    )


@pytest.mark.parametrize(
    "section",
    [
        "Front Matter",
        "IAAI18 - Deployed",
        "EAAI18 - Full Papers",
        "Innovative Applications of Artificial Intelligence",
        "Educational Advances in Artificial Intelligence",
    ],
)
def test_excluded_sections(section):
    """Recognize modern acronyms, older labels, and co-located full names."""
    assert AAAIDownloaderPlugin._EXCLUDED_SECTIONS.search(section)
