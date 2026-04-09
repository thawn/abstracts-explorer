# Web Interface

Abstracts Explorer ships with a browser-based UI for searching, chatting, rating,
and visualizing conference abstracts. Start it with:

```bash
abstracts-explorer web-ui          # production (Waitress)
abstracts-explorer web-ui --dev    # development (Flask debug server)
```

Then open <http://127.0.0.1:5000> in your browser.

A live demo is available at [abstracts.hzdr.de](https://abstracts.hzdr.de).

![Web UI overview](images/screenshot.png)

The interface is organized into four tabs described below. A **header bar** at
the top lets you filter globally by conference and year; the selected filters
apply to every tab.

## Search Abstracts

The default tab. Type a query and press **Search** to find matching papers.

**Features:**

- **Keyword search** — matches titles and abstracts by default.
- **Semantic search** — when embeddings are available, results are ranked by
  AI-powered similarity rather than simple keyword matching. Toggle this in
  the settings (gear icon).
- **Filters** — open the settings modal to narrow results by session/track.
  You can also use the global conference and year selectors in the header.
- **Results per page** — choose 10, 25, 50, or 100 results in settings.
- **Star ratings** — click the stars on any paper card to rate it (1–5).
  Rated papers automatically appear in the *Interesting Papers* tab.

**Example use-cases:**

- Search `"uncertainty quantification"` to find UQ-related papers across all
  loaded conferences.
- Filter to *ICLR 2025* in the header, then search `"vision transformer"` to
  see only that conference's results.

## AI Chat

An interactive RAG (Retrieval-Augmented Generation) assistant that answers
questions about the loaded abstracts.

**Features:**

- **Conversational interface** — ask follow-up questions; the assistant
  remembers the conversation context.
- **Relevant papers panel** — papers retrieved as context for each answer are
  displayed in a side panel (desktop) or accessible via a button (mobile).
- **MCP tool integration** — when clustering data is available, the assistant
  can automatically call clustering tools to answer questions about topics,
  trends, and developments across conferences.
- **Settings** — adjust the number of abstracts used as context (3–50) and
  filter by session/track via the gear icon.
- **Feedback** — give thumbs-up/down on individual answers. You can
  optionally donate anonymized chat transcripts to help improve the service.
- **Reset** — click *Reset* to clear conversation history and start fresh.

**Example use-cases:**

- Ask *"What are the main trends in reinforcement learning?"* to get an
  AI-generated summary backed by relevant papers.
- Follow up with *"Show me the top papers on RLHF"* to drill deeper into a
  subtopic.
- Ask *"How has research on diffusion models evolved from 2022 to 2025?"* to
  trigger a trend-analysis tool call.

## Interesting Papers

A personal reading list of papers you have rated.

**Features:**

- **Automatic collection** — every paper you rate with stars (in the Search
  or Chat tabs) is added here.
- **Sorting** — sort by search term, rating, or poster number using the
  dropdown.
- **Session sub-tabs** — papers are organized by session/track for easy
  browsing.
- **Export** — download your collection as a ZIP archive (with Markdown
  files) or as a JSON file.
- **Import** — load a previously saved JSON file to restore your list on
  another device or browser.
- **Donate data** — optionally share your anonymized ratings to help improve
  the service.

**Example use-cases:**

- Rate papers while browsing search results, then switch to this tab to
  review your shortlist before a conference poster session.
- Export the list as a ZIP to share an annotated reading list with
  colleagues.

## Clusters

Interactive 2-D visualization of paper embeddings grouped by topic.

**Features:**

- **Scatter plot** — each dot represents a paper; colors indicate
  automatically identified topic clusters. Hover over a dot to see the
  paper title; click to view full details below the plot.
- **Legend** — cluster names (generated via TF-IDF keywords or LLM labeling)
  are listed alongside the plot. Click a cluster name to highlight it.
- **Custom query search** — enter a free-text query and a distance radius to
  highlight papers close to that query in embedding space.
- **Export** — download the raw clustering data as JSON for further analysis.

**Example use-cases:**

- Open the Clusters tab to get a bird's-eye view of research topics at a
  conference.
- Type `"graph neural networks"` in the custom query box with a distance of
  1.0 to see which papers are semantically close to that topic.
- Click on an outlier dot to discover an unusual or cross-disciplinary paper
  you might otherwise miss.

## Starting the Web UI

### Command-line options

```bash
abstracts-explorer web-ui [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--host TEXT` | Bind address (default: `127.0.0.1`) |
| `--port INTEGER` | Port number (default: `5000`) |
| `--dev` | Use Flask development server instead of Waitress |
| `-v` / `-vv` | Increase log verbosity |

### Docker / Podman

When running via Docker Compose the web UI is exposed on port 5000 by default.
See the [Docker Guide](docker.md) for details.
