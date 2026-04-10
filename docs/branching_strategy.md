# Branching Strategy

This document describes the Git branching model used by Abstracts Explorer. It is
based on the well-known
[Git-Flow model](https://gist.github.com/Rishav-Git/b774bc5a1e3332395f214b02f1006687)
with simplifications suited to our project.

## Overview

The central repository (`origin`) holds two long-lived branches:

| Branch | Purpose |
|---|---|
| **main** | Production-ready code. Every commit on `main` is a tagged release. |
| **develop** | Integration branch with the latest delivered changes for the next release. |

In addition, short-lived **feature** and **hotfix** branches are created as needed.
We do **not** use release branches — instead, `develop` is tested directly in our
staging environment until it is ready, then merged into `main` with a version tag.

## Branch types

### `main`

* Always reflects production-ready state.
* Only receives merge commits from `develop` (new releases) or `hotfix-*` branches
  (urgent fixes).
* Every merge into `main` is tagged with a version number (e.g. `v1.2.0`).

### `develop`

* Integration branch for the next release.
* Receives merges from feature branches and hotfix back-merges.
* When stable and ready for release, it is merged into `main`.

### Feature branches

| | |
|---|---|
| Branch off from | `develop` |
| Merge back into | `develop` |
| Naming convention | author/brief-description |

Feature branches are used for all new features, enhancements, and major refactoring.
They exist as long as the work is in progress and are deleted after merging.

```bash
# Create a feature branch
git checkout -b my-feature develop

# After work is done, merge back into develop
git checkout develop
git merge --no-ff my-feature
git branch -d my-feature
git push origin develop
```

Always use `--no-ff` (no fast-forward) so that a merge commit is created. This
preserves the history of the feature branch and makes it easy to revert an entire
feature if needed.

### Hotfix branches

| | |
|---|---|
| Branch off from | `main` |
| Merge back into | `main` **and** `develop` |
| Naming convention | `hotfix-*` |

Hotfix branches address critical bugs in the current production release. They are the
**only** branches that may branch off from `main`.

```bash
# Create a hotfix branch from main
git checkout -b hotfix-1.2.1 main

# Fix the issue, then merge into main and tag
git checkout main
git merge --no-ff hotfix-1.2.1
git tag -a v1.2.1 -m "Hotfix: description"

# Also merge into develop
git checkout develop
git merge --no-ff hotfix-1.2.1

# Clean up
git branch -d hotfix-1.2.1
```

## Release workflow

Because we do not use release branches, the release process is straightforward:

1. Features are developed on feature branches and merged into `develop`.
2. When `develop` is considered release-ready, it is deployed to the **staging
   environment**.
3. The [staging end-to-end tests](#staging-end-to-end-tests) are executed against the
   staging deployment.
4. If all tests pass and the release is approved, `develop` is merged into `main`:
   ```bash
   git checkout main
   git merge --no-ff develop
   git tag -a v1.3.0 -m "Release v1.3.0"
   git push origin main --tags
   ```
5. The version tag on `main` marks the new production release.

## Staging end-to-end tests

Before merging `develop` into `main`, the following critical end-to-end tests must
pass on the staging environment. This is a small, focused subset of the full
[e2e test suite](https://github.com/thawn/abstracts-explorer/blob/main/tests/test_web_e2e.py)
designed to catch show-stopping regressions quickly.

### 1. Application startup

| Test | What it verifies |
|---|---|
| **Page loads successfully** | The web UI returns HTTP 200, the page title is correct, and the main application container is present. |
| **No JavaScript errors on load** | The browser console contains no errors after the initial page load. |

### 2. Core search functionality

| Test | What it verifies |
|---|---|
| **Keyword search returns results** | Entering a query and clicking search displays matching papers. |
| **Empty search shows a message** | Submitting an empty search shows appropriate user feedback. |
| **Search with no results** | A query that matches nothing displays a "no results" message instead of an error. |

### 3. Paper display

| Test | What it verifies |
|---|---|
| **Paper detail view** | Clicking a paper shows its title, authors, and abstract. |
| **Collapsible abstract** | The abstract section can be expanded and collapsed. |

### 4. Chat (RAG) interface

| Test | What it verifies |
|---|---|
| **Chat UI elements present** | The chat input, send button, and reset button are visible. |
| **Send a chat message** | Typing a question and clicking send returns a response (or a graceful error if the LLM backend is unavailable). |

#### MCP tool smoke tests

Each MCP tool should be exercised via the chat interface with a representative query.
The test passes when the response contains the listed success criteria (or a graceful
error when the LLM backend is unavailable).

| MCP tool | Example query | Success criteria |
|---|---|---|
| `get_conference_topics` | *"What are the main topics at NeurIPS 2025?"* | Response lists topic names with keywords and paper counts. |
| `get_topic_evolution` | *"How has research on transformers evolved at NeurIPS over the years?"* | Response includes a year-by-year breakdown of paper counts or trend description. |
| `search_papers` | *"Find papers about reinforcement learning at NeurIPS 2025."* | Response returns paper titles with authors and abstracts. |
| `get_paper_details` | *"Who are the authors of 'Attention is All You Need'?"* | Response includes author names, URL/PDF links, and session info (if available). |
| `analyze_topic_relevance` | *"How relevant is uncertainty quantification at NeurIPS 2025?"* | Response contains a relevance score or paper count within the embedding distance threshold. |
| `get_cluster_visualization` | *"Show me a visual overview of NeurIPS 2025 clusters."* | Response returns or references visualization data (Plotly JSON or a rendered plot). |

### 5. Clustering tab

| Test | What it verifies |
|---|---|
| **Clustering tab exists** | The clustering tab is present and can be activated. |
| **Clustering plot loads** | Switching to the clustering tab renders a Plotly visualization (or shows a meaningful placeholder). |

### 6. Accessibility & responsiveness

| Test | What it verifies |
|---|---|
| **Keyboard navigation** | Core interactive elements (search input, tabs) are reachable via keyboard. |
| **Responsive layout** | The page renders without horizontal overflow at a narrow viewport width (e.g. 768 px). |

```{note}
This list is a **first draft** and should evolve as the project matures. The staging
tests are implemented as Selenium tests in
[`tests/test_staging_e2e.py`](https://github.com/thawn/abstracts-explorer/blob/main/tests/test_staging_e2e.py)
and can be run against any deployment URL::

    uv run pytest tests/test_staging_e2e.py -m staging --staging-url http://localhost:5000

Alternatively, set the ``STAGING_URL`` environment variable::

    STAGING_URL=https://staging.example.com uv run pytest tests/test_staging_e2e.py -m staging
```
