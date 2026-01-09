# Downloader Module

The downloader module handles downloading conference papers from various sources.

## Overview

The `NeurIPSDownloader` class provides methods for:

- Fetching papers from OpenReview API
- Parsing paper metadata
- Caching API responses
- Handling API rate limits and errors

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.downloader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Download

```python
from abstracts_explorer.downloader import NeurIPSDownloader

# Initialize downloader
downloader = NeurIPSDownloader()

# Download papers for a specific year
papers = downloader.get_neurips_papers(year=2025)

print(f"Downloaded {len(papers)} papers")
for paper in papers[:5]:
    print(f"- {paper['title']}")
```

### With Caching

```python
# Enable caching to avoid repeated API calls
downloader = NeurIPSDownloader(use_cache=True)

# First call hits API
papers = downloader.get_neurips_papers(year=2025)

# Second call uses cache
papers = downloader.get_neurips_papers(year=2025)
```

### Error Handling

```python
try:
    papers = downloader.get_neurips_papers(year=2025)
except requests.RequestException as e:
    print(f"Network error: {e}")
except ValueError as e:
    print(f"Invalid data: {e}")
```

## API Integration

The downloader uses the OpenReview API v2:

- **Base URL**: `https://api.openreview.net/notes`
- **Rate Limits**: Respects OpenReview rate limits
- **Authentication**: No authentication required for public papers

### Paper Data Structure

Each paper returned is a dictionary with:

```python
{
    'openreview_id': 'abc123',          # OpenReview ID
    'title': 'Paper Title',              # Paper title
    'abstract': 'Abstract text...',      # Full abstract
    'year': 2025,                        # Conference year
    'pdf_url': 'https://...',            # PDF download URL
    'authors': ['Name1', 'Name2'],       # List of authors
}
```

## Caching

The downloader can cache API responses to reduce network calls:

- Cache files stored in `data/` directory
- Cache filename: `neurips_{year}_cached.json`
- Cache invalidation: Manual or via `force=True` parameter

### Cache Management

```python
# Use cached data if available
downloader = NeurIPSDownloader(use_cache=True)
papers = downloader.get_neurips_papers(year=2025)

# Force fresh download
papers = downloader.get_neurips_papers(year=2025, force=True)
```

## Best Practices

1. **Use caching** during development to avoid repeated API calls
2. **Handle network errors** gracefully with try-except blocks
3. **Respect rate limits** - avoid rapid sequential calls
4. **Validate data** after download before storing in database
