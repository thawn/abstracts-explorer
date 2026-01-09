# Plugin System

Abstracts Explorer includes an extensible plugin system that allows you to download papers from multiple sources beyond the official NeurIPS conference.

## Overview

The plugin system provides two APIs:

1. **Full Schema API** (`DownloaderPlugin`) - For complex data sources with rich metadata
2. **Lightweight API** (`LightweightDownloaderPlugin`) - For simple workshops and conferences

## Available Plugins

### neurips

Official NeurIPS conference data downloader.

```bash
uv run neurips-abstracts download --plugin neurips --year 2025 --db-path data/neurips.db
```

- **Years**: 2020-2025
- **Source**: [neurips.cc](https://neurips.cc/)
- **Fields**: Full schema (~40+ fields)

### ml4ps

ML4PS (Machine Learning for Physical Sciences) workshop downloader.

```bash
uv run neurips-abstracts download --plugin ml4ps --year 2025 --db-path data/ml4ps.db
```

- **Years**: 2025
- **Source**: [ML4PS Workshop](https://ml4physicalsciences.github.io/)
- **Fields**: Full schema with abstracts from NeurIPS virtual site

## Using Plugins via CLI

### List Available Plugins

```bash
uv run neurips-abstracts download --list-plugins
```

### Download with a Plugin

```bash
# Basic usage
uv run neurips-abstracts download --plugin ml4ps --year 2025 --db-path data/output.db

# With options
uv run neurips-abstracts download \
    --plugin ml4ps \
    --year 2025 \
    --db-path data/ml4ps_2025.db \
    --fetch-abstracts \
    --max-workers 10
```

## Creating Your Own Plugin

### When to Use Each API

**Use Full Schema API when:**
- Downloading from official sources with rich metadata
- You need precise control over all ~40+ schema fields
- Handling complex data transformations
- Source provides detailed information (PDFs, posters, event times, etc.)

**Use Lightweight API when:**
- Scraping workshops or small conferences
- Only essential paper information is available
- You want quick plugin development
- Minimal boilerplate code

### Lightweight API (Recommended)

The Lightweight API requires only 5 fields per paper and optionally supports 8 more.

#### Required Fields

- `title` (str): Paper title
- `authors` (list): List of author names (strings or dicts with 'name' key)
- `abstract` (str): Paper abstract
- `session` (str): Session/workshop name
- `poster_position` (str): Poster identifier or position

#### Optional Fields

- `id` (int): Paper ID (auto-generated if not provided)
- `paper_pdf_url` (str): URL to paper PDF
- `poster_image_url` (str): URL to poster image
- `url` (str): General URL (OpenReview, ArXiv, etc.)
- `room_name` (str): Presentation room
- `keywords` (list): Keywords/tags
- `starttime` (str): Start time
- `endtime` (str): End time
- `award` (str): Award name (e.g., "Best Paper Award", "Outstanding Paper")

#### Example: Simple Workshop Plugin

Create `my_workshop_plugin.py`:

```python
from neurips_abstracts.plugins import (
    LightweightDownloaderPlugin,
    convert_lightweight_to_neurips_schema,
    register_plugin
)
import requests
from bs4 import BeautifulSoup


class MyWorkshopPlugin(LightweightDownloaderPlugin):
    """Downloader for My Workshop."""
    
    plugin_name = "myworkshop"
    plugin_description = "My Workshop downloader"
    supported_years = [2024, 2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        """Download papers from My Workshop website."""
        self.validate_year(year)
        
        # Scrape workshop website
        url = f'https://myworkshop.com/{year}/papers'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        papers = []
        for paper_elem in soup.find_all('div', class_='paper'):
            paper = {
                # Required fields
                'title': paper_elem.find('h2', class_='title').text.strip(),
                'authors': [
                    a.text.strip() 
                    for a in paper_elem.find_all('span', class_='author')
                ],
                'abstract': paper_elem.find('p', class_='abstract').text.strip(),
                'session': paper_elem.get('data-session', 'Main Track'),
                'poster_position': paper_elem.get('data-poster', 'TBD'),
                
                # Optional fields
                'paper_pdf_url': paper_elem.find('a', class_='pdf')['href'] if paper_elem.find('a', class_='pdf') else None,
                'keywords': [
                    tag.text.strip() 
                    for tag in paper_elem.find_all('span', class_='tag')
                ],
            }
            papers.append(paper)
        
        # Convert to full NeurIPS schema
        return convert_lightweight_to_neurips_schema(
            papers,
            session_default=f'My Workshop {year}',
            event_type='Workshop Poster',
            source_url=url
        )
    
    def get_metadata(self):
        """Return plugin metadata."""
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }


# Auto-register the plugin
def _register():
    register_plugin(MyWorkshopPlugin())

_register()
```

#### Using Your Plugin

```python
# Import to register
from my_workshop_plugin import MyWorkshopPlugin

# Or use via CLI
# uv run neurips-abstracts download --plugin myworkshop --year 2025 --db-path data/workshop.db
```

#### Flexible Author Format

The Lightweight API accepts authors in multiple formats:

```python
# Simple strings
'authors': ['John Doe', 'Jane Smith']

# Dicts with name
'authors': [
    {'name': 'John Doe', 'affiliation': 'MIT'},
    {'name': 'Jane Smith', 'affiliation': 'Stanford'}
]

# Mixed (will be converted to strings)
'authors': [
    'John Doe',
    {'name': 'Jane Smith'}
]
```

### Full Schema API

For more complex cases where you need complete control over all fields.

#### Example: Advanced Plugin

```python
from neurips_abstracts.plugins import DownloaderPlugin, register_plugin
import requests


class AdvancedPlugin(DownloaderPlugin):
    """Advanced plugin with full schema control."""
    
    plugin_name = "advanced"
    plugin_description = "Advanced conference downloader"
    supported_years = [2024, 2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        """Download papers with full schema."""
        self.validate_year(year)
        
        # Your complex data fetching logic
        papers_data = self._fetch_from_api(year)
        
        # Build full schema
        results = []
        for raw_paper in papers_data:
            paper = {
                # Core fields
                'id': raw_paper['id'],
                'name': raw_paper['title'],
                'abstract': raw_paper['abstract'],
                
                # Author information
                'authors': [
                    {
                        'name': author['full_name'],
                        'institution': author.get('affiliation', ''),
                        'email': author.get('email', '')
                    }
                    for author in raw_paper['authors']
                ],
                
                # Event information
                'session': raw_paper.get('session_name', f'Conference {year}'),
                'event_type': raw_paper.get('presentation_type', 'Poster'),
                'poster_position': raw_paper.get('poster_id', 'TBD'),
                'room_name': raw_paper.get('room', ''),
                
                # URLs and media
                'paper_pdf_url': raw_paper.get('pdf_link'),
                'poster_image_url': raw_paper.get('poster_image'),
                'openreview_url': raw_paper.get('openreview_link'),
                
                # Metadata
                'keywords': raw_paper.get('keywords', []),
                'tldr': raw_paper.get('summary', ''),
                
                # Timestamps
                'starttime': raw_paper.get('start_time'),
                'endtime': raw_paper.get('end_time'),
                
                # Additional fields...
            }
            results.append(paper)
        
        return {
            'count': len(results),
            'next': None,
            'previous': None,
            'results': results
        }
    
    def get_metadata(self):
        """Return plugin metadata."""
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }
    
    def _fetch_from_api(self, year):
        """Fetch data from external API."""
        response = requests.get(f'https://api.example.com/papers?year={year}')
        return response.json()['papers']


def _register():
    register_plugin(AdvancedPlugin())

_register()
```

## Schema Conversion

The `convert_lightweight_to_neurips_schema()` function automatically converts lightweight format to the full NeurIPS schema:

```python
from neurips_abstracts.plugins import convert_lightweight_to_neurips_schema

papers = [
    {
        'title': 'Paper Title',
        'authors': ['John Doe', 'Jane Smith'],
        'abstract': 'Paper abstract...',
        'session': 'Morning Session',
        'poster_position': 'A1',
        'paper_pdf_url': 'https://example.com/paper.pdf',
    }
]

full_schema_data = convert_lightweight_to_neurips_schema(
    papers,
    session_default='Workshop 2025',
    event_type='Workshop Poster',
    source_url='https://workshop.com/2025'
)

# Result has full schema with ~40+ fields
print(full_schema_data['count'])  # 1
print(full_schema_data['results'][0]['name'])  # 'Paper Title'
print(full_schema_data['results'][0]['eventmedia'])  # Generated from URLs
```

### Converter Parameters

- `papers` (list): List of lightweight format papers
- `session_default` (str): Default session name if not provided
- `event_type` (str): Default event type (e.g., 'Workshop Poster', 'Conference Talk')
- `source_url` (str, optional): Source URL for reference

## Plugin Installation

### From Package

If your plugin is in the package's `src/neurips_abstracts/plugins/` directory, it will be auto-discovered when the package loads.

### External Plugin

For plugins outside the package:

```python
# In your code
from neurips_abstracts.plugins import register_plugin
from my_external_plugin import MyPlugin

register_plugin(MyPlugin())
```

Or set `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/plugins:$PYTHONPATH
uv run neurips-abstracts download --plugin myplugin --year 2025
```

## Testing Your Plugin

### Unit Test Example

```python
import pytest
from my_plugin import MyPlugin


def test_plugin_metadata():
    """Test plugin metadata."""
    plugin = MyPlugin()
    metadata = plugin.get_metadata()
    
    assert metadata['name'] == 'myplugin'
    assert 2025 in metadata['supported_years']


def test_plugin_download():
    """Test plugin download."""
    plugin = MyPlugin()
    data = plugin.download(year=2025)
    
    # Check schema
    assert 'count' in data
    assert 'results' in data
    assert data['count'] == len(data['results'])
    
    # Check paper structure
    if data['results']:
        paper = data['results'][0]
        assert 'name' in paper
        assert 'abstract' in paper
        assert 'authors' in paper


def test_invalid_year():
    """Test invalid year handling."""
    plugin = MyPlugin()
    
    with pytest.raises(ValueError):
        plugin.download(year=2099)
```

### Manual Testing

```python
from my_plugin import MyPlugin

# Create instance
plugin = MyPlugin()

# Test metadata
print(plugin.get_metadata())

# Test download
data = plugin.download(year=2025)
print(f"Downloaded {data['count']} papers")

# Verify first paper
if data['results']:
    paper = data['results'][0]
    print(f"Title: {paper['name']}")
    print(f"Authors: {len(paper['authors'])} authors")
    print(f"Abstract length: {len(paper['abstract'])} chars")
```

## Best Practices

### 1. Error Handling

```python
def download(self, year=None, **kwargs):
    self.validate_year(year)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data: {e}")
    
    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response: {e}")
```

### 2. Caching

```python
def download(self, year=None, output_path=None, force_download=False, **kwargs):
    # Check if already downloaded
    if output_path and Path(output_path).exists() and not force_download:
        logger.info(f"Data already exists at {output_path}")
        # Load and return cached data
        return self._load_cached_data(output_path)
    
    # Download fresh data
    return self._fetch_from_source(year)
```

### 3. Logging

```python
import logging

logger = logging.getLogger(__name__)

class MyPlugin(LightweightDownloaderPlugin):
    def download(self, year=None, **kwargs):
        logger.info(f"Starting download for {year}")
        
        papers = self._scrape_papers(year)
        logger.info(f"Found {len(papers)} papers")
        
        return convert_lightweight_to_neurips_schema(papers, ...)
```

### 4. Progress Indication

```python
from tqdm import tqdm

def _scrape_papers(self, year):
    paper_links = self._get_paper_links(year)
    papers = []
    
    for link in tqdm(paper_links, desc="Scraping papers"):
        paper = self._scrape_single_paper(link)
        papers.append(paper)
    
    return papers
```

### 5. Rate Limiting

```python
import time

def _scrape_papers(self, year, delay=1.0):
    papers = []
    
    for link in paper_links:
        paper = self._scrape_single_paper(link)
        papers.append(paper)
        time.sleep(delay)  # Be nice to the server
    
    return papers
```

## API Comparison

| Feature             | Lightweight API              | Full Schema API                 |
| ------------------- | ---------------------------- | ------------------------------- |
| **Required Fields** | 5 fields                     | ~15 fields                      |
| **Total Fields**    | 13 fields                    | ~40+ fields                     |
| **Complexity**      | Low                          | High                            |
| **Setup Time**      | Minutes                      | Hours                           |
| **Use Case**        | Workshops, small conferences | Official conferences, rich data |
| **Auto-conversion** | Yes                          | N/A                             |
| **Author Format**   | Flexible (strings/dicts)     | Strict dict format              |
| **Validation**      | Automatic                    | Manual                          |

## See Also

- [CLI Reference](cli_reference.md) - Command-line interface documentation
- [Usage Guide](usage.md) - General package usage
- [Plugin README](../src/neurips_abstracts/plugins/README.md) - Technical plugin guide
