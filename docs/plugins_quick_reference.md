# Plugin API Quick Reference

Quick reference for the NeurIPS Abstracts plugin APIs.

## Lightweight API

### Minimal Example

```python
from neurips_abstracts.plugins import (
    LightweightDownloaderPlugin,
    convert_lightweight_to_neurips_schema,
    register_plugin
)

class MyPlugin(LightweightDownloaderPlugin):
    plugin_name = "myplugin"
    plugin_description = "My Plugin"
    supported_years = [2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        self.validate_year(year)
        
        papers = [
            {
                'title': 'Paper Title',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'Paper abstract...',
                'session': 'Session Name',
                'poster_position': 'A1',
            }
        ]
        
        return convert_lightweight_to_neurips_schema(
            papers,
            session_default=f'My Event {year}',
            event_type='Workshop Poster',
            source_url='https://myevent.com'
        )
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }

def _register():
    register_plugin(MyPlugin())

_register()
```

### Required Fields

| Field             | Type | Description                     |
| ----------------- | ---- | ------------------------------- |
| `title`           | str  | Paper title                     |
| `authors`         | list | Author names (strings or dicts) |
| `abstract`        | str  | Paper abstract text             |
| `session`         | str  | Session/workshop name           |
| `poster_position` | str  | Poster identifier               |

### Optional Fields

| Field              | Type | Description                           |
| ------------------ | ---- | ------------------------------------- |
| `id`               | int  | Paper ID (auto-generated if missing)  |
| `paper_pdf_url`    | str  | URL to paper PDF                      |
| `poster_image_url` | str  | URL to poster image                   |
| `url`              | str  | General URL (OpenReview, ArXiv, etc.) |
| `room_name`        | str  | Presentation room                     |
| `keywords`         | list | Keywords/tags                         |
| `starttime`        | str  | Start time                            |
| `endtime`          | str  | End time                              |
| `award`            | str  | Award name (e.g., "Best Paper Award") |

### Author Formats

```python
# Simple strings
'authors': ['John Doe', 'Jane Smith']

# Dicts with name
'authors': [
    {'name': 'John Doe'},
    {'name': 'Jane Smith', 'affiliation': 'MIT'}
]

# Mixed
'authors': ['John Doe', {'name': 'Jane Smith'}]
```

### Converter Function

```python
convert_lightweight_to_neurips_schema(
    papers,                          # List of lightweight papers
    session_default='My Event 2025', # Default session name
    event_type='Workshop Poster',    # Default event type
    source_url='https://...'        # Optional source URL
)
```

## Full Schema API

### Minimal Example

```python
from neurips_abstracts.plugins import DownloaderPlugin, register_plugin

class MyPlugin(DownloaderPlugin):
    plugin_name = "myplugin"
    plugin_description = "My Plugin"
    supported_years = [2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        self.validate_year(year)
        
        results = [
            {
                'id': 1,
                'name': 'Paper Title',
                'abstract': 'Abstract...',
                'authors': [
                    {'name': 'Author One', 'institution': 'MIT'}
                ],
                'session': 'Session Name',
                'event_type': 'Poster',
                'poster_position': 'A1',
                # ... ~35 more fields
            }
        ]
        
        return {
            'count': len(results),
            'next': None,
            'previous': None,
            'results': results
        }
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }

def _register():
    register_plugin(MyPlugin())

_register()
```

### Core Fields (Required)

- `id` (int)
- `name` (str) - Paper title
- `abstract` (str)
- `authors` (list of dicts)
- `session` (str)
- `event_type` (str)
- `poster_position` (str)

### Additional Fields (Optional)

See full schema documentation for ~35 additional fields including URLs, timestamps, media, keywords, etc.

## CLI Commands

### List Plugins

```bash
neurips-abstracts download --list-plugins
```

### Download with Plugin

```bash
# Basic
neurips-abstracts download --plugin myplugin --year 2025 --db-path output.db

# With options
neurips-abstracts download \
    --plugin ml4ps \
    --year 2025 \
    --db-path data/ml4ps_2025.db \
    --fetch-abstracts \
    --max-workers 10
```

## Common Patterns

### Web Scraping

```python
import requests
from bs4 import BeautifulSoup

def download(self, year=None, **kwargs):
    response = requests.get(f'https://event.com/{year}')
    soup = BeautifulSoup(response.content, 'html.parser')
    
    papers = []
    for elem in soup.find_all('div', class_='paper'):
        papers.append({
            'title': elem.find('h2').text.strip(),
            'authors': [a.text for a in elem.find_all('.author')],
            # ...
        })
    
    return convert_lightweight_to_neurips_schema(papers, ...)
```

### Error Handling

```python
def download(self, year=None, **kwargs):
    self.validate_year(year)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch: {e}")
```

### Caching

```python
def download(self, year=None, output_path=None, force_download=False, **kwargs):
    if output_path and Path(output_path).exists() and not force_download:
        return self._load_cached(output_path)
    
    return self._fetch_fresh(year)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def download(self, year=None, **kwargs):
    logger.info(f"Downloading {year}")
    papers = self._fetch(year)
    logger.info(f"Found {len(papers)} papers")
    return papers
```

## Testing

### Unit Test

```python
def test_plugin():
    plugin = MyPlugin()
    data = plugin.download(year=2025)
    
    assert data['count'] > 0
    assert all('name' in p for p in data['results'])
```

### Manual Test

```python
plugin = MyPlugin()
print(plugin.get_metadata())

data = plugin.download(year=2025)
print(f"Papers: {data['count']}")
```

## API Comparison

| Feature         | Lightweight | Full Schema |
| --------------- | ----------- | ----------- |
| Required fields | 5           | ~15         |
| Total fields    | 13          | ~40+        |
| Complexity      | Low         | High        |
| Setup time      | Minutes     | Hours       |
| Auto-conversion | Yes         | N/A         |
| Best for        | Workshops   | Conferences |

## See Also

- [Full Plugin Documentation](plugins.md)
- [Plugin Technical Guide](../src/neurips_abstracts/plugins/README.md)
- [CLI Reference](cli_reference.md)
