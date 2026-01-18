# Download Plugins

This directory contains downloader plugins for the neurips-abstracts package.

## Available Plugins

### neurips

Official NeurIPS conference data downloader.
- **File**: `neurips_downloader.py`
- **Years**: 2013-2025
- **Source**: <https://neurips.cc/>
- **API**: Full schema

### ml4ps

ML4PS (Machine Learning for Physical Sciences) workshop downloader.
- **File**: `ml4ps_downloader.py`
- **Years**: 2025
- **Source**: <https://ml4physicalsciences.github.io/>
- **API**: Full schema

### example_lightweight

Example demonstrating the lightweight plugin API.
- **File**: `example_lightweight.py`
- **Years**: 2024-2025
- **API**: Lightweight schema

## Plugin APIs

The package provides two plugin APIs with different complexity levels:

### Full Schema API (`DownloaderPlugin`)

For complete control and complex data sources.

**Use when:**
- Downloading from official sources with rich metadata
- Need precise control over all ~40+ schema fields
- Handling complex data transformations

**Example:**
```python
from abstracts_explorer.plugins import DownloaderPlugin, register_plugin

class MyFullPlugin(DownloaderPlugin):
    plugin_name = "myplugin"
    plugin_description = "My custom downloader"
    supported_years = [2024, 2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        # Return full NeurIPS schema
        return {
            'count': 0,
            'next': None,
            'previous': None,
            'results': [...]  # Papers with ~40+ fields each
        }
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }
```

### Lightweight API (`LightweightDownloaderPlugin`)

For simple data sources with minimal fields.

**Use when:**
- Scraping workshops, small conferences
- Only essential paper information is available
- Want quick plugin development

**Required fields per paper:**
- `title` (str): Paper title
- `authors` (list): List of author names or dicts
- `abstract` (str): Paper abstract
- `session` (str): Session/workshop name
- `poster_position` (str): Poster identifier

**Optional fields per paper:**
- `id` (int): Paper ID (auto-generated if missing)
- `paper_pdf_url` (str): URL to paper PDF
- `poster_image_url` (str): URL to poster image
- `url` (str): General URL (OpenReview, ArXiv, etc.)
- `room_name` (str): Presentation room
- `keywords` (list): Keywords/tags
- `starttime` (str): Start time
- `endtime` (str): End time

**Example:**
```python
from abstracts_explorer.plugins import (
    LightweightDownloaderPlugin,
    convert_lightweight_to_neurips_schema,
    register_plugin
)

class MyLightweightPlugin(LightweightDownloaderPlugin):
    plugin_name = "myworkshop"
    plugin_description = "My Workshop 2025"
    supported_years = [2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        # Scrape papers in simple format
        papers = [
            {
                'title': 'Paper Title',
                'authors': ['John Doe', 'Jane Smith'],
                'abstract': 'Paper abstract...',
                'session': 'Morning Session',
                'poster_position': 'A1',
                # Optional
                'paper_pdf_url': 'https://example.com/paper.pdf',
                'keywords': ['ML', 'Physics'],
            }
        ]
        
        # Automatic conversion to full schema
        return convert_lightweight_to_neurips_schema(
            papers,
            session_default='My Workshop 2025',
            event_type='Workshop Poster',
            source_url='https://myworkshop.com'
        )
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }
```

## Creating a New Plugin

1. **Choose your API**: Full schema or lightweight
2. **Create a Python file** in this directory
3. **Implement the plugin class**
4. **Register the plugin** (auto-register on import)
5. **Test your plugin**

### Step-by-step Example (Lightweight)

Create `my_workshop_plugin.py`:

```python
from abstracts_explorer.plugins import (
    LightweightDownloaderPlugin,
    convert_lightweight_to_neurips_schema,
    register_plugin
)
import requests
from bs4 import BeautifulSoup

class MyWorkshopPlugin(LightweightDownloaderPlugin):
    plugin_name = "myworkshop"
    plugin_description = "My Workshop downloader"
    supported_years = [2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        self.validate_year(year)
        
        # Scrape your website
        response = requests.get(f'https://myworkshop.com/{year}')
        soup = BeautifulSoup(response.content, 'html.parser')
        
        papers = []
        for paper_elem in soup.find_all('div', class_='paper'):
            papers.append({
                'title': paper_elem.find('h2').text,
                'authors': [a.text for a in paper_elem.find_all('span', class_='author')],
                'abstract': paper_elem.find('p', class_='abstract').text,
                'session': paper_elem.get('data-session', 'General'),
                'poster_position': paper_elem.get('data-poster', 'TBD'),
                'paper_pdf_url': paper_elem.find('a', class_='pdf')['href'],
            })
        
        return convert_lightweight_to_neurips_schema(
            papers,
            session_default=f'My Workshop {year}',
            event_type='Workshop Poster',
            source_url=f'https://myworkshop.com/{year}'
        )
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years
        }

# Auto-register
def _register():
    register_plugin(MyWorkshopPlugin())

_register()
```

Then use it:

```bash
# Import to register
python -c "from abstracts_explorer.plugins import my_workshop_plugin"

# Or via CLI
neurips-abstracts download --plugin myworkshop --year 2025
```

## Plugin Interface Requirements

### All plugins must implement

- `download(year, output_path, force_download, **kwargs)` - Download and return data
- `get_metadata()` - Return plugin information

### All plugins should set

- `plugin_name` - Unique identifier
- `plugin_description` - Human-readable description  
- `supported_years` - List of supported years

## Testing Your Plugin

```python
from your_plugin_module import YourPlugin

# Create instance
plugin = YourPlugin()

# Test metadata
metadata = plugin.get_metadata()
print(f"Plugin: {metadata['name']}")

# Test download
data = plugin.download(year=2025)
print(f"Downloaded {data['count']} papers")

# Verify schema
assert 'count' in data
assert 'results' in data
assert all('title' in p for p in data['results'])
```

## Tips

1. **Use caching**: Check if output file exists before scraping
2. **Handle errors gracefully**: Network issues, missing data, etc.
3. **Add logging**: Use `logger` for debugging
4. **Validate year**: Call `self.validate_year(year)` early
5. **Test thoroughly**: Verify with small samples first
6. **Document**: Add docstrings and comments
