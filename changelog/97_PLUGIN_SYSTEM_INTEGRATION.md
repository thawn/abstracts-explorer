# Plugin System Integration

## Summary

Successfully integrated `scrape_ml4ps.py` as a download plugin and converted the existing `downloader.py` to use the plugin system. The package now supports multiple data sources through an extensible plugin architecture.

## Changes Made

### 1. Plugin Infrastructure (`src/neurips_abstracts/plugins/__init__.py`)
- Created base `DownloaderPlugin` abstract class
- Implemented `PluginRegistry` for managing plugins
- Added helper functions: `register_plugin()`, `get_plugin()`, `list_plugins()`, `list_plugin_names()`

### 2. NeurIPS Plugin (`src/neurips_abstracts/plugins/neurips_downloader.py`)
- Wrapped existing `download_neurips_data()` functionality as a plugin
- Plugin name: `neurips`
- Supports years: 2013-2025
- Auto-registers on import

### 3. ML4PS Plugin (`src/neurips_abstracts/plugins/ml4ps_downloader.py`)
- Migrated `scrape_ml4ps.py` functionality into plugin
- Plugin name: `ml4ps`
- Supports year: 2025
- Features:
  - Scrapes ML4PS workshop website
  - Fetches abstracts from NeurIPS virtual site
  - Parallel abstract fetching with configurable workers
  - Extracts OpenReview URLs
- Auto-registers on import

### 4. CLI Updates (`src/neurips_abstracts/cli.py`)
- Modified `download_command()` to support plugin selection
- Added `--plugin` flag (default: neurips)
- Added `--list-plugins` flag to show available plugins
- Added `--fetch-abstracts` and `--max-workers` flags for ML4PS plugin
- Enhanced help text with plugin examples

### 5. Package Updates (`src/neurips_abstracts/__init__.py`)
- Exported plugin system components
- Added comprehensive plugin system documentation
- Included example usage for creating custom plugins

## Usage

### List Available Plugins

```bash
neurips-abstracts download --list-plugins
```

Output:
```
Available downloader plugins:
======================================================================

📦 neurips
   Official NeurIPS conference data downloader
   Supported years: 2013-2025

📦 ml4ps
   ML4PS (Machine Learning for Physical Sciences) workshop downloader
   Supported years: 2025

======================================================================
```

### Download with NeurIPS Plugin (Default)

```bash
# Default - downloads official NeurIPS data
neurips-abstracts download --year 2025 --output neurips_2025.db

# Explicitly specify neurips plugin
neurips-abstracts download --plugin neurips --year 2025 --output neurips_2025.db
```

### Download with ML4PS Plugin

```bash
# Download ML4PS workshop papers with abstracts
neurips-abstracts download --plugin ml4ps --year 2025 --output ml4ps_2025.db

# Without abstracts (faster)
neurips-abstracts download --plugin ml4ps --year 2025 --output ml4ps_2025.db --fetch-abstracts=False

# With custom parallel workers
neurips-abstracts download --plugin ml4ps --year 2025 --output ml4ps_2025.db --max-workers 30
```

### Python API

```python
from neurips_abstracts.plugins import get_plugin, list_plugins

# List all plugins
plugins = list_plugins()
for plugin in plugins:
    print(f"{plugin['name']}: {plugin['description']}")

# Use NeurIPS plugin
neurips_plugin = get_plugin('neurips')
data = neurips_plugin.download(year=2025, output_path='neurips_2025.json')

# Use ML4PS plugin with options
ml4ps_plugin = get_plugin('ml4ps')
data = ml4ps_plugin.download(
    year=2025,
    output_path='ml4ps_2025.json',
    fetch_abstracts=True,
    max_workers=20
)
```

### Creating Custom Plugins

```python
from neurips_abstracts.plugins import DownloaderPlugin, register_plugin

class MyWorkshopPlugin(DownloaderPlugin):
    plugin_name = "myworkshop"
    plugin_description = "My custom workshop downloader"
    supported_years = [2024, 2025]
    
    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        # Implement your download logic
        papers = []  # Your scraping/download logic here
        
        return {
            'count': len(papers),
            'next': None,
            'previous': None,
            'results': papers
        }
    
    def get_metadata(self):
        return {
            'name': self.plugin_name,
            'description': self.plugin_description,
            'supported_years': self.supported_years,
            'parameters': {
                'year': {'type': 'int', 'required': True, 'default': 2025},
                'output_path': {'type': 'str', 'required': False},
            }
        }

# Register your plugin
register_plugin(MyWorkshopPlugin())
```

## Plugin Features

### Base Plugin Class
- Abstract interface for all downloaders
- Year validation
- Metadata support
- Extensible parameter system

### NeurIPS Plugin
- Downloads official conference data from neurips.cc
- Supports years 2013-2025
- Uses existing `download_neurips_data()` function
- Includes caching and retry logic

### ML4PS Plugin
- Scrapes ML4PS workshop website
- Parallel abstract fetching from NeurIPS virtual site
- OpenReview URL extraction
- Configurable parallel workers
- Award detection (Spotlight, Best Poster, etc.)
- Full metadata in database schema format

## Benefits

1. **Extensibility**: Easy to add new data sources
2. **Consistency**: All plugins follow same interface
3. **Backward Compatibility**: Existing code still works with default neurips plugin
4. **CLI Integration**: Seamless command-line experience
5. **Documentation**: Self-documenting through metadata
6. **Type Safety**: Abstract base class enforces implementation

## Files Created/Modified

### Created:
- `src/neurips_abstracts/plugins/__init__.py` - Plugin infrastructure
- `src/neurips_abstracts/plugins/neurips_downloader.py` - NeurIPS plugin
- `src/neurips_abstracts/plugins/ml4ps_downloader.py` - ML4PS plugin

### Modified:
- `src/neurips_abstracts/cli.py` - Added plugin support to download command
- `src/neurips_abstracts/__init__.py` - Exported plugin system

### Preserved:
- `scrape_ml4ps.py` - Original file preserved for reference
- `src/neurips_abstracts/downloader.py` - Still used by NeurIPS plugin

## Testing

All functionality verified:
- ✅ Plugin registration
- ✅ Plugin discovery via CLI
- ✅ Plugin metadata retrieval
- ✅ NeurIPS plugin download
- ✅ ML4PS plugin with abstract fetching
- ✅ CLI help text
- ✅ Python API

## Future Enhancements

Potential additions:
1. More workshop plugins (ICLR, ICML, etc.)
2. OpenReview API plugin
3. ArXiv plugin
4. Plugin configuration files
5. Plugin versioning
6. Async download support
7. Progress bars for all plugins
8. Plugin dependency management
