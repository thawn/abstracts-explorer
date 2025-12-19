# Download Plugins

This directory contains downloader plugins for the neurips-abstracts package.

## Available Plugins

### neurips
Official NeurIPS conference data downloader.
- **File**: `neurips_downloader.py`
- **Years**: 2013-2025
- **Source**: https://neurips.cc/

### ml4ps
ML4PS (Machine Learning for Physical Sciences) workshop downloader.
- **File**: `ml4ps_downloader.py`
- **Years**: 2025
- **Source**: https://ml4physicalsciences.github.io/

## Creating a New Plugin

1. Create a new Python file in this directory
2. Import the base class:
   ```python
   from neurips_abstracts.plugins import DownloaderPlugin, register_plugin
   ```

3. Implement your plugin:
   ```python
   class MyPlugin(DownloaderPlugin):
       plugin_name = "myplugin"
       plugin_description = "My custom downloader"
       supported_years = [2024, 2025]
       
       def download(self, year=None, output_path=None, force_download=False, **kwargs):
           # Your implementation
           return {'count': 0, 'next': None, 'previous': None, 'results': []}
       
       def get_metadata(self):
           return {
               'name': self.plugin_name,
               'description': self.plugin_description,
               'supported_years': self.supported_years
           }
   ```

4. Register your plugin:
   ```python
   def _register():
       register_plugin(MyPlugin())
   
   _register()
   ```

5. Import your plugin in the CLI or wherever needed:
   ```python
   import neurips_abstracts.plugins.my_plugin
   ```

## Plugin Interface

All plugins must implement:
- `download(year, output_path, force_download, **kwargs)` - Download and return data
- `get_metadata()` - Return plugin information

All plugins should set:
- `plugin_name` - Unique identifier
- `plugin_description` - Human-readable description  
- `supported_years` - List of supported years
