# Plugin Module

The plugin module provides the framework for extending Abstracts Explorer
with custom conference data downloaders.

## Overview

The framework includes:

- **Base classes** for plugin implementation (`DownloaderPlugin`, `LightweightDownloaderPlugin`)
- **Schema conversion** utilities for standardizing paper data
- **Plugin registry** for managing and discovering plugins
- **Data validation** via the `LightweightPaper` Pydantic model

## Available Plugins

Built-in plugins for downloading conference data:

| Plugin | Conference | Years | API |
|--------|-----------|-------|-----|
| `neurips` | NeurIPS | 2020–2025 | Full Schema |
| `iclr` | ICLR | 2024–2026 | Lightweight |
| `icml` | ICML | 2024–2025 | Lightweight |
| `ml4ps` | ML4PS Workshop | 2025 | Lightweight |

## Quick Start

```python
from abstracts_explorer.plugin import (
    get_plugin,
    list_plugins,
    LightweightPaper,
)

# List available plugins
for plugin_info in list_plugins():
    print(f"{plugin_info['name']}: {plugin_info['description']}")

# Download papers using a plugin
plugin = get_plugin("neurips")
papers = plugin.download(year=2025)
```

## Creating a Custom Plugin

```python
from abstracts_explorer.plugin import (
    LightweightDownloaderPlugin,
    register_plugin,
)

class MyConferencePlugin(LightweightDownloaderPlugin):
    plugin_name = "myconf"
    plugin_description = "My Conference Downloader"
    supported_years = [2025]

    def download(self, year=None, output_path=None, force_download=False, **kwargs):
        self.validate_year(year)
        # ... fetch and return paper data
        return papers

    def get_metadata(self):
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "supported_years": self.supported_years,
        }

register_plugin(MyConferencePlugin())
```

See the [Plugins Guide](../plugins.md) for detailed instructions.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.plugin
   :members:
   :undoc-members:
   :show-inheritance:
```
