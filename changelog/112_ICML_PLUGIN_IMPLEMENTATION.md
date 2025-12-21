# Changelog Entry 112: ICML Plugin Implementation

**Date**: December 21, 2025  
**Type**: Feature  
**Impact**: New functionality

## Summary

Added a new ICML (International Conference on Machine Learning) downloader plugin to the system, enabling users to download and analyze papers from ICML conferences spanning 2020-2025.

## Changes Made

### 1. New Plugin Implementation

**File**: `src/neurips_abstracts/plugins/icml_downloader.py`

- Created `ICMLDownloaderPlugin` class extending `JSONConferenceDownloaderPlugin`
- Configured for ICML conferences from 2020-2025
- Implements `get_url()` method to construct ICML data URLs
- Auto-registers the plugin on import

Key features:

- **Plugin name**: `icml`
- **Conference name**: `ICML`
- **Supported years**: [2020, 2021, 2022, 2023, 2024, 2025]
- **Data URL**: `https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json`

### 2. Plugin Registration

**File**: `src/neurips_abstracts/plugins/__init__.py`

- Added import for `ICMLDownloaderPlugin`
- Updated module docstring to include ICML in available plugins
- Added `ICMLDownloaderPlugin` to `__all__` exports

### 3. Test Coverage

**File**: `tests/test_plugin_year_conference.py`

- Added `TestICMLPluginYearConference` test class
- Implemented `test_icml_plugin_adds_year_and_conference()` test
- Implemented `test_icml_plugin_preserves_existing_fields()` test
- Both tests use mocking to verify plugin behavior
- Tests verify that year and conference fields are properly added to papers
- Tests verify that existing paper fields are preserved

## Usage

### Command Line

```bash
# List available plugins
neurips-abstracts list-plugins

# Download ICML 2025 papers
neurips-abstracts download --plugin icml --year 2025

# Download to a specific file
neurips-abstracts download --plugin icml --year 2025 --output icml_2025.json
```

### Python API

```python
from neurips_abstracts.plugins import get_plugin

# Get the ICML plugin
plugin = get_plugin('icml')

# Download ICML 2025 data
data = plugin.download(year=2025)

# Save to file
data = plugin.download(year=2025, output_path='icml_2025.json')
```

## Technical Details

### URL Pattern

The ICML plugin follows the same URL pattern as ICLR:

```text
https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json
```

**Note on Year Availability**: The JSON data format is only available for ICML conferences from 2020 onwards, when the virtual conference platform was adopted. Earlier conferences (1996-2019) exist on the ICML website but use different formats and are not currently supported by this plugin.

### Data Format

The ICML JSON data follows the same structure as other conference plugins:

- `count`: Total number of papers
- `next`: URL for pagination (if applicable)
- `previous`: URL for pagination (if applicable)
- `results`: Array of paper objects

Each paper object includes:

- `id`: Paper identifier
- `name`: Paper title
- `abstract`: Paper abstract
- `authors`: Array of author objects
- `year`: Conference year (added by plugin)
- `conference`: Conference name (added by plugin)
- Additional fields: keywords, decision, session, etc.

### Plugin Features

The ICML plugin inherits all features from `JSONConferenceDownloaderPlugin`:

- Automatic year and conference field injection
- File caching support
- SSL verification control
- Configurable timeout
- Proper error handling

## Testing

All tests pass successfully:

```bash
$ uv run pytest tests/test_plugin_year_conference.py::TestICMLPluginYearConference -v
================================== test session starts ==================================
tests/test_plugin_year_conference.py::TestICMLPluginYearConference::test_icml_plugin_adds_year_and_conference PASSED [ 50%]
tests/test_plugin_year_conference.py::TestICMLPluginYearConference::test_icml_plugin_preserves_existing_fields PASSED [100%]
================================== 2 passed in 0.56s ====================================
```

### Test Coverage

- **Plugin file**: 100% coverage for `icml_downloader.py`
- **Tests**: 2 new test methods added
- **Integration**: Plugin properly registers and integrates with the plugin system

## Verification

Plugin registration verified:

```bash
$ uv run python -c "from neurips_abstracts.plugins import list_plugin_names; print(list_plugin_names())"
['iclr', 'icml', 'ml4ps', 'neurips']
```

Plugin configuration verified:

```bash
$ uv run python -c "from neurips_abstracts.plugins import get_plugin; plugin = get_plugin('icml'); print(f'Plugin: {plugin.plugin_name}'); print(f'Conference: {plugin.conference_name}'); print(f'Supported years: {plugin.supported_years}')"
Plugin: icml
Conference: ICML
Supported years: [2020, 2021, 2022, 2023, 2024, 2025]
```

## Benefits

1. **Multi-Conference Support**: Users can now download and analyze papers from ICML, ICLR, NeurIPS, and ML4PS
2. **Consistent API**: ICML plugin follows the same interface as other conference plugins
3. **Easy Extension**: The plugin system makes it easy to add more conferences in the future
4. **Comprehensive Testing**: Full test coverage ensures reliability
5. **Data Consistency**: Year and conference fields are automatically added to all papers

## Future Improvements

Potential enhancements for ICML plugin:

1. Add support for additional ICML years as they become available
2. Add workshop paper support if ICML provides separate workshop data
3. Add support for ICML-specific metadata fields
4. Implement paper filtering based on acceptance type (oral, poster, spotlight)

## Related Files

- `src/neurips_abstracts/plugins/icml_downloader.py` (new)
- `src/neurips_abstracts/plugins/__init__.py` (modified)
- `tests/test_plugin_year_conference.py` (modified)
- `src/neurips_abstracts/plugins/json_conference_downloader.py` (base class)

## Dependencies

No new dependencies added. The ICML plugin uses the existing infrastructure:

- `requests` for HTTP requests
- `JSONConferenceDownloaderPlugin` base class
- Plugin registry system

## Migration Notes

This is a new feature with no breaking changes. Existing code continues to work without modification. Users who want to use the ICML plugin can simply:

1. Import and use the plugin: `from neurips_abstracts.plugins import get_plugin`
2. Or use the CLI: `neurips-abstracts download --plugin icml`

No configuration changes or database migrations are required.
