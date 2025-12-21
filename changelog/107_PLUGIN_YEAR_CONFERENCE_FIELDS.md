# Plugin Enhancement: Year and Conference Fields

## Summary

Modified the NeurIPS and ML4PS plugins to populate the `year` and `conference` fields in the database for all downloaded papers. This enables filtering and querying papers by year and conference/workshop.

## Changes

### NeurIPS Plugin (`src/neurips_abstracts/plugins/neurips_downloader.py`)

- Modified `download()` method to add `year` and `conference` fields to each paper in the results
- `year` is set to the conference year parameter (e.g., 2024, 2025)
- `conference` is set to `"NeurIPS"`

### ML4PS Plugin (`src/neurips_abstracts/plugins/ml4ps_downloader.py`)

- Modified `_convert_to_lightweight_format()` method to include `year` and `conference` fields
- `year` is set to `2025` (the currently supported year)
- `conference` is set to `"ML4PS"`

### Tests (`tests/test_plugin_year_conference.py`)

- Added comprehensive test suite for year and conference field population
- Tests verify that both plugins correctly set these fields
- Tests ensure existing paper fields are preserved

## Impact

### Users

- Can now filter papers by year: `SELECT * FROM papers WHERE year = 2025`
- Can distinguish between different conferences/workshops in the same database
- Web UI can display year and conference information
- Better organization of multi-year or multi-conference datasets

### Developers

- Database schema already supported these fields but they were not being populated
- Plugins now consistently populate these optional but useful fields
- Consistent data structure across all plugins

## Technical Details

### Database Schema

The `papers` table already had these fields defined:

- `year INTEGER` - Conference/workshop year
- `conference TEXT` - Conference/workshop name

These fields were indexed for efficient querying but were not being populated by the plugins.

### Implementation

**NeurIPS Plugin:**

```python
# Add year and conference fields to each paper
if "results" in data and isinstance(data["results"], list):
    for paper in data["results"]:
        paper["year"] = year
        paper["conference"] = "NeurIPS"
```

**ML4PS Plugin:**

```python
lightweight_paper = {
    # ... other fields ...
    "year": 2025,
    "conference": "ML4PS",
}
```

## Testing

All tests pass successfully:

```bash
$ pytest tests/test_plugin_year_conference.py -v
================================== test session starts ==================================
tests/test_plugin_year_conference.py::TestNeurIPSPluginYearConference::test_neurips_plugin_adds_year_and_conference PASSED
tests/test_plugin_year_conference.py::TestNeurIPSPluginYearConference::test_neurips_plugin_preserves_existing_fields PASSED
tests/test_plugin_year_conference.py::TestML4PSPluginYearConference::test_ml4ps_lightweight_format_includes_year_and_conference PASSED
tests/test_plugin_year_conference.py::TestML4PSPluginYearConference::test_ml4ps_lightweight_format_preserves_fields PASSED
================================== 4 passed in 0.41s ==================================
```

Integration tests also pass:

```bash
$ pytest tests/test_integration.py -v -k "download"
================================== 2 passed ==================================
```

## Usage Example

After downloading data with the updated plugins:

```python
from neurips_abstracts import DatabaseManager

# Open database
with DatabaseManager("neurips_2025.db") as db:
    # Query papers by year
    papers_2025 = db.query("SELECT * FROM papers WHERE year = 2025")
    
    # Query papers by conference
    neurips_papers = db.query("SELECT * FROM papers WHERE conference = 'NeurIPS'")
    ml4ps_papers = db.query("SELECT * FROM papers WHERE conference = 'ML4PS'")
    
    # Combined queries
    neurips_2024 = db.query(
        "SELECT * FROM papers WHERE year = 2024 AND conference = 'NeurIPS'"
    )
```

## Notes

- The changes are backward compatible - existing databases will work fine
- The fields were already in the schema and indexed, just not populated
- Future plugins should follow this pattern and populate these fields
- For multi-year or multi-conference databases, these fields are essential for filtering

## Related

- Database schema defined in `src/neurips_abstracts/database.py`
- Plugin framework in `src/neurips_abstracts/plugin.py`
- See changelog entries 100-106 for plugin system development history
