# Pydantic Models Refactoring: Moved to Plugin System

**Date**: December 21, 2024  
**Status**: ✅ Complete

## Summary

Moved Pydantic models from `database.py` to the plugin system (`plugins/models.py`) and created lightweight models specifically for the plugin API. This centralizes all data validation models in one place and provides proper model validation for both full schema and lightweight plugin formats.

## Motivation

Previously, Pydantic models (`EventMediaModel`, `AuthorModel`, `PaperModel`) were defined in `database.py` and only used for database validation. With the introduction of the plugin system, we needed:

1. **Centralized Models**: Single source of truth for data validation
2. **Plugin-Specific Models**: Lightweight models for simplified plugin development
3. **Reusability**: Models accessible to plugins, database, and other modules
4. **Type Safety**: Strong validation for both full schema and lightweight formats

## Changes Made

### 1. Created New Models Module (`src/neurips_abstracts/plugins/models.py`)

#### Lightweight Schema Models (New)

**`LightweightAuthor`**
- Simple author model for plugins
- Fields: `name` (required), `affiliation` (optional), `email` (optional)
- Validates that author name is not empty

**`LightweightPaper`**
- Simplified paper model for lightweight plugins
- **Required fields**: `title`, `authors`, `abstract`, `session`, `poster_position`
- **Optional fields**: `id`, `paper_pdf_url`, `poster_image_url`, `url`, `room_name`, `keywords`, `starttime`, `endtime`, `award`
- Validates all required fields are present and non-empty
- Supports both string authors and dict authors
- Extra fields allowed via `ConfigDict(extra="allow")`

#### Full Schema Models (Moved)

**`EventMediaModel`** - Moved from `database.py`
- Full event media validation
- All optional fields for flexibility
- Supports posters, PDFs, URLs, images

**`AuthorModel`** - Moved from `database.py`
- Full author validation
- Required: `id`, `fullname`
- Optional: `url`, `institution`
- Validates fullname is not empty

**`PaperModel`** - Moved from `database.py`
- Complete NeurIPS schema validation
- Required: `id`, `name`
- ~40 optional fields for full schema support
- Validates paper name is not empty
- Validates and converts ID to integer
- Extra fields allowed

#### Validation Helper Functions (New)

```python
validate_lightweight_paper(paper: Dict) -> LightweightPaper
validate_lightweight_papers(papers: List[Dict]) -> List[LightweightPaper]
validate_paper(paper: Dict) -> PaperModel
validate_papers(papers: List[Dict]) -> List[PaperModel]
```

These functions provide convenient validation with proper error handling.

### 2. Updated Plugin System (`src/neurips_abstracts/plugins/__init__.py`)

**Imports Added:**
```python
from .models import (
    LightweightAuthor,
    LightweightPaper,
    EventMediaModel,
    AuthorModel,
    PaperModel,
    validate_lightweight_paper,
    validate_lightweight_papers,
    validate_paper,
    validate_papers,
)
```

**Exports Added to `__all__`:**
- Model classes: `LightweightAuthor`, `LightweightPaper`, `EventMediaModel`, `AuthorModel`, `PaperModel`
- Validation functions: `validate_lightweight_paper`, `validate_lightweight_papers`, `validate_paper`, `validate_papers`

### 3. Updated Database Module (`src/neurips_abstracts/database.py`)

**Before:**
```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

class EventMediaModel(BaseModel):
    # ... 230+ lines of model definitions ...

class AuthorModel(BaseModel):
    # ...

class PaperModel(BaseModel):
    # ...
```

**After:**
```python
from pydantic import ValidationError

# Import Pydantic models from plugin system
from neurips_abstracts.plugins.models import (
    EventMediaModel,
    AuthorModel,
    PaperModel,
)
```

**Result:** Removed ~230 lines of duplicated model code from database.py

### 4. Created Comprehensive Test Suite (`tests/test_plugins_models.py`)

Added 28 tests covering:

#### Lightweight Model Tests (11 tests)
- **LightweightAuthor**: Valid creation, validation errors, affiliation support
- **LightweightPaper**: Required fields, optional fields, validation, mixed author formats

#### Full Schema Model Tests (7 tests)
- **AuthorModel**: Valid creation, empty fullname validation
- **PaperModel**: Minimal creation, full fields, empty name validation, ID conversion
- **EventMediaModel**: Valid media creation

#### Validation Helper Tests (6 tests)
- Single paper validation (lightweight and full)
- Multiple papers validation (lightweight and full)
- Invalid data error handling

#### Integration Tests (4 tests)
- Model to dict conversion
- Extra fields support

**Test Coverage**: 98% on `plugins/models.py` (136 statements, 3 missed)

## API Usage

### For Plugin Developers

#### Using Lightweight Models

```python
from neurips_abstracts.plugins import (
    LightweightDownloaderPlugin,
    LightweightPaper,
    validate_lightweight_paper,
    convert_lightweight_to_neurips_schema,
)

class MyPlugin(LightweightDownloaderPlugin):
    def download(self, year=None, **kwargs):
        # Create papers with validation
        papers_data = [
            {
                'title': 'My Paper',
                'authors': ['Alice', 'Bob'],
                'abstract': 'Paper abstract',
                'session': 'Workshop 2025',
                'poster_position': 'A1',
                'award': 'Best Paper',  # Optional
            }
        ]
        
        # Option 1: Validate manually
        for paper_data in papers_data:
            paper = validate_lightweight_paper(paper_data)
            print(f"Validated: {paper.title}")
        
        # Option 2: Use dicts directly (validation in conversion)
        return convert_lightweight_to_neurips_schema(
            papers_data,
            session_default='Workshop 2025',
            event_type='Poster',
        )
```

#### Using Full Schema Models

```python
from neurips_abstracts.plugins import (
    DownloaderPlugin,
    PaperModel,
    AuthorModel,
    validate_paper,
)

class MyFullPlugin(DownloaderPlugin):
    def download(self, year=None, **kwargs):
        # Create paper with full validation
        paper_data = {
            'id': 1,
            'name': 'My Paper',
            'authors': [
                {'id': 1, 'fullname': 'Alice Smith', 'institution': 'MIT'},
                {'id': 2, 'fullname': 'Bob Jones', 'institution': 'Stanford'},
            ],
            'abstract': 'Full abstract',
            'session': 'Main Conference',
            'decision': 'Accept (oral)',
        }
        
        # Validate
        paper = validate_paper(paper_data)
        
        return {
            'count': 1,
            'results': [paper.model_dump()]
        }
```

### For Database Operations

```python
from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.plugins.models import PaperModel, ValidationError

db = DatabaseManager('papers.db')
db.connect()

# Load data with automatic validation
try:
    db.load_json_data(data)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Benefits

### 1. Centralized Validation
- Single source of truth for all data models
- Consistent validation across plugins and database
- Easier to maintain and update schemas

### 2. Better Developer Experience
- Clear documentation in one place
- Type hints and IDE autocomplete
- Validation errors with helpful messages

### 3. Plugin Development
- Lightweight models simplify plugin creation
- Strong validation catches errors early
- Helper functions reduce boilerplate

### 4. Code Organization
- Logical separation: models in plugins, database uses them
- Reduced duplication (~230 lines removed from database.py)
- Models accessible from single import location

### 5. Type Safety
- Pydantic provides runtime type checking
- ValidationError exceptions for invalid data
- Automatic type coercion (e.g., string ID to int)

## Test Results

### Model Tests
```
tests/test_plugins_models.py::TestLightweightAuthor .............. (4 tests)
tests/test_plugins_models.py::TestLightweightPaper ............... (7 tests)
tests/test_plugins_models.py::TestAuthorModel .................... (2 tests)
tests/test_plugins_models.py::TestPaperModel ..................... (5 tests)
tests/test_plugins_models.py::TestEventMediaModel ................ (1 test)
tests/test_plugins_models.py::TestValidationHelpers .............. (6 tests)
tests/test_plugins_models.py::TestModelIntegration ............... (3 tests)

============================== 28 passed in 0.41s ==============================
Coverage: 98% on plugins/models.py
```

### Existing Tests Still Pass
```
tests/test_plugins_ml4ps.py ............ 32 passed (plugin tests)
tests/test_database.py ................. 15 passed (database tests)
```

## Model Field Comparison

### Lightweight vs Full Schema

| Feature               | Lightweight     | Full Schema |
| --------------------- | --------------- | ----------- |
| Required fields       | 5               | 2           |
| Optional fields       | 9               | ~40         |
| Validation strictness | Required fields | All fields  |
| Use case              | Simple plugins  | Full data   |
| Extra fields allowed  | Yes             | Yes         |
| ID auto-generation    | Yes             | No          |

### Field Mapping

| Lightweight Field  | Full Schema Field | Notes                  |
| ------------------ | ----------------- | ---------------------- |
| `title`            | `name`            | Paper title            |
| `authors`          | `authors`         | List format differs    |
| `abstract`         | `abstract`        | Same                   |
| `session`          | `session`         | Same                   |
| `poster_position`  | `poster_position` | Same                   |
| `award`            | `decision`        | Maps to decision field |
| `url`              | `url`             | Same                   |
| `paper_pdf_url`    | `paper_pdf_url`   | Same                   |
| `poster_image_url` | -                 | Mapped to eventmedia   |
| `keywords`         | `keywords`        | Same                   |
| `room_name`        | `room_name`       | Same                   |
| `starttime`        | `starttime`       | Same                   |
| `endtime`          | `endtime`         | Same                   |

## Migration Guide

### No Breaking Changes

This refactoring is **backward compatible**. All existing code continues to work:

```python
# This still works (imports from plugins now)
from neurips_abstracts.plugins import PaperModel, AuthorModel

# This also works (database imports from plugins)
from neurips_abstracts.database import DatabaseManager
```

### Recommended Updates

If you're using models directly, update imports for clarity:

**Before:**
```python
from neurips_abstracts.database import PaperModel, AuthorModel
```

**After:**
```python
from neurips_abstracts.plugins import PaperModel, AuthorModel
# or
from neurips_abstracts.plugins.models import PaperModel, AuthorModel
```

## Files Modified

1. **`src/neurips_abstracts/plugins/models.py`** - NEW (470 lines)
   - Added `LightweightAuthor` model
   - Added `LightweightPaper` model
   - Moved `EventMediaModel` from database.py
   - Moved `AuthorModel` from database.py
   - Moved `PaperModel` from database.py
   - Added validation helper functions

2. **`src/neurips_abstracts/plugins/__init__.py`** - MODIFIED
   - Added model imports
   - Updated `__all__` exports

3. **`src/neurips_abstracts/database.py`** - MODIFIED
   - Removed model definitions (~230 lines)
   - Added import from plugins.models
   - Removed Pydantic imports (except ValidationError)

4. **`tests/test_plugins_models.py`** - NEW (370 lines)
   - 28 comprehensive tests
   - 98% coverage on models module

## Files Created

- `src/neurips_abstracts/plugins/models.py`
- `tests/test_plugins_models.py`
- `changelog/103_PYDANTIC_MODELS_REFACTORING.md` - This file

## Related Changelogs

- Changelog 97: Plugin System Integration
- Changelog 98: Lightweight Plugin API
- Changelog 99: Plugin Documentation
- Changelog 100: ML4PS Lightweight Conversion
- Changelog 101: ML4PS Plugin Tests
- Changelog 102: Lightweight API Award Field

## Future Enhancements

1. **Additional Models**: Consider adding models for:
   - Conference metadata
   - Session information
   - Review data (if applicable)

2. **Stricter Validation**: Option for plugins to enable strict mode:
   - Required abstract length
   - Valid URL formats
   - Date format validation

3. **Model Versioning**: Support for multiple schema versions:
   - Legacy format support
   - Automatic migration between versions

4. **Serialization**: Enhanced serialization options:
   - JSON Schema export
   - OpenAPI spec generation
   - GraphQL schema generation
