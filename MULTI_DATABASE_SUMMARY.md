# Multi-Database Backend Support - Implementation Summary

## Overview

This document summarizes the implementation of multi-database backend support for abstracts-explorer, enabling the use of both SQLite and PostgreSQL databases.

## Changes Made

### 1. Dependencies

**Added to `pyproject.toml`:**
- `sqlalchemy>=2.0.0` - Core ORM dependency
- `postgres` optional dependency group with `psycopg2-binary>=2.9.0`

**Installation:**
```bash
# SQLite only (default)
uv sync

# With PostgreSQL support
uv sync --extra postgres
```

### 2. Database Models (`db_models.py`)

Created SQLAlchemy ORM models:
- `Paper` - Main paper table with all fields from LightweightPaper schema
- `EmbeddingsMetadata` - Tracks embedding model versions
- Uses timezone-aware datetime for Python 3.12+ compatibility

### 3. DatabaseManager Refactoring (`database.py`)

**Key Changes:**
- Replaced direct `sqlite3` usage with SQLAlchemy
- Added support for `database_url` parameter (any SQLAlchemy-compatible URL)
- Maintained `db_path` parameter for backward compatibility
- Kept `connection` attribute for legacy test compatibility
- Converted raw SQL parameter format (? → :param0, :param1, etc.)

**API Compatibility:**
- All existing methods work unchanged
- `connection` attribute provides raw database connection for tests
- `db_path` attribute maintained for backward compatibility

### 4. Configuration System (`config.py`)

**New Environment Variables:**
- `DATABASE_URL` - SQLAlchemy database URL (takes precedence)
- `PAPER_DB_PATH` - Legacy SQLite path (still supported)

**Priority Order:**
1. `DATABASE_URL` (if set)
2. `PAPER_DB_PATH` (converted to SQLite URL)
3. Default: `sqlite:///data/abstracts.db`

**Security:**
- Passwords masked in configuration output
- Passwords masked in database connection logs

### 5. Tests

**Existing Tests:**
- ✅ 27/27 database tests passing
- ✅ All maintain backward compatibility

**New Tests (`test_multi_database.py`):**
- `test_sqlite_via_database_url` - SQLite with DATABASE_URL
- `test_sqlite_legacy_path` - SQLite with legacy db_path
- `test_must_provide_db_path_or_url` - Validation
- `test_database_url_takes_precedence` - Priority handling
- `test_postgresql_basic_operations` - PostgreSQL operations (skipped if unavailable)
- `test_postgresql_multiple_papers` - PostgreSQL batch operations (skipped if unavailable)
- `test_postgresql_embedding_model_metadata` - PostgreSQL metadata (skipped if unavailable)
- `test_database_url_in_config` - Configuration system
- `test_legacy_paper_db_path_in_config` - Legacy configuration

**Test Results:**
- 326/329 core tests passing
- 3 failures due to optional web dependencies (not related to database changes)
- 0 deprecation warnings

### 6. Documentation

**Updated Files:**
- `.env.example` - Added DATABASE_URL examples
- `README.md` - Added database backend configuration section

**Usage Examples:**

SQLite (default):
```bash
PAPER_DB_PATH=data/abstracts.db
```

PostgreSQL:
```bash
DATABASE_URL=postgresql://user:password@localhost/abstracts
```

## Migration Guide

### For Existing Users

No changes required! The system automatically works with existing SQLite databases:

```bash
# Existing configuration still works
PAPER_DB_PATH=data/abstracts.db
```

### For PostgreSQL Users

1. Install PostgreSQL support:
```bash
uv sync --extra postgres
```

2. Create database:
```bash
createdb abstracts
```

3. Configure in `.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost/abstracts
```

4. Run application as normal - tables created automatically

## Testing PostgreSQL

To run PostgreSQL tests (requires PostgreSQL server):

```bash
# Set PostgreSQL connection URL
export POSTGRES_TEST_URL=postgresql://user:password@localhost/test_db

# Run tests
uv run pytest tests/test_multi_database.py -v
```

## Benefits

1. **Flexibility** - Choose database backend based on needs
2. **Scalability** - PostgreSQL for larger datasets
3. **Compatibility** - Full backward compatibility with SQLite
4. **Standards** - Uses industry-standard SQLAlchemy ORM
5. **Security** - Password masking in logs and configuration

## Technical Details

### SQLAlchemy Architecture

- **Engine** - Manages database connections
- **Session** - Handles transactions
- **ORM Models** - Type-safe database operations
- **Raw Connection** - Available for legacy compatibility

### Database URL Format

SQLite:
```
sqlite:///path/to/database.db
sqlite:////absolute/path/to/database.db
```

PostgreSQL:
```
postgresql://user:password@localhost/database
postgresql://user:password@host:5432/database
```

### Connection Handling

- Automatic connection pooling via SQLAlchemy
- Proper resource cleanup via context managers
- Raw connection available via `connection` attribute

## Limitations & Notes

1. **ChromaDB** - Still uses separate vector database (unchanged)
2. **Migrations** - Manual migration required for schema changes
3. **Testing** - PostgreSQL tests require server setup
4. **Performance** - Initial testing shows similar performance to raw sqlite3

## Future Enhancements

Potential improvements for future releases:
- MySQL/MariaDB support
- Database migration tools (Alembic)
- Connection pool configuration
- Read replica support
- Async database operations

## Support

For issues or questions:
1. Check documentation in `docs/configuration.md`
2. Review test examples in `tests/test_multi_database.py`
3. Open issue on GitHub with `database` label
