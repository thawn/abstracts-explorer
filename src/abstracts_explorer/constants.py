"""
Shared limits for database and embedding queries.

``MAX_BIND_PARAMETERS`` is the maximum number of host parameters SQLite accepts
in a single statement since version 3.32.0 (released 2020-05-22), where the
default ``SQLITE_MAX_VARIABLE_NUMBER`` was raised from 999 to 32766. The project
requires SQLite >= 3.32 for the paper database and bundles a modern SQLite in the
ChromaDB wheel, so this value is safe for both SQLite and PostgreSQL (which
allows 65535 parameters per statement).
"""

MAX_BIND_PARAMETERS = 32766
