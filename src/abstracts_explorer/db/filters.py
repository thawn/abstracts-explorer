"""
Where-filter DSL and field-filter parsing.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import String, cast, or_, and_

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Paper,
)

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


def _filter_has_key(filters: Optional[Dict[str, Any]], key: str) -> bool:
    """
    Return whether *key* is constrained at the top level or under ``$and``.

    Keys nested inside ``$or``/``$not`` do not count as an equivalent
    top-level restriction and must not suppress caller-supplied filters.
    """
    if not isinstance(filters, dict):
        return False
    key_lower = key.lower()
    for raw_key, value in filters.items():
        if str(raw_key).lower() == key_lower:
            return True
        if str(raw_key).lower() == "$and" and isinstance(value, list):
            if any(_filter_has_key(item, key) for item in value):
                return True
    return False


def _escape_like(value: Any) -> str:
    """Escape SQL ``LIKE`` wildcards so the operand is matched literally."""
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class WhereFilterMixin(SessionMixin):
    """Where-filter DSL and field-filter parsing."""

    def _build_where_conditions(self, where_filters: Optional[Dict[str, Any]]) -> List[Any]:
        """
        Translate a where-filter dictionary into SQLAlchemy conditions.

        Filter keys are Paper column names (``author`` is accepted as an alias
        for ``authors``).  A scalar value adds an equality condition.  A dict
        value adds one or more operator conditions:

        - ``$eq``, ``$ne``
        - ``$in``, ``$nin``
        - ``$contains`` (case-insensitive substring)
        - ``$startswith``, ``$endswith``
        - ``$gt``, ``$gte``, ``$lt``, ``$lte``

        The logical operators ``$and``, ``$or`` and ``$not`` accept nested
        filter dictionaries (``$not`` accepts one dict; the others accept a
        list).
        """
        if not where_filters:
            return []

        columns = {c.name: c for c in Paper.__table__.columns if c.name != "created_at"}
        alias_lower = {k.lower(): v for k, v in WhereFilterMixin.FIELD_ALIASES.items()}
        conditions: List[Any] = []

        for raw_key, value in where_filters.items():
            key = raw_key.lower()

            if key in ("$and", "$or"):
                if not isinstance(value, list):
                    raise DatabaseError(f"'{raw_key}' expects a list of filter dictionaries")
                nested: List[List[Any]] = [
                    self._build_where_conditions(item) for item in value if isinstance(item, dict)
                ]
                groups = [group for group in nested if group]
                if groups:
                    if key == "$and":
                        conditions.append(and_(*[cond for group in groups for cond in group]))
                    else:
                        conditions.append(or_(*[and_(*group) for group in groups]))
                continue

            if key == "$not":
                if not isinstance(value, dict):
                    raise DatabaseError("'$not' expects a filter dictionary")
                not_conditions: List[Any] = self._build_where_conditions(value)
                if not_conditions:
                    conditions.append(~and_(*not_conditions))
                continue

            field_name = alias_lower.get(key, key)
            column = columns.get(field_name)
            if column is None:
                raise DatabaseError(f"Unknown paper field: {raw_key}")

            if isinstance(value, dict):
                for op, operand in value.items():
                    op_lower = op.lower()
                    if field_name == "year" and op_lower not in (
                        "$contains",
                        "$startswith",
                        "$endswith",
                    ):
                        operand = self._normalize_year_operand(operand)
                    if op_lower == "$eq":
                        conditions.append(column == operand)
                    elif op_lower == "$ne":
                        if operand is None:
                            conditions.append(column.isnot(None))
                        else:
                            conditions.append(or_(column != operand, column.is_(None)))
                    elif op_lower == "$in":
                        conditions.append(column.in_(operand))
                    elif op_lower == "$nin":
                        conditions.append(column.notin_(operand))
                    elif op_lower == "$contains":
                        conditions.append(cast(column, String).ilike(f"%{_escape_like(operand)}%", escape="\\"))
                    elif op_lower == "$startswith":
                        conditions.append(cast(column, String).ilike(f"{_escape_like(operand)}%", escape="\\"))
                    elif op_lower == "$endswith":
                        conditions.append(cast(column, String).ilike(f"%{_escape_like(operand)}", escape="\\"))
                    elif op_lower == "$gt":
                        conditions.append(column > operand)
                    elif op_lower == "$gte":
                        conditions.append(column >= operand)
                    elif op_lower == "$lt":
                        conditions.append(column < operand)
                    elif op_lower == "$lte":
                        conditions.append(column <= operand)
                    else:
                        raise DatabaseError(f"Unknown where-filter operator: {op}")
            else:
                if field_name == "year":
                    value = self._normalize_year_operand(value)
                conditions.append(column == value)

        return conditions

    @classmethod
    def _normalize_year_operand(cls, value: Any) -> Any:
        """Convert year filter operands to integers where possible."""
        if isinstance(value, list):
            return [cls._normalize_year_operand(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._normalize_year_operand(item) for key, item in value.items()}
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def _normalize_where_filters(cls, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize year values in a nested filter dictionary."""
        if not isinstance(filters, dict):
            return filters

        normalized: Dict[str, Any] = {}
        for key, value in filters.items():
            lower = key.lower()
            if lower in ("$and", "$or") and isinstance(value, list):
                normalized[key] = [cls._normalize_where_filters(item) for item in value]
            elif lower == "$not":
                normalized[key] = cls._normalize_where_filters(value)
            elif lower == "year":
                normalized[key] = cls._normalize_year_operand(value)
            else:
                normalized[key] = value
        return normalized

    #: Paper model column names that can be used as ``field:"value"`` filters
    #: in search queries.  Internal columns (``uid``, ``created_at``) are
    #: excluded because they are not meaningful search targets for users.
    SEARCHABLE_FIELDS: set = {c.name for c in Paper.__table__.columns if c.name not in ("uid", "created_at")}

    #: Aliases for field names in search queries.  An alias is transparently
    #: resolved to the canonical column name before applying the filter.
    FIELD_ALIASES: Dict[str, str] = {"author": "authors"}

    @staticmethod
    def parse_field_filters(query: str) -> tuple:
        """
        Parse field-specific filters from a search query string.

        Extracts all ``field:"value"`` patterns from the query where *field*
        is a column name of the :class:`~abstracts_explorer.db_models.Paper`
        model (or a recognised alias; see :attr:`FIELD_ALIASES`).
        Unrecognised field names are left in the query text as-is.

        Parameters
        ----------
        query : str
            The raw search query, e.g.
            ``'authors:"John Smith" award:"Best Paper" transformers'``.

        Returns
        -------
        tuple of (dict, str)
            A tuple ``(field_filters, remaining_query)`` where
            *field_filters* maps canonical column names to their search values
            and *remaining_query* is the query with recognised filters removed.

        Examples
        --------
        >>> DatabaseManager.parse_field_filters('authors:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('author:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('Author:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('transformers')
        ({}, 'transformers')
        >>> DatabaseManager.parse_field_filters('award:"Best Paper" authors:"Doe"')
        ({'award': 'Best Paper', 'authors': 'Doe'}, '')
        """
        field_filters: Dict[str, str] = {}
        remaining = query

        # Build a lower-cased lookup for aliases and searchable fields so
        # that user input is matched case-insensitively (e.g. Author, AUTHOR).
        alias_lower = {k.lower(): v for k, v in WhereFilterMixin.FIELD_ALIASES.items()}
        fields_lower = {f.lower(): f for f in WhereFilterMixin.SEARCHABLE_FIELDS}

        # Iterate in reverse so that removing matched spans does not
        # invalidate the start/end offsets of earlier matches.
        for match in reversed(list(re.finditer(r'(\w+):"([^"]+)"', query))):
            raw_field = match.group(1).lower()
            # Resolve alias (e.g. "author" → "authors") if applicable
            field_name = alias_lower.get(raw_field, raw_field)
            value = match.group(2).strip()
            if field_name in fields_lower:
                canonical = fields_lower[field_name]
                if canonical not in field_filters:
                    field_filters[canonical] = value
                remaining = remaining[: match.start()] + remaining[match.end() :]

        remaining = " ".join(remaining.split())  # normalise whitespace
        return field_filters, remaining
