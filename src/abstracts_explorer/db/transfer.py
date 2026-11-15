"""
SQLite/JSON export and import (registry transfer).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import create_engine, select, delete, and_, text
from sqlalchemy.orm import Session

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Base,
    Paper,
    EmbeddingsMetadata,
    ClusteringCache,
    HierarchicalLabelCache,
)

from abstracts_explorer.db.errors import (
    DatabaseError,
    EmbeddingModelConflictError,
    normalize_model_name,
)
from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


def _row_to_dict(model: Any, obj: Any) -> Dict[str, Any]:
    """
    Serialize an ORM instance into a dict of its column values.

    Parameters
    ----------
    model : Any
        SQLAlchemy declarative model class; its ``__table__`` columns define
        the mapping.
    obj : Any
        ORM instance (or row) to serialize.

    Returns
    -------
    dict
        Mapping of column name to value.
    """
    return {c.name: getattr(obj, c.name) for c in model.__table__.columns}


class TransferMixin(SessionMixin):
    """SQLite/JSON export and import (registry transfer)."""

    # ------------------------------------------------------------------
    # Registry export / import helpers
    # ------------------------------------------------------------------

    def export_papers_to_sqlite(
        self,
        output_path: "Path",
        conference: str,
        year: int,
    ) -> int:
        """
        Export papers for a given conference and year to a standalone SQLite file.

        The exported file includes hierarchical label cache and embeddings
        metadata rows.  Clustering cache is **not** included — it is exported
        separately via :meth:`export_clustering_cache_to_json`.

        Parameters
        ----------
        output_path : Path
            Destination path for the SQLite file.
        conference : str
            Conference name to export.
        year : int
            Year to export.

        Returns
        -------
        int
            Number of papers exported.

        Raises
        ------
        DatabaseError
            If the export fails or no papers are found.
        """
        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_engine = create_engine(f"sqlite:///{output_path}")
            Base.metadata.create_all(export_engine)

            paper_count = 0
            with Session(export_engine) as export_session:
                # Export papers filtered by conference and year
                query = select(Paper).where(and_(Paper.conference == conference, Paper.year == year))
                for paper in self.session.execute(query).scalars():
                    paper_dict = _row_to_dict(Paper, paper)
                    export_session.add(Paper(**paper_dict))
                    paper_count += 1

                # Export hierarchical label cache
                for entry in self.session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = _row_to_dict(HierarchicalLabelCache, entry)
                    export_session.add(HierarchicalLabelCache(**entry_dict))

                # Export embeddings metadata
                for meta_entry in self.session.execute(select(EmbeddingsMetadata)).scalars():
                    entry_dict = _row_to_dict(EmbeddingsMetadata, meta_entry)
                    export_session.add(EmbeddingsMetadata(**entry_dict))

                export_session.commit()

            export_engine.dispose()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to export papers: {str(e)}") from e

    def import_papers_from_sqlite(
        self,
        sqlite_path: "Path",
        conference: str,
        year: int,
    ) -> int:
        """
        Import papers for a given conference and year from a SQLite file.

        Existing papers for the given conference/year are **replaced** (not
        merged).  Hierarchical label cache entries that match the conference
        and year are replaced.  Embeddings metadata is validated for
        consistency (the embedding model must match).

        Parameters
        ----------
        sqlite_path : Path
            Path to the source SQLite file.
        conference : str
            Conference name being imported.
        year : int
            Year being imported.

        Returns
        -------
        int
            Number of papers imported.

        Raises
        ------
        DatabaseError
            If the import fails or the embedding model is inconsistent.
        """
        try:
            source_engine = create_engine(
                f"sqlite:///{sqlite_path}",
                connect_args={"check_same_thread": False},
            )

            paper_count = 0
            with Session(source_engine) as source_session:
                # --- Validate EmbeddingsMetadata consistency ---
                imported_meta = source_session.execute(select(EmbeddingsMetadata)).scalars().first()
                if imported_meta:
                    existing_meta = self.session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if existing_meta and normalize_model_name(
                        str(existing_meta.embedding_model)
                    ) != normalize_model_name(str(imported_meta.embedding_model)):
                        raise EmbeddingModelConflictError(
                            str(existing_meta.embedding_model), str(imported_meta.embedding_model)
                        )

                # Delete existing papers for this conference+year
                self.session.execute(delete(Paper).where(and_(Paper.conference == conference, Paper.year == year)))

                # Delete only hierarchical label cache entries whose
                # embedding_model matches one from the imported data
                imported_models_result: List[str] = cast(
                    List[str],
                    source_session.execute(select(HierarchicalLabelCache.embedding_model).distinct()).scalars().all(),
                )
                if imported_models_result:
                    self.session.execute(
                        delete(HierarchicalLabelCache).where(
                            HierarchicalLabelCache.embedding_model.in_(imported_models_result)
                        )
                    )

                self.session.commit()

                # Import papers — use merge() to handle any UID collisions
                # (e.g. the same paper existing under a different conference
                # casing that the DELETE above didn't catch).
                for paper in source_session.execute(select(Paper)).scalars():
                    paper_dict = _row_to_dict(Paper, paper)
                    self.session.merge(Paper(**paper_dict))
                    paper_count += 1

                # Import hierarchical labels
                for entry in source_session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = _row_to_dict(HierarchicalLabelCache, entry)
                    self.session.add(HierarchicalLabelCache(**entry_dict))

                # Import embeddings metadata (only if not already present)
                if imported_meta:
                    existing_meta = self.session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if not existing_meta:
                        meta_dict = {
                            c.name: getattr(imported_meta, c.name) for c in EmbeddingsMetadata.__table__.columns
                        }
                        self.session.add(EmbeddingsMetadata(**meta_dict))

                self.session.commit()

            source_engine.dispose()
            self._invalidate_metadata_cache()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            raise self._db_error("Failed to import papers", e)

    # ------------------------------------------------------------------
    # Clustering cache JSON export / import
    # ------------------------------------------------------------------

    def export_clustering_cache_to_json(
        self,
        conference: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Export clustering cache entries matching *conference* and *year* as JSON.

        Parameters
        ----------
        conference : str
            Conference name to match.
        year : int
            Year to match.

        Returns
        -------
        dict
            A JSON-serialisable dictionary with an ``entries`` list.
            Each entry contains all :class:`ClusteringCache` columns except
            ``id`` (auto-generated on import).

        Raises
        ------
        DatabaseError
            If the export fails.
        """
        try:
            entries: List[Dict[str, Any]] = []
            stmt = select(ClusteringCache).where(
                and_(
                    ClusteringCache.conference == conference,
                    ClusteringCache.year == year,
                )
            )
            for entry in self.session.execute(stmt).scalars():
                row: Dict[str, Any] = {}
                for col in ClusteringCache.__table__.columns:
                    if col.name == "id":
                        continue  # skip PK; it will be auto-generated on import
                    val = getattr(entry, col.name)
                    if col.name in ("clustering_params", "results_json") and isinstance(val, str):
                        val = json.loads(val)
                    elif col.name == "created_at" and val is not None:
                        val = val.isoformat()
                    row[col.name] = val
                entries.append(row)

            return {"entries": entries}

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to export clustering cache: {str(e)}") from e

    def import_clustering_cache_from_json(
        self,
        data: Dict[str, Any],
        conference: str,
        year: int,
        overwrite_embedding_model: Optional[str] = None,
    ) -> int:
        """
        Import clustering cache entries from a JSON dictionary.

        Existing clustering cache entries matching *conference* and *year*
        are deleted before importing the new entries.

        Parameters
        ----------
        data : dict
            Dictionary previously returned by
            :meth:`export_clustering_cache_to_json`.
        conference : str
            Conference name for scoping the delete.
        year : int
            Year for scoping the delete.
        overwrite_embedding_model : str, optional
            When provided, the ``embedding_model`` field of every imported
            entry is replaced with this value.  Use this when importing an
            artifact whose embedding model differs from the locally
            configured one (i.e. when ``--ignore-embedding-model-mismatch``
            was passed) so that :meth:`get_clustering_cache` can find the
            entries using the local model name.

        Returns
        -------
        int
            Number of cache entries imported.

        Raises
        ------
        DatabaseError
            If the import fails.
        """
        try:
            # Delete existing matching entries using column-based filtering.
            self.session.execute(
                delete(ClusteringCache).where(
                    and_(
                        ClusteringCache.conference == conference,
                        ClusteringCache.year == year,
                    )
                )
            )
            self.session.flush()

            # For PostgreSQL the auto-increment sequence can fall out of sync when rows
            # were previously inserted with explicit primary-key values (e.g. by an older
            # version of import_papers_from_sqlite that included the clustering cache).
            # If other rows for a different conference/year remain in the table and the
            # sequence tries to reuse one of their IDs, the next INSERT will raise
            # UniqueViolation.  Reset the sequence to max(existing id)+1 so that every
            # new row gets a safe ID regardless of sequence history.
            db_url = self.database_url.lower()
            if "postgresql" in db_url or "postgres" in db_url:
                self.session.execute(
                    text(
                        "SELECT setval("
                        "  pg_get_serial_sequence('clustering_cache', 'id'),"
                        "  COALESCE((SELECT MAX(id) FROM clustering_cache), 0) + 1,"
                        "  false"
                        ")"
                    )
                )
                self.session.flush()

            count = 0
            for item in data.get("entries", []):
                row = dict(item)  # shallow copy

                # Re-serialise parsed JSON fields back to strings for DB storage
                if "clustering_params" in row and row["clustering_params"] is not None:
                    if not isinstance(row["clustering_params"], str):
                        row["clustering_params"] = json.dumps(row["clustering_params"])

                if "results_json" in row and row["results_json"] is not None:
                    if not isinstance(row["results_json"], str):
                        row["results_json"] = json.dumps(row["results_json"])

                # Convert ISO-format created_at back to datetime
                if "created_at" in row and isinstance(row["created_at"], str):
                    row["created_at"] = datetime.fromisoformat(row["created_at"])

                # Drop 'id' if present — let the DB auto-generate it
                row.pop("id", None)

                # Replace embedding_model when instructed (e.g. mismatch ignored)
                if overwrite_embedding_model is not None:
                    row["embedding_model"] = overwrite_embedding_model

                # Ensure conference/year columns are set from the import scope
                row["conference"] = conference
                row["year"] = year

                # Fill n_clusters from results statistics if not set
                if row.get("n_clusters") is None:
                    results = row.get("results_json")
                    if isinstance(results, str):
                        try:
                            results = json.loads(results)
                        except (ValueError, TypeError):
                            results = {}
                    if isinstance(results, dict):
                        actual = results.get("statistics", {}).get("n_clusters")
                        if actual is not None:
                            row["n_clusters"] = int(actual)

                self.session.add(ClusteringCache(**row))
                count += 1

            self.session.commit()
            return count

        except DatabaseError:
            raise
        except Exception as e:
            raise self._db_error("Failed to import clustering cache", e)
