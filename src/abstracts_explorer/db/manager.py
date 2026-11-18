"""
DatabaseManager facade composing the db package mixins.
"""

from abstracts_explorer.db.catalog import CatalogMixin
from abstracts_explorer.db.clustering_cache import ClusteringCacheMixin
from abstracts_explorer.db.donations import DonationsMixin
from abstracts_explorer.db.evaluation import EvaluationRepoMixin
from abstracts_explorer.db.search import SearchMixin
from abstracts_explorer.db.transfer import TransferMixin


class DatabaseManager(
    SearchMixin,  # brings PapersMixin -> WhereFilterMixin -> SessionMixin
    CatalogMixin,  # brings PapersMixin
    ClusteringCacheMixin,
    DonationsMixin,
    EvaluationRepoMixin,
    TransferMixin,
):
    """
    Manager for SQL database operations using SQLAlchemy.

    Supports SQLite and PostgreSQL backends through SQLAlchemy connection URLs.
    Database configuration is read from the config file (PAPER_DB variable).

    Attributes
    ----------
    database_url : str
        SQLAlchemy database URL from configuration.
    engine : Engine or None
        SQLAlchemy engine instance.
    SessionLocal : sessionmaker or None
        SQLAlchemy session factory.
    _session : Session or None
        Internal active database session.
    session : Session
        Lazy session property.  The first access connects to the database and
        creates any missing tables.

    Examples
    --------
    >>> # Database configuration comes from config file
    >>> db = DatabaseManager()
    >>> db.get_paper_count()  # connects and creates tables automatically
    0
    >>> db.close()
    """
