# Database Models Module

The database models module defines SQLAlchemy ORM models for all database tables.
These models support both SQLite and PostgreSQL backends.

## Models

- **Paper** — research paper with metadata (title, authors, abstract, year, conference, etc.)
- **EmbeddingsMetadata** — tracks the embedding model used to generate vector embeddings
- **ClusteringCache** — caches clustering results with visualization coordinates
- **HierarchicalLabelCache** — caches hierarchical cluster labels
- **ValidationData** — stores donated validation data for evaluation
- **ChatDonation** — stores donated chat interaction data
- **EvalQAPair** — evaluation question-answer pairs
- **EvalResult** — evaluation run results

## Quick Start

```python
from abstracts_explorer.db_models import Paper, EmbeddingsMetadata, Base

# The models are typically used through DatabaseManager,
# but can be used directly with SQLAlchemy sessions:
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("sqlite:///papers.db")
Base.metadata.create_all(engine)

with Session(engine) as session:
    paper = Paper(
        title="Example Paper",
        abstract="An example abstract.",
        year=2025,
        conference="NeurIPS",
    )
    session.add(paper)
    session.commit()
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.db_models
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: metadata, registry
```
