# Clustering Module

The clustering module clusters paper embeddings with BERTopic and produces a
compact JSON payload for the web UI, MCP tools, and exports. The server never
imports Plotly; all rendering happens in the browser.

## Features

- BERTopic with Agglomerative clustering (`distance_threshold=0.5`, ward,
  euclidean) on 5-D UMAP embeddings
- 2-D UMAP coordinates for visualization
- Full BERTopic merge tree; hierarchy levels are derived on demand
- LLM-generated labels for flat and merged topics, with retry and keyword
  fallback
- Optional `Labeling clusters (LLM)` progress bar for flat-topic labeling;
  hierarchy labeling uses BERTopic's per-merge progress bar
- HTTP 429 handling honors `Retry-After` and pauses all LLM label calls for a
  shared cooldown, avoiding repeated retries against an exhausted quota
- Base c-TF-IDF keyword extraction for every flat topic
- Compact JSON cache (points, statistics, flat labels/keywords, full hierarchy tree)
- Export clustering results to JSON

## Quick Start

```python
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.clustering import ClusteringManager

em = EmbeddingsManager()
db = DatabaseManager()

cm = ClusteringManager(em, db)
n_papers = cm.load_embeddings()

results = cm.fit(use_llm_labels=True)
print(results["statistics"])
print(len(results["hierarchy"]["tree"]))

from abstracts_explorer.clustering import derive_hierarchy_levels
levels = derive_hierarchy_levels(results, nr_levels=10, level_scale="linear")
print(len(levels["levels"]))
```

Use `compute_clusters_with_cache()` for the CLI/web/MCP workflow. Old
Agglomerative + t-SNE caches are invalid; regenerate them with
`abstracts-explorer clustering cache populate`.

Hierarchy presentation settings (`nr_levels` and `level_scale`) are not part
of the clustering cache key. The web API and MCP tools derive levels from the
cached tree, so changing the level count or scale does not require
re-clustering.

Clustering caches must contain `hierarchy.tree` with precomputed per-node
`center` values (per-axis median) and flat `cluster_centers`. The web API and
MCP tools read these centers directly instead of recomputing them per request.
Caches without a tree or centers are considered outdated and rejected; run
`python scripts/backfill_cluster_centers_median.py` to add median centers to
existing rows, or regenerate them with
`abstracts-explorer clustering cache populate --force`.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.clustering
   :members:
   :undoc-members:
   :show-inheritance:
```
