# Clustering Module

The clustering module clusters paper embeddings with BERTopic and produces a
compact JSON payload for the web UI, MCP tools, and exports. The server never
imports Plotly; all rendering happens in the browser.

## Features

- BERTopic with Agglomerative clustering (`distance_threshold=0.5`, ward,
  euclidean) on 5-D UMAP embeddings
- 2-D UMAP coordinates for visualization
- 10 linear hierarchy levels with per-level flat-topic mappings
- LLM-generated labels for flat and merged topics, with retry and keyword
  fallback
- Base c-TF-IDF keyword extraction for every flat topic
- Compact JSON cache (points, statistics, flat labels/keywords, hierarchy levels)
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

results = cm.fit(use_llm_labels=True, nr_levels=10)
print(results["statistics"])
print(len(results["hierarchy"]["levels"]))
```

Use `compute_clusters_with_cache()` for the CLI/web/MCP workflow. Old
Agglomerative + t-SNE caches are invalid; regenerate them with
`abstracts-explorer clustering pre-generate`.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.clustering
   :members:
   :undoc-members:
   :show-inheritance:
```
