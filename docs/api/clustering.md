# Clustering Module

The clustering module provides functionality to cluster and visualize paper embeddings
using dimensionality reduction and clustering algorithms.

## Features

- Dimensionality reduction using PCA, t-SNE, and UMAP
- Clustering using K-Means, DBSCAN, Agglomerative, Spectral, and Fuzzy C-Means
- Automatic cluster labeling using TF-IDF and LLM-based methods
- Keyword extraction for each cluster
- Representative paper selection based on cluster centroids
- Hierarchical cluster structure for agglomerative clustering
- Export clustering results to JSON for visualization

## Quick Start

```python
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.clustering import ClusteringManager

# Initialize managers
em = EmbeddingsManager()
db = DatabaseManager()

# Create clustering manager
cm = ClusteringManager(em, db)

# Load embeddings
n_papers = cm.load_embeddings()
print(f"Loaded {n_papers} papers")

# Reduce dimensions and cluster
cm.reduce_dimensions(method="pca", n_components=2)
cm.cluster(method="kmeans", n_clusters=8)

# Generate labels and get results
cm.extract_cluster_keywords()
cm.generate_cluster_labels(use_llm=True)
results = cm.get_clustering_results()
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.clustering
   :members:
   :undoc-members:
   :show-inheritance:
```
