"""
t-SNE dimensionality reduction and embedding scaling.
"""

import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from abstracts_explorer.clustering.errors import ClusteringError

logger = logging.getLogger(__name__)


class DimensionalityMixin:
    """Standardize embeddings and reduce them with t-SNE."""

    embeddings: Any
    scaler: Any
    reduced_embeddings: Any

    def _scaled_embeddings(self) -> np.ndarray:
        """
        Return standardized embeddings, fitting the scaler on first use.

        Returns
        -------
        np.ndarray
            Standardized embeddings.

        Raises
        ------
        ClusteringError
            If embeddings have not been loaded.
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(self.embeddings)
        return self.scaler.transform(self.embeddings)

    def reduce_dimensions(self, n_components: int = 2, random_state: int = 42, **kwargs) -> np.ndarray:
        """
        Reduce dimensionality of embeddings using t-SNE.

        Parameters
        ----------
        n_components : int, optional
            Number of components to reduce to, by default 2
        random_state : int, optional
            Random state for reproducibility, by default 42
        **kwargs
            Additional arguments passed to t-SNE (e.g. perplexity, max_iter)

        Returns
        -------
        np.ndarray
            Reduced embeddings array of shape (n_samples, n_components)

        Raises
        ------
        ClusteringError
            If embeddings not loaded or reduction fails
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        try:
            from sklearn.manifold import TSNE  # lazy import to reduce initial load time

            scaled_embeddings = self._scaled_embeddings()

            perplexity = kwargs.pop("perplexity", min(30, len(self.embeddings) - 1))
            max_iter = kwargs.pop("max_iter", 1000)
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                max_iter=max_iter,
                **kwargs,
            )
            logger.info(f"Applying t-SNE to reduce to {n_components} dimensions (perplexity={perplexity})")

            self.reduced_embeddings = reducer.fit_transform(scaled_embeddings)
            logger.info(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")

            return self.reduced_embeddings

        except Exception as e:
            raise ClusteringError(f"Failed to reduce dimensions: {str(e)}") from e
