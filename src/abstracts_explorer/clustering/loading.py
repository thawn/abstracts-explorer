"""
Loading embeddings and paper metadata from ChromaDB and the database.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from abstracts_explorer.clustering.errors import ClusteringError

logger = logging.getLogger(__name__)


class LoadingMixin:
    """Load embeddings and their paper metadata."""

    embeddings_manager: Any
    database: Any
    embeddings: Any
    paper_ids: Any
    metadatas: Any

    def load_embeddings(
        self,
        limit: Optional[int] = None,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> int:
        """
        Load embeddings from ChromaDB collection.

        Parameters
        ----------
        limit : int, optional
            Maximum number of embeddings to load. If None, load all.
        conferences : list of str, optional
            Filter to only load embeddings for these conferences.
        years : list of int, optional
            Filter to only load embeddings for these years.

        Returns
        -------
        int
            Number of embeddings loaded

        Raises
        ------
        ClusteringError
            If loading fails or collection is empty
        """
        if not self.embeddings_manager.collection:
            raise ClusteringError("Collection not initialized in embeddings manager")

        try:
            has_filters = bool(conferences or years)
            ids = self.database.resolve_paper_uids(conferences=conferences, years=years) if has_filters else None
            if ids is not None and not ids:
                raise ClusteringError("No embeddings found in collection")

            get_kwargs: Dict[str, Any] = {"include": ["embeddings"]}
            if limit:
                get_kwargs["limit"] = limit
            if ids is not None:
                get_kwargs["ids"] = ids
            results = self.embeddings_manager.collection.get(**get_kwargs)

            chroma_ids = results.get("ids") or []
            chroma_embeddings = results.get("embeddings")
            if not chroma_ids or chroma_embeddings is None:
                raise ClusteringError("No embeddings found in collection")

            summaries = {paper["uid"]: paper for paper in self.database.get_paper_summaries_by_uids(list(chroma_ids))}
            keep_indices = [index for index, uid in enumerate(chroma_ids) if uid in summaries]
            if not keep_indices:
                raise ClusteringError("No matching papers found in the database")
            if len(keep_indices) != len(chroma_ids):
                logger.warning(
                    "%d embedding(s) have no matching paper row and will be ignored",
                    len(chroma_ids) - len(keep_indices),
                )

            self.paper_ids = [chroma_ids[index] for index in keep_indices]
            self.embeddings = np.array([chroma_embeddings[index] for index in keep_indices])
            self.metadatas = [summaries[uid] for uid in self.paper_ids]

            logger.info(f"Loaded {len(self.paper_ids)} embeddings with dimension {self.embeddings.shape[1]}")
            return len(self.paper_ids)

        except ClusteringError:
            raise
        except Exception as e:
            raise ClusteringError(f"Failed to load embeddings: {str(e)}") from e
