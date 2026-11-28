"""
Generated keyword support.

This package provides TF-IDF keyword generation for papers.  Generated keywords
are stored in the ``papers.generated_keywords`` column and used for runtime
topic aggregation instead of the source keywords supplied by downloaders.
"""

from abstracts_explorer.keywords.generation import (
    GENERATOR_VERSION,
    generate_keywords_for_scope,
    top_terms_for_documents,
)

__all__ = [
    "GENERATOR_VERSION",
    "generate_keywords_for_scope",
    "top_terms_for_documents",
]
