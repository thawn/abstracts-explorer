"""
Interesting-paper summaries and recommendations.

The public API is re-exported here so ``from abstracts_explorer.interesting_papers
import ...`` keeps working.
"""

from abstracts_explorer.interesting_papers.loading import load_interesting_papers, normalize_priorities
from abstracts_explorer.interesting_papers.recommend import recommend_related_papers
from abstracts_explorer.interesting_papers.summary import summarize_interesting_papers

__all__ = [
    "normalize_priorities",
    "load_interesting_papers",
    "summarize_interesting_papers",
    "recommend_related_papers",
]
