"""
Paper export facade.

The implementation lives in :mod:`abstracts_explorer.export`; this module is
kept as a stable import path.
"""

from abstracts_explorer.export import (
    build_search_term_filename_map,
    export_papers_to_zip,
    generate_all_papers_markdown,
    generate_folder_structure_export,
    generate_main_readme,
    generate_search_term_markdown,
    get_poster_url,
    natural_sort_key,
)

__all__ = [
    "natural_sort_key",
    "build_search_term_filename_map",
    "get_poster_url",
    "generate_all_papers_markdown",
    "generate_search_term_markdown",
    "generate_main_readme",
    "generate_folder_structure_export",
    "export_papers_to_zip",
]
