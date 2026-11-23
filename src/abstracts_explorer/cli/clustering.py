"""
clustering commands
"""

import argparse
import logging
import sys
from contextlib import closing
from typing import Optional


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.clustering import (
    perform_clustering,
    compute_clusters_with_cache,
    ClusteringError,
)


from abstracts_explorer.cli._common import (
    _print_header,
    _resolve_conference_arg,
    _unexpected_error,
    _validate_embeddings_db,
)

logger = logging.getLogger(__name__)


def cluster_embeddings_command(args: argparse.Namespace) -> int:
    """
    Cluster embeddings and optionally export results.

    Uses BERTopic with Agglomerative clustering (ward linkage, distance
    threshold 0.5) on 5-D UMAP embeddings and 2-D UMAP visualization.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - output: Path to export JSON results
        - limit: Maximum number of embeddings to process

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not _validate_embeddings_db(config):
        return 1

    requests_per_minute = getattr(args, "requests_per_minute", config.requests_per_minute)
    rate_limit_str = f"{requests_per_minute} req/min" if requests_per_minute > 0 else "disabled"
    _print_header(
        "Clustering",
        f"Embeddings: {config.embedding_db}",
        f"Collection: {args.collection}",
        "Reduction: umap (2 components for visualization)",
        "Clustering: BERTopic + agglomerative (ward, distance_threshold=0.5)",
        f"Rate limit: {rate_limit_str}",
        f"Limit: {args.limit} papers" if args.limit else None,
    )

    try:
        # Perform clustering with fixed parameters
        print("\n🚀 Starting clustering pipeline...")
        results = perform_clustering(
            collection_name=args.collection,
            output_path=args.output,
            limit=args.limit,
            show_progress=True,
            use_llm_labels=not getattr(args, "no_llm_labels", False),
            requests_per_minute=requests_per_minute,
        )

        # Display statistics
        stats = results["statistics"]
        print("\n📊 Clustering Results:")
        print(f"   Total papers:  {stats['total_papers']:,}")
        print(f"   Clusters:      {stats['n_clusters']}")
        if stats["n_noise"] > 0:
            print(f"   Noise points:  {stats['n_noise']}")
        print("\n   Cluster sizes:")
        for cluster_id, size in sorted(stats["cluster_sizes"].items()):
            print(f"      Cluster {cluster_id}: {size:,} papers")

        if args.output:
            print(f"\n💾 Results exported to: {args.output}")
            print("\nYou can use the web UI to visualize the clusters!")
        else:
            print("\n💡 Tip: Use --output to export results for visualization")

        return 0

    except ClusteringError as e:
        print(f"\n❌ Clustering error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)


def clear_clustering_cache_command(args: argparse.Namespace) -> int:
    """
    Clear clustering cache from the database.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - embedding_model: Optional embedding model to filter by

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    _print_header(
        "Clear Clustering Cache",
        f"Database: {config.masked_database_url}",
        f"Filtering by model: {args.embedding_model}" if args.embedding_model else "Clearing all cache entries",
    )

    try:
        # Connect to database
        with DatabaseManager() as db:
            # Clear the cache
            count = db.clear_clustering_cache(embedding_model=args.embedding_model)

            if count == 0:
                print("\n✅ No cache entries found to clear.")
            else:
                entry_word = "entry" if count == 1 else "entries"
                if args.embedding_model:
                    print(f"\n✅ Cleared {count} clustering cache {entry_word} for model: {args.embedding_model}")
                else:
                    print(f"\n✅ Cleared all {count} clustering cache {entry_word}")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error clearing clustering cache")


def pre_generate_clustering_command(args: argparse.Namespace) -> int:
    """
    Pre-generate clustering results for one or all conference/year combinations.

    Uses BERTopic with Agglomerative clustering (ward linkage, distance
    threshold 0.5) on 5-D UMAP embeddings and 2-D UMAP visualization.
    Results are persisted to the database cache so that the web UI can serve
    them instantly.

    Without ``--conference`` or ``--year``, generates clustering for every
    conference in the database combined with each individual year.

    With ``--conference`` only, generates for that conference for each individual year.

    With both ``--conference`` and ``--year``, generates for that specific
    conference + year only.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - conference: Single conference to filter by (optional)
        - year: Single year to filter by (optional)
        - force: Force recompute even if cache exists

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not _validate_embeddings_db(config):
        return 1

    # Resolve conference name once to canonical form
    resolved_conference = _resolve_conference_arg(getattr(args, "conference", None))
    year_arg: Optional[int] = getattr(args, "year", None)

    # Build the list of (conference, year) combinations from the database
    combos: list = []
    if resolved_conference is not None and year_arg is not None:
        # Both specified: single combo, no DB lookup needed
        combos = [(resolved_conference, year_arg)]
    else:
        try:
            with DatabaseManager() as db:
                conferences = [resolved_conference] if resolved_conference else db.get_conferences()
                for conf in conferences:
                    years_for_conf = db.get_years(conference=conf)
                    if year_arg is not None:
                        years_for_conf = [year for year in years_for_conf if year == year_arg]
                    for year in years_for_conf:
                        combos.append((conf, year))
        except Exception:
            pass

    if not combos:
        print("❌ No conferences found in the database.", file=sys.stderr)
        return 1

    rate_limit_str = f"{args.requests_per_minute} req/min" if args.requests_per_minute > 0 else "disabled"
    if len(combos) > 1:
        extra_lines = [f"Combinations: {len(combos)} total"]
    else:
        conf_name, yr = combos[0]
        extra_lines = []
        if conf_name:
            extra_lines.append(f"Conference: {conf_name}")
        if yr is not None:
            extra_lines.append(f"Year: {yr}")
    _print_header(
        "Pre-generate Clustering",
        f"Embeddings: {config.embedding_db}",
        f"Collection: {args.collection}",
        "Clustering: BERTopic + agglomerative (ward, distance_threshold=0.5)",
        "Reduction: umap",
        f"Rate limit: {rate_limit_str}",
        *extra_lines,
    )

    try:
        with closing(
            EmbeddingsManager(
                collection_name=args.collection,
                requests_per_minute=args.requests_per_minute,
                openai_max_retries=0,
            )
        ) as em:
            success_count = 0
            fail_count = 0

            for conf_name, yr in combos:
                c_conferences = [conf_name] if conf_name else None
                c_years = [yr] if yr is not None else None

                label = f"{conf_name or 'all'}"
                if yr is not None:
                    label += f" {yr}"
                else:
                    raise ValueError(
                        "Year must be specified for clustering pre-generation to ensure manageable computation"
                    )

                print(f"\n🚀 Clustering {label}...")

                try:
                    with DatabaseManager() as db:
                        results = compute_clusters_with_cache(
                            embeddings_manager=em,
                            database=db,
                            embedding_model=config.embedding_model,
                            limit=None,
                            force=args.force,
                            conferences=c_conferences,
                            years=c_years,
                            use_llm_labels=not getattr(args, "no_llm_labels", False),
                            show_progress=True,
                            backfill_missing_labels=True,
                        )

                    stats = results.get("statistics", {})
                    n_papers = stats.get("total_papers", "N/A")
                    n_clusters = stats.get("n_clusters", "N/A")
                    n_papers_str = f"{n_papers:,}" if isinstance(n_papers, int) else str(n_papers)
                    print(f"   ✅ {label}: {n_papers_str} papers → {n_clusters} clusters")
                    success_count += 1
                except Exception as e:
                    print(f"   ❌ {label}: {e}")
                    fail_count += 1

            print(f"\n{'=' * 70}")
            print(f"✅ Pre-generation complete: {success_count} succeeded, {fail_count} failed")

        return 0 if fail_count == 0 else 1

    except ClusteringError as e:
        print(f"\n❌ Clustering error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)
