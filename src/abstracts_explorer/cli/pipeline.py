"""
``pipeline run`` command

"""

import argparse
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple


from abstracts_explorer.config import get_config


from abstracts_explorer.cli._common import (
    _print_header,
)
from abstracts_explorer.cli.clustering import pre_generate_clustering_command
from abstracts_explorer.cli.download import download_command
from abstracts_explorer.cli.embeddings import create_embeddings_command

logger = logging.getLogger(__name__)


def _build_pre_process_embeddings_args(
    args: argparse.Namespace,
    config: Any,
    conference: Optional[str],
    year: Optional[int],
    force: bool,
    non_interactive: bool = False,
) -> argparse.Namespace:
    """Build the ``embeddings create`` argument namespace for a pipeline unit."""
    return argparse.Namespace(
        conference=conference,
        year=year,
        collection=getattr(args, "collection", config.collection_name),
        embedding_backend_url=getattr(args, "embedding_backend_url", config.llm_backend_url),
        embedding_model=getattr(args, "embedding_model", config.embedding_model),
        force=force,
        where=getattr(args, "where", None),
        requests_per_minute=getattr(args, "requests_per_minute", config.requests_per_minute),
        ignore_embedding_model_mismatch=getattr(args, "ignore_embedding_model_mismatch", False),
        non_interactive=non_interactive,
        verbose=getattr(args, "verbose", 0),
    )


def _build_pre_process_clustering_args(
    args: argparse.Namespace,
    config: Any,
    conference: Optional[str],
    year: Optional[int],
    force: bool,
) -> argparse.Namespace:
    """Build the ``clustering cache populate`` argument namespace for a pipeline unit."""
    return argparse.Namespace(
        conference=conference,
        year=year,
        collection=getattr(args, "collection", config.collection_name),
        embedding_model=getattr(args, "embedding_model", config.embedding_model),
        force=force,
        no_llm_labels=getattr(args, "no_llm_labels", False),
        requests_per_minute=getattr(args, "requests_per_minute", config.requests_per_minute),
        verbose=getattr(args, "verbose", 0),
    )


def _post_process_worker(
    unit_queue: "queue.Queue[Optional[Tuple[str, int, int]]]",
    args: argparse.Namespace,
    config: Any,
    force: bool,
    successes: List[str],
    failures: List[Tuple[str, str]],
) -> None:
    """Embed and cluster downloaded units sequentially as they become available."""
    while True:
        unit = unit_queue.get()
        if unit is None:
            unit_queue.task_done()
            return

        unit_conference, unit_year, paper_count = unit
        label = f"{unit_conference} {unit_year}"

        try:
            if paper_count <= 0:
                print(f"⏭️  Skipping {label}: no papers downloaded")
                continue

            print(f"\n🧮 [{label}] Creating embeddings...")
            rc = create_embeddings_command(
                _build_pre_process_embeddings_args(
                    args,
                    config,
                    unit_conference,
                    unit_year,
                    force,
                    non_interactive=True,
                )
            )
            if rc != 0:
                failures.append((label, "embeddings create"))
                print(f"❌ [{label}] Embeddings create failed (exit code {rc})", file=sys.stderr)
                continue

            print(f"\n🔮 [{label}] Populating clustering cache...")
            rc = pre_generate_clustering_command(
                _build_pre_process_clustering_args(args, config, unit_conference, unit_year, force)
            )
            if rc != 0:
                failures.append((label, "clustering cache populate"))
                print(f"❌ [{label}] Clustering cache populate failed (exit code {rc})", file=sys.stderr)
                continue

            successes.append(label)
        except Exception as e:
            failures.append((label, str(e)))
            print(f"❌ [{label}] Unexpected error: {e}", file=sys.stderr)
        finally:
            unit_queue.task_done()


def _embedding_worker(
    unit_queue: "queue.Queue[Optional[Tuple[str, int, int]]]",
    clustering_queue: "queue.Queue[Optional[Tuple[str, int]]]",
    args: argparse.Namespace,
    config: Any,
    force: bool,
    successes: List[str],
    failures: List[Tuple[str, str]],
) -> None:
    """Embed downloaded units and hand each completed unit to the clustering worker."""
    try:
        while True:
            unit = unit_queue.get()
            if unit is None:
                return

            unit_conference, unit_year, paper_count = unit
            label = f"{unit_conference} {unit_year}"

            try:
                if paper_count <= 0:
                    print(f"⏭️  Skipping {label}: no papers downloaded")
                    continue

                print(f"\n🧮 [{label}] Creating embeddings...")
                rc = create_embeddings_command(
                    _build_pre_process_embeddings_args(
                        args,
                        config,
                        unit_conference,
                        unit_year,
                        force,
                        non_interactive=True,
                    )
                )
                if rc != 0:
                    failures.append((label, "embeddings create"))
                    print(f"❌ [{label}] Embeddings create failed (exit code {rc})", file=sys.stderr)
                    continue

                successes.append(label)
                clustering_queue.put((unit_conference, unit_year))
            except Exception as e:
                failures.append((label, str(e)))
                print(f"❌ [{label}] Unexpected error while creating embeddings: {e}", file=sys.stderr)
    finally:
        # Always terminate the downstream worker, even if this worker fails.
        clustering_queue.put(None)


def _clustering_worker(
    clustering_queue: "queue.Queue[Optional[Tuple[str, int]]]",
    args: argparse.Namespace,
    config: Any,
    force: bool,
    successes: List[str],
    failures: List[Tuple[str, str]],
) -> None:
    """Pre-generate clustering results for units whose embeddings are ready."""
    while True:
        unit = clustering_queue.get()
        if unit is None:
            return

        unit_conference, unit_year = unit
        label = f"{unit_conference} {unit_year}"

        try:
            print(f"\n🔮 [{label}] Populating clustering cache...")
            rc = pre_generate_clustering_command(
                _build_pre_process_clustering_args(args, config, unit_conference, unit_year, force)
            )
            if rc != 0:
                failures.append((label, "clustering cache populate"))
                print(f"❌ [{label}] Clustering cache populate failed (exit code {rc})", file=sys.stderr)
                continue

            successes.append(label)
        except Exception as e:
            failures.append((label, str(e)))
            print(f"❌ [{label}] Unexpected error while clustering: {e}", file=sys.stderr)


def _download_into_queue(
    download_args: argparse.Namespace,
    unit_queue: "queue.Queue[Optional[Tuple[str, int, int]]]",
) -> int:
    """Download units, enqueueing each completed unit and a final sentinel."""

    def on_unit_complete(unit_conference: str, unit_year: int, paper_count: int) -> None:
        if paper_count > 0:
            unit_queue.put((unit_conference, unit_year, paper_count))

    print("\n📥 Downloading conference data and post-processing units in parallel...")
    try:
        return download_command(download_args, on_unit_complete=on_unit_complete)
    finally:
        unit_queue.put(None)


def _pre_process_parallel(
    args: argparse.Namespace,
    config: Any,
    download_args: argparse.Namespace,
    overlap: bool = True,
) -> int:
    """Download units and post-process each unit as soon as it is committed."""
    force = getattr(args, "force", False)
    unit_queue: "queue.Queue[Optional[Tuple[str, int, int]]]" = queue.Queue()

    if overlap:
        clustering_queue: "queue.Queue[Optional[Tuple[str, int]]]" = queue.Queue()
        embedding_successes: List[str] = []
        embedding_failures: List[Tuple[str, str]] = []
        clustering_successes: List[str] = []
        clustering_failures: List[Tuple[str, str]] = []

        embedding_worker = threading.Thread(
            target=_embedding_worker,
            args=(
                unit_queue,
                clustering_queue,
                args,
                config,
                force,
                embedding_successes,
                embedding_failures,
            ),
            name="pipeline-embeddings",
            daemon=True,
        )
        clustering_worker = threading.Thread(
            target=_clustering_worker,
            args=(
                clustering_queue,
                args,
                config,
                force,
                clustering_successes,
                clustering_failures,
            ),
            name="pipeline-clustering",
            daemon=True,
        )
        embedding_worker.start()
        clustering_worker.start()

        rc_download = _download_into_queue(download_args, unit_queue)
        embedding_worker.join()
        clustering_worker.join()

        print("\n" + "=" * 70)
        print(f"📊 Embeddings: {len(embedding_successes)} succeeded, {len(embedding_failures)} failed")
        print(f"📊 Clustering: {len(clustering_successes)} succeeded, {len(clustering_failures)} failed")
        if clustering_successes:
            print("   ✅ " + ", ".join(clustering_successes))
        for label, reason in embedding_failures:
            print(f"   ❌ {label}: {reason}", file=sys.stderr)
        for label, reason in clustering_failures:
            print(f"   ❌ {label}: {reason}", file=sys.stderr)

        rc = 0
        if rc_download != 0:
            rc = rc_download
        if embedding_failures or clustering_failures:
            rc = 1
    else:
        successes: List[str] = []
        failures: List[Tuple[str, str]] = []
        worker = threading.Thread(
            target=_post_process_worker,
            args=(unit_queue, args, config, force, successes, failures),
            name="pipeline-post",
            daemon=True,
        )
        worker.start()

        rc_download = _download_into_queue(download_args, unit_queue)
        worker.join()

        print("\n" + "=" * 70)
        print(f"📊 Post-processing: {len(successes)} succeeded, {len(failures)} failed")
        if successes:
            print("   ✅ " + ", ".join(successes))
        for label, reason in failures:
            print(f"   ❌ {label}: {reason}", file=sys.stderr)

        rc = 0
        if rc_download != 0:
            rc = rc_download
        if failures:
            rc = 1

    if rc == 0:
        print("\n✅ Pipeline completed successfully!")
    return rc


def pre_process_command(args: argparse.Namespace) -> int:
    """
    Run the full data pipeline: data download → embeddings create → clustering cache populate.

    Each ``(conference, year)`` unit is post-processed as soon as its download
    is committed, while the remaining downloads continue in the background.
    By default embeddings and clustering run in separate workers so that
    clustering of one unit overlaps embeddings of the next.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - conference: Conference to process (optional; all conferences if omitted)
        - year: Year to process (optional; all available years if omitted)
        - force: Force re-download / reset embeddings / force recompute clustering
        - ignore_embedding_model_mismatch: Recreate embeddings on model mismatch
          without prompting
        - serial_post_process: Run embeddings and clustering in one worker
        - collection, embedding_backend_url, embedding_model, where,
          requests_per_minute: Forwarded
          to the embeddings and clustering steps

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    conference = getattr(args, "conference", None)
    year = getattr(args, "year", None)
    force = getattr(args, "force", False)
    verbose = getattr(args, "verbose", 0)

    _print_header(
        "Data Pipeline",
        f"Conference: {conference}" if conference else None,
        f"Year: {year}" if year is not None else None,
        f"Force: {force}",
        f"Collection: {getattr(args, 'collection', config.collection_name)}",
        f"Model: {getattr(args, 'embedding_model', config.embedding_model)}",
        "Steps: data download → embeddings create → clustering cache populate",
    )

    download_args = argparse.Namespace(
        conference=conference,
        year=year,
        output=str(Path(config.data_dir) / "abstracts.json"),
        force=force,
        list_plugins=False,
        max_workers=20,
        input_file=None,
        verbose=verbose,
    )

    overlap = not getattr(args, "serial_post_process", False)
    return _pre_process_parallel(args, config, download_args, overlap=overlap)
