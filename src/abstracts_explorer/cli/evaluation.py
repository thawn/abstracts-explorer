"""
``eval`` commands

"""

import argparse
import logging
import sys
from datetime import datetime


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager, EmbeddingsError
from abstracts_explorer.evaluation import (
    EvaluationError,
    Evaluator,
    format_eval_summary,
    format_eval_result_detail,
)


from abstracts_explorer.cli._common import (
    _confirm,
    _print_header,
    _unexpected_error,
)

logger = logging.getLogger(__name__)


def eval_generate_command(args: argparse.Namespace) -> int:
    """
    Generate evaluation Q/A pairs using an LLM.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - n_pairs: Number of Q/A pairs per tool
        - tools: Specific tools to generate for (or all)
        - no_followups: Disable follow-up generation
        - n_followups: Number of follow-ups per pair
        - model: Chat model name
        - backend_url: LLM backend URL

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    _print_header(
        "Generate Evaluation Q/A Pairs",
        f"Database: {config.database_url}",
        f"Model: {args.model}",
        f"Pairs/tool: {args.n_pairs}",
        f"Follow-ups: {'disabled' if args.no_followups else args.n_followups}",
        f"Tools: {', '.join(args.tools)}" if args.tools else "Tools: all",
    )

    try:
        # Initialize embeddings manager for its openai_client
        em = EmbeddingsManager(
            backend_url=args.backend_url,
        )

        with DatabaseManager() as db:
            db.create_tables()

            total_papers = db.get_paper_count()
            if total_papers == 0:
                print("\n❌ No papers in database. Download papers first.", file=sys.stderr)
                return 1
            print(f"\n📊 Found {total_papers:,} papers in database")

            evaluator = Evaluator(embeddings_manager=em, db=db, model=args.model)

            print("\n🤖 Generating Q/A pairs...")
            pairs = evaluator.generate_qa_pairs(
                n_pairs_per_tool=args.n_pairs,
                tools=args.tools if args.tools else None,
                generate_followups=not args.no_followups,
                n_followups=args.n_followups,
            )

            count = evaluator.store_qa_pairs(pairs)
            print(f"\n✅ Generated and stored {count} Q/A pair(s)")

            # Show summary by tool
            tool_counts: dict = {}
            for p in pairs:
                t = p.get("tool_name", "unknown")
                tool_counts[t] = tool_counts.get(t, 0) + 1
            print("\n📊 Pairs per tool:")
            for tool, cnt in sorted(tool_counts.items()):
                print(f"   {tool}: {cnt}")

        return 0

    except EvaluationError as e:
        print(f"\n❌ Evaluation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)


def eval_verify_command(args: argparse.Namespace) -> int:
    """
    Interactively verify, edit, or delete evaluation Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - sample: Number of pairs to sample for review
        - all: Review all unverified pairs

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            pairs = db.get_eval_qa_pairs(verified_only=False)
            if not pairs:
                print("No Q/A pairs found. Generate pairs first with 'eval generate'.")
                return 0

            # Filter to unverified pairs unless reviewing all
            if not args.all:
                unverified = [p for p in pairs if p["verified"] == 0]
            else:
                unverified = pairs

            if not unverified:
                print("✅ All pairs have been verified.")
                return 0

            # Sample if requested
            if args.sample and args.sample < len(unverified):
                import random

                review_set = random.sample(unverified, args.sample)
            else:
                review_set = unverified

            print(f"📝 Reviewing {len(review_set)} Q/A pair(s)")
            print("Commands: [a]ccept, [r]eject, [e]dit query, [E]dit answer, [s]kip, [h]elp, [q]uit\n")

            _VERIFY_HELP = (
                "  [a]  accept         — mark as verified\n"
                "  [r]  reject         — mark as rejected\n"
                "  [e]  edit query     — edit the query text, then accept\n"
                "  [E]  edit answer    — edit the expected answer, then accept\n"
                "  [s]  skip           — leave unchanged and move to the next pair\n"
                "  [h]  help           — show this help message\n"
                "  [q]  quit           — stop verification and show summary\n"
            )

            accepted = 0
            rejected = 0
            edited = 0
            quit_requested = False

            for i, pair in enumerate(review_set, 1):
                if quit_requested:
                    break
                print(f"--- Pair {i}/{len(review_set)} (ID: {pair['id']}) ---")
                print(f"Tool:     {pair.get('tool_name', 'N/A')}")
                print(f"Conv:     {pair['conversation_id']} turn {pair['turn_number']}")
                print(f"Query:    {pair['query']}")
                print(f"Answer:   {pair['expected_answer']}")
                status_map = {0: "unverified", 1: "verified", -1: "rejected"}
                print(f"Status:   {status_map.get(pair['verified'], 'unknown')}")

                action_taken = False
                while not action_taken:
                    try:
                        choice = input("\n[a/r/e/E/s/h/q]: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Verification stopped.")
                        quit_requested = True
                        action_taken = True
                        continue

                    if choice == "h":
                        print(_VERIFY_HELP)
                    elif choice in ("a", "A"):
                        db.update_eval_qa_pair(pair["id"], verified=1)
                        accepted += 1
                        print("✅ Accepted\n")
                        action_taken = True
                    elif choice in ("r", "R"):
                        db.update_eval_qa_pair(pair["id"], verified=-1)
                        rejected += 1
                        print("❌ Rejected\n")
                        action_taken = True
                    elif choice == "e":
                        try:
                            new_query = input("New query: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n\n👋 Verification stopped.")
                            quit_requested = True
                            action_taken = True
                            continue
                        if new_query:
                            db.update_eval_qa_pair(pair["id"], query=new_query, verified=1)
                            edited += 1
                            print("✏️  Updated and accepted\n")
                        else:
                            print("⏭️  Skipped (empty input)\n")
                        action_taken = True
                    elif choice in ("E", "ea"):
                        try:
                            new_answer = input("New answer: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n\n👋 Verification stopped.")
                            quit_requested = True
                            action_taken = True
                            continue
                        if new_answer:
                            db.update_eval_qa_pair(pair["id"], expected_answer=new_answer, verified=1)
                            edited += 1
                            print("✏️  Updated and accepted\n")
                        else:
                            print("⏭️  Skipped (empty input)\n")
                        action_taken = True
                    elif choice in ("q", "Q"):
                        print("👋 Quitting verification.")
                        quit_requested = True
                        action_taken = True
                    else:
                        print("⏭️  Skipped\n")
                        action_taken = True

            print(f"\n📊 Summary: {accepted} accepted, {rejected} rejected, {edited} edited")
        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")


def eval_run_command(args: argparse.Namespace) -> int:
    """
    Run evaluation on stored Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: ChromaDB collection name
        - model: Chat model name
        - backend_url: LLM backend URL
        - embedding_model: Embedding model name
        - limit: Maximum pairs to evaluate
        - include_unverified: Include unverified pairs

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    _print_header(
        "Run Evaluation",
        f"Database: {config.database_url}",
        f"Model: {args.model}",
        f"Embeddings: {config.embedding_db}",
        f"Collection: {args.collection}",
    )

    try:
        # Initialize embeddings
        em = EmbeddingsManager(
            backend_url=args.backend_url,
            model_name=args.embedding_model,
            collection_name=args.collection,
        )

        if not em.test_api_connection(verbose=True):
            return 1

        with DatabaseManager() as db:
            db.create_tables()

            pair_count = db.get_eval_qa_pair_count(verified_only=not args.include_unverified)
            if pair_count == 0:
                print("\n❌ No Q/A pairs found. Generate and verify pairs first.", file=sys.stderr)
                em.close()
                return 1

            actual_count = min(pair_count, args.limit) if args.limit else pair_count
            print(f"\n📊 Evaluating {actual_count} Q/A pair(s)...")

            evaluator = Evaluator(embeddings_manager=em, db=db, model=args.model)
            run_id = evaluator.run_evaluation(
                verified_only=not args.include_unverified,
                limit=args.limit,
            )

            # Show summary
            print(f"\n{evaluator.format_run_summary(run_id)}")

        em.close()
        return 0

    except EvaluationError as e:
        print(f"\n❌ Evaluation error: {e}", file=sys.stderr)
        return 1
    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)


def eval_results_command(args: argparse.Namespace) -> int:
    """
    Browse evaluation results.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - run_id: Specific run to show (or latest)
        - sample: Random sample size for browsing
        - detail: Show detailed per-pair results
        - clear: Delete results for the selected run
        - yes: Skip confirmation prompt when clearing

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            run_ids = db.get_eval_run_ids()
            if not run_ids:
                print("No evaluation results found. Run 'eval run' first.")
                return 0

            # Determine which run to show (runs are oldest-first, so [-1] is the most recent)
            if args.run_id:
                target_run = args.run_id
            else:
                target_run = run_ids[-1]
                print(f"Showing latest run: {target_run}")

            if target_run not in run_ids:
                print(f"❌ Run '{target_run}' not found.", file=sys.stderr)
                print(f"Available runs: {', '.join(run_ids)}")
                return 1

            # Handle --clear option
            if getattr(args, "clear", False):
                n = len(db.get_eval_results(run_id=target_run))
                if not getattr(args, "yes", False):
                    if not _confirm(
                        f"⚠️  This will permanently delete {n} result(s) for run '{target_run}'. Are you sure?"
                    ):
                        print("Aborted.")
                        return 0
                deleted = db.delete_eval_results(run_id=target_run)
                print(f"✅ Deleted {deleted} result(s) for run '{target_run}'.")
                return 0

            # Show summary
            summary = db.get_eval_run_summary(target_run)
            print(f"\n{format_eval_summary(summary, target_run)}")

            # Show details if requested
            if args.detail:
                results = db.get_eval_results(run_id=target_run)

                # Sample if requested
                if args.sample and args.sample < len(results):
                    import random

                    results = random.sample(results, args.sample)
                    print(f"\n📋 Showing random sample of {len(results)} result(s):\n")
                else:
                    print(f"\n📋 Showing all {len(results)} result(s):\n")

                # Build lookup for QA pairs
                pairs = db.get_eval_qa_pairs()
                pair_lookup = {p["id"]: p for p in pairs}

                for r in results:
                    qa = pair_lookup.get(r["qa_pair_id"])
                    print(format_eval_result_detail(r, qa))
                    print()

            # List all available runs (oldest first, so most recent is shown last)
            if len(run_ids) >= 1:
                print(f"\n📁 Available runs ({len(run_ids)}):")
                for rid in run_ids:
                    s = db.get_eval_run_summary(rid)
                    score_str = f"{s['avg_score']:.2f}" if s.get("avg_score") else "N/A"
                    run_date = s.get("run_date")
                    date_str = (
                        run_date.strftime("%Y-%m-%d %H:%M") if isinstance(run_date, datetime) else "unknown date"
                    )
                    marker = " ◀ current" if rid == target_run else ""
                    print(f"   {date_str}  {rid}: {s['total']} pairs, avg score: {score_str}{marker}")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")


def eval_clear_command(args: argparse.Namespace) -> int:
    """
    Delete all accepted (verified) evaluation Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - yes: Skip confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            count = db.get_eval_qa_pair_count(verified_only=True)
            if count == 0:
                print("No accepted Q/A pairs found — nothing to clear.")
                return 0

            print(f"⚠️  This will permanently delete {count} accepted Q/A pair(s).")
            if not args.yes and not _confirm("Are you sure?"):
                print("Aborted.")
                return 0

            deleted = db.delete_verified_eval_qa_pairs()
            print(f"✅ Deleted {deleted} accepted Q/A pair(s).")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")
