"""
``feedback`` commands

"""

import argparse
import logging
import sys


from abstracts_explorer.database import DatabaseManager


from abstracts_explorer.cli._common import (
    _confirm,
    _unexpected_error,
)

logger = logging.getLogger(__name__)


def feedback_stats_command(args: argparse.Namespace) -> int:
    """
    Show summary statistics for user feedback (chat donations and data donations).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments (no extra fields required).

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            chat_stats = db.get_chat_donation_stats()
            val_stats = db.get_validation_data_stats()

            print("\n📊 Feedback Statistics")
            print("=" * 40)

            print("\n💬 Chat Donations")
            print(f"   Total:      {chat_stats['total']}")
            print(f"   👍 Up:       {chat_stats['up']}")
            print(f"   👎 Down:     {chat_stats['down']}")
            if chat_stats["total"] > 0:
                print(f"   Avg turns:  {chat_stats['avg_turns']:.1f}")

            print("\n📌 Data Donations (Interesting Papers)")
            print(f"   Total:          {val_stats['total']}")
            print(f"   Unique papers:  {val_stats['unique_papers']}")
            if val_stats["total"] > 0:
                print(f"   Avg priority:   {val_stats['avg_priority']:.1f}")
                dist = val_stats["priority_distribution"]
                if dist:
                    print("   Priority breakdown:")
                    for priority in sorted(dist.keys()):
                        print(f"      {priority}: {dist[priority]}")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")


def feedback_list_command(args: argparse.Namespace) -> int:
    """
    List individual user feedback entries.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat', 'data', or 'all'
        - rating: optional filter for chat donations ('up' or 'down')
        - limit: maximum number of entries to show
        - offset: number of entries to skip

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "all")
            rating_filter = getattr(args, "rating", None)
            limit = getattr(args, "limit", 20)
            offset = getattr(args, "offset", 0)

            if fb_type in ("chat", "all"):
                donations = db.get_chat_donations(limit=limit, rating=rating_filter, offset=offset)
                if donations:
                    print(f"\n💬 Chat Donations ({len(donations)} shown):")
                    print("-" * 60)
                    for entry in donations:
                        ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                        n_turns = len(entry["transcript"])
                        icon = "👍" if entry["rating"] == "up" else "👎"
                        print(f"  ID {entry['id']:5d}  {icon} {entry['rating']:<4}  {n_turns} turns  {ts}")
                else:
                    print("\n💬 No chat donations found.")

            if fb_type in ("data", "all"):
                val_data = db.get_validation_data(limit=limit, offset=offset)
                if val_data:
                    print(f"\n📌 Data Donations ({len(val_data)} shown):")
                    print("-" * 60)
                    for entry in val_data:
                        ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                        term = entry["search_term"] or ""
                        print(
                            f"  ID {entry['id']:5d}  UID {entry['paper_uid']}  "
                            f"priority {entry['priority']}  {ts}  {term}"
                        )
                else:
                    print("\n📌 No data donations found.")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")


def feedback_browse_command(args: argparse.Namespace) -> int:
    """
    Interactively browse individual user feedback entries.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat' or 'data'
        - rating: optional filter for chat donations ('up' or 'down')
        - sample: optional number of entries to randomly sample

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "chat")
            rating_filter = getattr(args, "rating", None)
            sample_size = getattr(args, "sample", None)

            if fb_type == "chat":
                entries = db.get_chat_donations(rating=rating_filter)
                if not entries:
                    print("No chat donations found.")
                    return 0

                if sample_size and sample_size < len(entries):
                    import random

                    entries = random.sample(entries, sample_size)

                print(f"\n💬 Browsing {len(entries)} chat donation(s)")
                print("Press [Enter] to continue, [q] to quit.\n")

                for i, entry in enumerate(entries, 1):
                    ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                    icon = "👍" if entry["rating"] == "up" else "👎"
                    print(f"--- Entry {i}/{len(entries)} (ID: {entry['id']}) {icon} {entry['rating']} — {ts} ---")
                    for msg in entry["transcript"]:
                        role = msg.get("role", "?")
                        text = msg.get("text", "")
                        print(f"  [{role}] {text}")
                    print()
                    try:
                        choice = input("[Enter] next, [q] quit: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Stopped.")
                        break
                    if choice == "q":
                        print("👋 Stopped.")
                        break

            elif fb_type == "data":
                entries = db.get_validation_data()
                if not entries:
                    print("No data donations found.")
                    return 0

                if sample_size and sample_size < len(entries):
                    import random

                    entries = random.sample(entries, sample_size)

                print(f"\n📌 Browsing {len(entries)} data donation(s)")
                print("Press [Enter] to continue, [q] to quit.\n")

                for i, entry in enumerate(entries, 1):
                    ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                    term = entry["search_term"] or "(no search term)"
                    print(
                        f"--- Entry {i}/{len(entries)} (ID: {entry['id']}) ---\n"
                        f"  Paper UID:   {entry['paper_uid']}\n"
                        f"  Priority:    {entry['priority']}\n"
                        f"  Search term: {term}\n"
                        f"  Donated at:  {ts}\n"
                    )
                    try:
                        choice = input("[Enter] next, [q] quit: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Stopped.")
                        break
                    if choice == "q":
                        print("👋 Stopped.")
                        break
            else:
                print(f"❌ Unknown type '{fb_type}'. Use 'chat' or 'data'.", file=sys.stderr)
                return 1

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")


def feedback_clear_command(args: argparse.Namespace) -> int:
    """
    Delete user feedback data (chat donations and/or data donations).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat', 'data', or 'all'
        - yes: skip the confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "all")

            chat_stats = db.get_chat_donation_stats() if fb_type in ("chat", "all") else {"total": 0}
            val_stats = db.get_validation_data_stats() if fb_type in ("data", "all") else {"total": 0}

            total = chat_stats["total"] + val_stats["total"]
            if total == 0:
                print("No feedback data found — nothing to clear.")
                return 0

            parts = []
            if chat_stats["total"] > 0:
                parts.append(f"{chat_stats['total']} chat donation(s)")
            if val_stats["total"] > 0:
                parts.append(f"{val_stats['total']} data donation(s)")

            print(f"⚠️  This will permanently delete {' and '.join(parts)}.")
            if not getattr(args, "yes", False) and not _confirm("Are you sure?"):
                print("Aborted.")
                return 0

            deleted_chat = 0
            deleted_val = 0
            if fb_type in ("chat", "all"):
                deleted_chat = db.delete_chat_donations()
            if fb_type in ("data", "all"):
                deleted_val = db.delete_validation_data()

            if deleted_chat:
                print(f"✅ Deleted {deleted_chat} chat donation(s).")
            if deleted_val:
                print(f"✅ Deleted {deleted_val} data donation(s).")

        return 0

    except Exception as e:
        return _unexpected_error(e, label="Error")
