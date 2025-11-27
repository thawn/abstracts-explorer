#!/usr/bin/env python3
"""
Demo script to show search results with author names and new metadata fields.
"""

from pathlib import Path
from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.config import get_config


def demo_author_lookup():
    """Demonstrate author name lookup from database."""

    config = get_config()
    db_path = Path(config.paper_db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"Please run: neurips-abstracts download --year 2025 --output {db_path}")
        return

    print("=" * 70)
    print("Demo: Author Name Lookup from Database")
    print("=" * 70)

    db = DatabaseManager(db_path)
    db.connect()

    # Get a sample paper with author IDs
    papers = db.query(
        """
        SELECT id, name, authors, paper_url, poster_position 
        FROM papers 
        WHERE authors IS NOT NULL 
        LIMIT 3
    """
    )

    for paper in papers:
        print(f"\nPaper ID: {paper['id']}")
        print(f"Title: {paper['name']}")
        print(f"Author IDs (from papers table): {paper['authors']}")

        # Get author names using get_paper_authors()
        authors = db.get_paper_authors(paper["id"])
        if authors:
            author_names = [author["fullname"] for author in authors]
            print(f"Author Names (from join query): {', '.join(author_names)}")
            print(f"Number of authors: {len(authors)}")
        else:
            print("Author Names: No authors found")

        # Show new metadata fields
        if paper["paper_url"]:
            print(f"URL: {paper['paper_url']}")
        if paper["poster_position"]:
            print(f"Poster Position: {paper['poster_position']}")

    db.close()
    print("\n" + "=" * 70)
    print("✅ Demo complete! The search command now supports:")
    print("   - Displaying author names instead of IDs (with --db-path)")
    print("   - Showing paper_url in results")
    print("   - Showing poster_position in results")
    print("=" * 70)


if __name__ == "__main__":
    demo_author_lookup()
