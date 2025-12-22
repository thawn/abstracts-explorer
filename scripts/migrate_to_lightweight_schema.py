#!/usr/bin/env python3
"""
Migration script to convert NeurIPS database from complex schema to lightweight ML4PS schema.

This script:
1. Creates backup of original database
2. Creates new tables with lightweight schema
3. Migrates data:
   - Converts authors from separate table to comma-separated list
   - Renames 'name' column to 'title'
   - Generates new UID based on title+original_id hash
   - Stores original UID in 'original_id' column
4. Removes old tables (authors, paper_authors)
"""

import argparse
import hashlib
import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def backup_database(db_path: Path) -> Path:
    """Create a backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.db")

    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    logger.info(f"Backup created successfully")

    return backup_path


def migrate_database(db_path: Path, dry_run: bool = False):
    """Migrate database from NeurIPS schema to lightweight ML4PS schema."""

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    # Create backup
    if not dry_run:
        backup_path = backup_database(db_path)
        logger.info(f"Original database backed up to: {backup_path}")
    else:
        logger.info("DRY RUN MODE - No changes will be made")

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Check if migration is needed
        cursor.execute("PRAGMA table_info(papers)")
        columns = {row[1] for row in cursor.fetchall()}

        if "title" in columns and "original_id" in columns:
            logger.info("Database already appears to be using lightweight schema")
            return True

        if "name" not in columns:
            logger.error("Database schema not recognized")
            return False

        logger.info("Starting migration...")

        # Check if authors table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='authors'")
        has_authors_table = cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_authors'")
        has_paper_authors_table = cursor.fetchone() is not None

        # Step 1: Create temporary table with new lightweight schema
        logger.info("Creating temporary table with new lightweight schema...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS papers_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT UNIQUE,
                original_id TEXT,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                session TEXT,
                poster_position TEXT,
                paper_pdf_url TEXT,
                poster_image_url TEXT,
                url TEXT,
                room_name TEXT,
                keywords TEXT,
                starttime TEXT,
                endtime TEXT,
                award TEXT,
                year INTEGER,
                conference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Step 2: Get all papers
        logger.info("Fetching papers from old schema...")
        cursor.execute("SELECT * FROM papers")
        papers = cursor.fetchall()
        logger.info(f"Found {len(papers)} papers to migrate")

        # Step 3: Migrate each paper
        migrated = 0
        for paper in papers:
            paper_id = paper["id"]
            title = paper["name"]  # Rename from 'name' to 'title'
            original_id = paper.get("uid")  # Store original UID

            # Generate new UID as hash from title + original_id
            uid_source = f"{title}:{original_id if original_id else ''}"
            new_uid = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]

            # Get authors if separate table exists
            authors_str = paper.get("authors", "")
            if has_authors_table and has_paper_authors_table:
                try:
                    cursor.execute(
                        """
                        SELECT a.fullname
                        FROM authors a
                        JOIN paper_authors pa ON a.id = pa.author_id
                        WHERE pa.paper_id = ?
                        ORDER BY pa.author_order
                    """,
                        (paper_id,),
                    )
                    author_rows = cursor.fetchall()
                    if author_rows:
                        authors_str = ", ".join([a["fullname"] for a in author_rows])
                except sqlite3.Error:
                    # If query fails, use existing authors field
                    pass

            # Insert into new table with lightweight schema
            cursor.execute(
                """
                INSERT INTO papers_new (
                    id, uid, original_id, title, authors, abstract, session, poster_position,
                    paper_pdf_url, poster_image_url, url, room_name, keywords, starttime, endtime,
                    award, year, conference
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    paper_id,
                    new_uid,
                    original_id,
                    title,
                    authors_str,
                    paper.get("abstract"),
                    paper.get("session"),
                    paper.get("poster_position"),
                    paper.get("paper_pdf_url"),
                    paper.get("poster_image_url"),
                    paper.get("url"),
                    paper.get("room_name"),
                    paper.get("keywords"),
                    paper.get("starttime"),
                    paper.get("endtime"),
                    paper.get("award"),
                    paper.get("year"),
                    paper.get("conference"),
                ),
            )

            migrated += 1
            if migrated % 100 == 0:
                logger.info(f"Migrated {migrated}/{len(papers)} papers...")

        logger.info(f"Successfully migrated {migrated} papers")

        if not dry_run:
            # Step 4: Drop old tables
            logger.info("Dropping old tables...")
            cursor.execute("DROP TABLE IF EXISTS papers")
            if has_authors_table:
                cursor.execute("DROP TABLE IF EXISTS authors")
            if has_paper_authors_table:
                cursor.execute("DROP TABLE IF EXISTS paper_authors")

            # Step 5: Rename new table
            logger.info("Renaming new table...")
            cursor.execute("ALTER TABLE papers_new RENAME TO papers")

            # Step 6: Create indices for lightweight schema
            logger.info("Creating indices...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_uid ON papers(uid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_original_id ON papers(original_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON papers(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON papers(session)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conference ON papers(conference)")

            # Commit changes
            conn.commit()
            logger.info("Migration completed successfully!")
        else:
            # Rollback in dry run mode
            conn.rollback()
            cursor.execute("DROP TABLE IF EXISTS papers_new")
            logger.info("DRY RUN completed - no changes made")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        conn.rollback()
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NeurIPS database from complex schema to lightweight ML4PS schema"
    )
    parser.add_argument("database", type=Path, help="Path to database file to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")

    args = parser.parse_args()

    if not args.database.exists():
        logger.error(f"Database not found: {args.database}")
        sys.exit(1)

    success = migrate_database(args.database, dry_run=args.dry_run)

    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
