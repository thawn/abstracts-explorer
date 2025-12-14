"""
Flask web application for NeurIPS Abstracts.

Provides a web interface for searching papers, chatting with RAG,
and exploring the NeurIPS abstracts database.
"""

import os
import logging
import tempfile
import shutil
import zipfile
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, g, send_file
from flask_cors import CORS
import requests

from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.embeddings import EmbeddingsManager
from neurips_abstracts.rag import RAGChat
from neurips_abstracts.config import get_config
from neurips_abstracts.paper_utils import get_paper_with_authors, format_search_results, PaperFormattingError

logger = logging.getLogger(__name__)

# Get the directory where this file is located
PACKAGE_DIR = Path(__file__).parent

# Initialize Flask app with correct template/static folders
app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"), static_folder=str(PACKAGE_DIR / "static"))
CORS(app)

# Initialize components (lazy loading)
embeddings_manager = None
rag_chat = None


def get_database():
    """
    Get or create database connection (thread-local using Flask g).

    Returns
    -------
    DatabaseManager
        Database instance
    """
    if "db" not in g:
        config = get_config()  # Get config lazily
        db_path = config.paper_db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found: {db_path}. " "Please run 'neurips-abstracts download' first."
            )
        g.db = DatabaseManager(db_path)
        g.db.connect()  # Explicitly connect to the database
    return g.db


def get_embeddings_manager():
    """
    Get or create embeddings manager.

    Returns
    -------
    EmbeddingsManager
        Embeddings manager instance
    """
    global embeddings_manager
    if embeddings_manager is None:
        config = get_config()  # Get config lazily
        embeddings_manager = EmbeddingsManager(
            lm_studio_url=config.llm_backend_url,
            model_name=config.embedding_model,
            chroma_path=config.embedding_db_path,
            collection_name=config.collection_name,
        )
        embeddings_manager.connect()  # Connect to ChromaDB
        embeddings_manager.create_collection()  # Get or create the collection
    return embeddings_manager


def get_rag_chat():
    """
    Get or create RAG chat instance.

    Returns
    -------
    RAGChat
        RAG chat instance
    """
    global rag_chat
    database = get_database()  # Get database connection first (required)

    if rag_chat is None:
        config = get_config()  # Get config lazily
        em = get_embeddings_manager()
        rag_chat = RAGChat(
            embeddings_manager=em,
            database=database,  # Database is now required
            lm_studio_url=config.llm_backend_url,
            model=config.chat_model,
        )
    else:
        # Update database reference for this request
        rag_chat.database = database

    return rag_chat


@app.teardown_appcontext
def teardown_db(exception):
    """Close database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    """
    Render the main page.

    Returns
    -------
    str
        Rendered HTML template
    """
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    """
    Get database statistics.

    Returns
    -------
    dict
        Statistics including paper count
    """
    try:
        database = get_database()
        total_papers = database.get_paper_count()

        return jsonify(
            {
                "total_papers": total_papers,
            }
        )
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/filters")
def get_filters():
    """
    Get available filter options.

    Returns
    -------
    dict
        Dictionary with sessions, topics, and eventtypes lists
    """
    try:
        database = get_database()
        filters = database.get_filter_options()
        return jsonify(filters)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """
    Search for papers.

    Expected JSON body:
    - query: str - Search query
    - use_embeddings: bool - Use semantic search
    - limit: int - Maximum results (default: 10)
    - sessions: list[str] - Filter by sessions (optional)
    - topics: list[str] - Filter by topics (optional)
    - eventtypes: list[str] - Filter by event types (optional)

    Returns
    -------
    dict
        Search results with papers
    """
    try:
        data = request.get_json()
        query = data.get("query", "")
        use_embeddings = data.get("use_embeddings", False)
        limit = data.get("limit", 10)
        sessions = data.get("sessions", [])
        topics = data.get("topics", [])
        eventtypes = data.get("eventtypes", [])

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if use_embeddings:
            # Semantic search using embeddings
            em = get_embeddings_manager()
            database = get_database()

            # Build metadata filter for embeddings search
            filter_conditions = []
            if sessions:
                filter_conditions.append({"session": {"$in": sessions}})
            if topics:
                filter_conditions.append({"topic": {"$in": topics}})
            if eventtypes:
                filter_conditions.append({"eventtype": {"$in": eventtypes}})

            # Use $or operator if multiple conditions, otherwise use single condition
            where_filter = None
            if len(filter_conditions) > 1:
                where_filter = {"$and": filter_conditions}
            elif len(filter_conditions) == 1:
                where_filter = filter_conditions[0]

            logger.info(f"Search filter: sessions={sessions}, topics={topics}, eventtypes={eventtypes}")
            logger.info(f"Where filter: {where_filter}")

            results = em.search_similar(query, n_results=limit * 2, where=where_filter)  # Get more results to filter

            logger.info(f"Search results count: {len(results.get('ids', [[]])[0]) if results else 0}")

            # Transform ChromaDB results to paper format using shared utility
            try:
                papers = format_search_results(results, database, include_documents=False)
            except PaperFormattingError:
                # No valid papers found
                return jsonify({"papers": [], "count": 0, "query": query, "use_embeddings": use_embeddings})

            # Limit results (filtering already done at database level)
            papers = papers[:limit]
        else:
            # Keyword search in database with multiple filter support
            database = get_database()
            papers = database.search_papers(
                keyword=query, sessions=sessions, topics=topics, eventtypes=eventtypes, limit=limit
            )

            # Convert to list of dicts for JSON serialization
            papers = [dict(p) for p in papers]

            # Get authors from authors table for each paper
            for paper in papers:
                paper_id = paper.get("id")
                if paper_id:
                    authors = database.get_paper_authors(paper_id)
                    paper["authors"] = [a["fullname"] for a in authors]
                else:
                    paper["authors"] = []

        return jsonify({"papers": papers, "count": len(papers), "query": query, "use_embeddings": use_embeddings})
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/paper/<int:paper_id>")
def get_paper(paper_id):
    """
    Get a specific paper by ID.

    Parameters
    ----------
    paper_id : int
        Paper ID

    Returns
    -------
    dict
        Paper details including authors
    """
    try:
        database = get_database()
        paper = get_paper_with_authors(database, paper_id)
        return jsonify(paper)
    except PaperFormattingError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/papers/batch", methods=["POST"])
def get_papers_batch():
    """
    Get multiple papers by their IDs.

    Parameters
    ----------
    paper_ids : list of int
        List of paper IDs to fetch

    Returns
    -------
    dict
        Dictionary with 'papers' key containing list of paper details
    """
    try:
        data = request.json
        paper_ids = data.get("paper_ids", [])

        if not paper_ids:
            return jsonify({"error": "No paper IDs provided"}), 400

        database = get_database()
        papers = []

        for paper_id in paper_ids:
            try:
                paper = get_paper_with_authors(database, paper_id)
                papers.append(paper)
            except PaperFormattingError as e:
                logger.warning(f"Paper {paper_id} not found: {e}")
                continue

        return jsonify({"papers": papers})
    except Exception as e:
        logger.error(f"Error fetching papers batch: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat with RAG system.

    Expected JSON body:
    - message: str - User message
    - n_papers: int (optional) - Number of papers for context
    - reset: bool (optional) - Reset conversation
    - sessions: list (optional) - Filter by sessions
    - topics: list (optional) - Filter by topics
    - eventtypes: list (optional) - Filter by event types

    Returns
    -------
    dict
        Chat response with papers used
    """
    try:
        config = get_config()  # Get config lazily
        data = request.get_json()
        message = data.get("message", "")
        n_papers = data.get("n_papers", config.max_context_papers)
        reset = data.get("reset", False)

        # Get filters
        sessions = data.get("sessions", [])
        topics = data.get("topics", [])
        eventtypes = data.get("eventtypes", [])

        if not message:
            return jsonify({"error": "Message is required"}), 400

        rag = get_rag_chat()

        if reset:
            rag.reset_conversation()

        # Build metadata filter
        filter_conditions = []
        if sessions:
            filter_conditions.append({"session": {"$in": sessions}})
        if topics:
            filter_conditions.append({"topic": {"$in": topics}})
        if eventtypes:
            filter_conditions.append({"eventtype": {"$in": eventtypes}})

        # Use $or operator if multiple conditions, otherwise use single condition
        metadata_filter = None
        if len(filter_conditions) > 1:
            metadata_filter = {"$or": filter_conditions}
        elif len(filter_conditions) == 1:
            metadata_filter = filter_conditions[0]

        # Get response with filters
        response = rag.query(message, n_results=n_papers, metadata_filter=metadata_filter)

        return jsonify(
            {
                "response": response,
                "message": message,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    """
    Reset the chat conversation.

    Returns
    -------
    dict
        Success message
    """
    try:
        rag = get_rag_chat()
        rag.reset_conversation()
        return jsonify({"success": True, "message": "Conversation reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/years")
def get_years():
    """
    Get available years in the database.

    Returns
    -------
    dict
        List of available years
    """
    try:
        database = get_database()
        papers = database.query("SELECT DISTINCT year FROM papers WHERE year IS NOT NULL ORDER BY year")
        years = [p["year"] for p in papers]
        return jsonify({"years": years})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/interesting-papers", methods=["POST"])
def export_interesting_papers():
    """
    Export interesting papers to markdown with optional PDF and image downloads.

    Parameters
    ----------
    paper_ids : list of int
        List of paper IDs to export
    priorities : dict
        Dictionary mapping paper IDs to priority ratings (1-5)
    search_query : str, optional
        Search query context
    download_assets : bool, optional
        Whether to download PDFs and images (default: True)

    Returns
    -------
    file
        Zip file containing markdown and downloaded assets
    """
    try:
        data = request.json
        paper_ids = data.get("paper_ids", [])
        priorities = data.get("priorities", {})
        search_query = data.get("search_query", "")
        download_assets = data.get("download_assets", True)

        if not paper_ids:
            return jsonify({"error": "No paper IDs provided"}), 400

        # Fetch papers from database
        database = get_database()
        papers = []
        for paper_id in paper_ids:
            try:
                paper = get_paper_with_authors(database, paper_id)
                priority_data = priorities.get(str(paper_id), {})

                # Handle both old format (int) and new format (dict with priority and searchTerm)
                if isinstance(priority_data, dict):
                    paper["priority"] = priority_data.get("priority", 0)
                    paper["searchTerm"] = priority_data.get("searchTerm", "Unknown")
                else:
                    # Backward compatibility: old format was just an integer
                    paper["priority"] = priority_data
                    paper["searchTerm"] = search_query or "Unknown"

                papers.append(paper)
            except PaperFormattingError:
                logger.warning(f"Paper {paper_id} not found")
                continue

        if not papers:
            return jsonify({"error": "No papers found"}), 404

        # Sort papers by session, search term, priority, and poster position
        papers.sort(
            key=lambda p: (
                p.get("session") or "",
                p.get("searchTerm") or "",
                -p.get("priority", 0),  # Descending priority
                p.get("poster_position") or "",
            )
        )

        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        assets_dir = Path(temp_dir) / "assets"
        assets_dir.mkdir(exist_ok=True)

        try:
            # Generate markdown with asset links
            markdown = generate_markdown_with_assets(papers, search_query, assets_dir if download_assets else None)

            # If not downloading assets, just return the markdown file
            if not download_assets:
                # Clean up temp directory
                shutil.rmtree(temp_dir)

                # Return markdown directly
                markdown_buffer = BytesIO(markdown.encode("utf-8"))
                markdown_buffer.seek(0)

                return send_file(
                    markdown_buffer,
                    mimetype="text/markdown",
                    as_attachment=True,
                    download_name=f'interesting-papers-{papers[0].get("year", "2025")}.md',
                )

            # Write markdown file
            markdown_path = Path(temp_dir) / "interesting-papers.md"
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            # Create zip file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Add markdown
                zip_file.write(markdown_path, "interesting-papers.md")

                # Add assets if they were downloaded
                if download_assets and assets_dir.exists():
                    for asset_file in assets_dir.glob("*"):
                        zip_file.write(asset_file, f"assets/{asset_file.name}")

            zip_buffer.seek(0)

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            return send_file(
                zip_buffer,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f'interesting-papers-{papers[0].get("year", "2025")}.zip',
            )

        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    except Exception as e:
        logger.error(f"Error exporting interesting papers: {e}")
        return jsonify({"error": str(e)}), 500


def download_paper_pdf_task(paper, assets_dir):
    """
    Prepare and execute PDF download for a paper.

    Parameters
    ----------
    paper : dict
        Paper dictionary
    assets_dir : Path
        Directory to save file to

    Returns
    -------
    tuple
        (paper_id, filename or None)
    """
    paper_id = paper["id"]
    pdf_url = paper.get("paper_pdf_url")
    if not pdf_url and paper.get("paper_url"):
        # Convert forum URL to PDF URL
        pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")

    if pdf_url:
        filename = download_file(pdf_url, assets_dir, f"paper_{paper_id}.pdf")
        return (paper_id, filename)
    return (paper_id, None)


def get_poster_url(eventmedia, paper_id):
    """
    Extract poster image URL from eventmedia field or construct from paper ID.

    Parameters
    ----------
    eventmedia : str
        Event media field (may contain JSON or URLs)
    paper_id : int
        Paper ID to use as fallback for constructing poster URL

    Returns
    -------
    str or None
        Poster image URL if found, None otherwise
    """
    try:
        import json

        # Try to parse eventmedia as JSON
        if eventmedia and eventmedia.strip().startswith("["):
            media_list = json.loads(eventmedia)
            for media_item in media_list:
                if isinstance(media_item, dict):
                    # Check for "file" key (poster images)
                    file_path = media_item.get("file")

                    # Prioritize poster files (skip thumbnails, prefer full size)
                    if file_path and any(
                        ext in file_path.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        # Skip thumbnail versions
                        if "-thumb" in file_path:
                            continue
                        # Construct full URL from file path
                        return f"https://neurips.cc{file_path}"

                    # Fall back to URL field
                    url = media_item.get("url") or media_item.get("uri")
                    if url and any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                        return url
    except Exception as e:
        logger.warning(f"Failed to parse eventmedia: {e}")

    # Fallback: construct poster URL from paper ID
    if paper_id:
        return f"https://neurips.cc/media/PosterPDFs/NeurIPS%202025/{paper_id}.png"

    return None


def download_poster_image_task(paper, assets_dir):
    """
    Prepare and execute poster image download for a paper.

    Parameters
    ----------
    paper : dict
        Paper dictionary
    assets_dir : Path
        Directory to save file to

    Returns
    -------
    tuple
        (paper_id, filename or None)
    """
    paper_id = paper["id"]
    filename = download_poster_image(paper.get("eventmedia"), assets_dir, f"poster_{paper_id}", paper_id)
    return (paper_id, filename)


def download_assets_parallel(papers, assets_dir, max_workers=10):
    """
    Download poster images in parallel for all papers.

    Note: PDFs are not downloaded; links to OpenReview are provided instead.

    Parameters
    ----------
    papers : list
        List of paper dictionaries
    assets_dir : Path
        Directory to save files to
    max_workers : int
        Maximum number of parallel download threads

    Returns
    -------
    dict
        Poster results dict where keys are paper IDs and values are filenames
    """
    poster_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all poster download tasks
        poster_futures = {
            executor.submit(download_poster_image_task, paper, assets_dir): paper["id"] for paper in papers
        }

        # Collect poster results as they complete
        for future in as_completed(poster_futures):
            try:
                paper_id, filename = future.result()
                if filename:
                    poster_results[paper_id] = filename
            except Exception as e:
                paper_id = poster_futures[future]
                logger.warning(f"Failed to download poster for paper {paper_id}: {e}")

    logger.info(f"Downloaded {len(poster_results)} posters in parallel")
    return poster_results


def generate_markdown_with_assets(papers, search_query, assets_dir):
    """
    Generate markdown content with links to remote assets.

    Note: No files are downloaded. PDFs and poster images link to their original URLs.

    Parameters
    ----------
    papers : list
        List of paper dictionaries
    search_query : str
        Search query context
    assets_dir : Path or None
        Ignored (kept for backward compatibility)

    Returns
    -------
    str
        Markdown content
    """
    from datetime import datetime

    markdown = "# Interesting Papers from NeurIPS 2025\n\n"
    markdown += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"**Total Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    # Group by session, then by search term
    sessions = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = {}

        search_term = paper.get("searchTerm") or "Unknown"
        if search_term not in sessions[session]:
            sessions[session][search_term] = []
        sessions[session][search_term].append(paper)

    # Write each session and its search terms
    for session, search_terms in sessions.items():
        markdown += f"## {session}\n\n"

        for search_term, term_papers in search_terms.items():
            markdown += f"### {search_term}\n\n"

            for paper in term_papers:
                stars = "⭐" * paper.get("priority", 0)
                markdown += f"#### {paper.get('name', 'Untitled')}\n\n"
                markdown += f"**Rating:** {stars} ({paper.get('priority', 0)}/5)\n\n"

                if paper.get("authors"):
                    authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
                    markdown += f"**Authors:** {authors}\n\n"

                if paper.get("poster_position"):
                    markdown += f"**Poster:** {paper['poster_position']}\n\n"

                # Always link to PDF on OpenReview (not downloaded)
                paper_id = paper["id"]
                pdf_url = paper.get("paper_pdf_url")
                if not pdf_url and paper.get("paper_url"):
                    pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
                if pdf_url:
                    markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

                if paper.get("paper_url"):
                    markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

                if paper.get("url"):
                    markdown += f"**Source URL:** {paper['url']}\n\n"

                # Link to poster image on neurips.cc (not downloaded)
                poster_url = get_poster_url(paper.get("eventmedia"), paper_id)
                if poster_url:
                    markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

                if paper.get("abstract"):
                    markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

                markdown += "---\n\n"

    return markdown


def download_file(url, target_dir, filename):
    """
    Download a file from URL to target directory.

    Parameters
    ----------
    url : str
        URL to download from
    target_dir : Path
        Directory to save file to
    filename : str
        Name for the downloaded file

    Returns
    -------
    str or None
        Filename if successful, None otherwise
    """
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        file_path = target_dir / filename
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {filename}")
        return filename
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def download_poster_image(eventmedia, target_dir, base_filename, paper_id=None):
    """
    Download poster image from eventmedia field or construct URL from paper ID.

    Parameters
    ----------
    eventmedia : str
        Event media field (may contain JSON or URLs)
    target_dir : Path
        Directory to save image to
    base_filename : str
        Base filename (extension will be added based on image type)
    paper_id : int, optional
        Paper ID to use as fallback for constructing poster URL

    Returns
    -------
    str or None
        Filename if successful, None otherwise
    """
    try:
        import json

        # Try to parse as JSON
        if eventmedia and eventmedia.strip().startswith("["):
            media_list = json.loads(eventmedia)
            for media_item in media_list:
                if isinstance(media_item, dict):
                    # Check for "file" key (poster images) or "url" key (other media)
                    file_path = media_item.get("file")
                    url = media_item.get("url") or media_item.get("uri")

                    # Prioritize poster files (skip thumbnails, prefer full size)
                    if file_path and any(
                        ext in file_path.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        # Skip thumbnail versions
                        if "-thumb" in file_path:
                            continue
                        # Construct full URL from file path
                        full_url = f"https://neurips.cc{file_path}"
                        # Determine extension
                        ext = file_path.split(".")[-1].split("?")[0]
                        filename = f"{base_filename}.{ext}"
                        return download_file(full_url, target_dir, filename)

                    # Fall back to URL field
                    elif url and any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                        # Determine extension
                        ext = url.split(".")[-1].split("?")[0]
                        filename = f"{base_filename}.{ext}"
                        return download_file(url, target_dir, filename)
    except Exception as e:
        logger.warning(f"Failed to parse eventmedia: {e}")

    # Fallback: try to construct poster URL from paper ID
    if paper_id:
        try:
            poster_url = f"https://neurips.cc/media/PosterPDFs/NeurIPS%202025/{paper_id}.png"
            filename = f"{base_filename}.png"
            return download_file(poster_url, target_dir, filename)
        except Exception as e:
            logger.debug(f"Failed to download poster from constructed URL for paper {paper_id}: {e}")

    return None


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def run_server(host="127.0.0.1", port=5000, debug=False):
    """
    Run the Flask web server.

    Parameters
    ----------
    host : str
        Host to bind to (default: 127.0.0.1)
    port : int
        Port to bind to (default: 5000)
    debug : bool
        Enable debug mode (default: False)
    """
    config = get_config()  # Get config lazily
    print("Starting NeurIPS Abstracts Web Interface...")
    print(f"Database: {config.paper_db_path}")
    print(f"Embeddings: {config.embedding_db_path}")
    print(f"Server: http://{host}:{port}")
    print("\nPress CTRL+C to stop the server")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeurIPS Abstracts Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
