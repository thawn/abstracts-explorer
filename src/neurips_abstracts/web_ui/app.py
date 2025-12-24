"""
Flask web application for NeurIPS Abstracts.

Provides a web interface for searching papers, chatting with RAG,
and exploring the NeurIPS abstracts database.
"""

import os
import logging
import zipfile
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, jsonify, g, send_file, Response
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
    Get database statistics, optionally filtered by year and conference.

    Query parameters:
    - year: int (optional) - Filter by specific year
    - conference: str (optional) - Filter by specific conference

    Returns
    -------
    dict
        Statistics including paper count, year, and conference
    """
    try:
        # Get optional query parameters
        year_param = request.args.get("year")
        conference_param = request.args.get("conference")

        # Convert year to int if provided
        year = int(year_param) if year_param else None
        conference = conference_param if conference_param else None

        database = get_database()

        # Build WHERE clause for filtered count
        conditions = []
        parameters = []

        if year is not None:
            conditions.append("year = ?")
            parameters.append(year)

        if conference is not None:
            conditions.append("conference = ?")
            parameters.append(conference)

        if conditions:
            where_clause = " AND ".join(conditions)
            result = database.query(f"SELECT COUNT(*) as count FROM papers WHERE {where_clause}", tuple(parameters))
            total_papers = result[0]["count"] if result else 0
        else:
            total_papers = database.get_paper_count()

        return jsonify(
            {
                "total_papers": total_papers,
                "year": year,
                "conference": conference,
            }
        )
    except ValueError as e:
        return jsonify({"error": f"Invalid year parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/filters")
def get_filters():
    """
    Get available filter options, optionally filtered by year and conference.

    Query parameters:
    - year: int (optional) - Filter by specific year
    - conference: str (optional) - Filter by specific conference

    Returns
    -------
    dict
        Dictionary with sessions, years, and conferences lists
    """
    try:
        # Get optional query parameters
        year_param = request.args.get("year")
        conference_param = request.args.get("conference")

        # Convert year to int if provided
        year = int(year_param) if year_param else None
        conference = conference_param if conference_param else None

        database = get_database()
        filters = database.get_filter_options(year=year, conference=conference)
        return jsonify(filters)
    except ValueError as e:
        return jsonify({"error": f"Invalid year parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/available-filters")
def get_available_filters():
    """
    Get available conferences and years from registered plugins.

    Returns a mapping of conferences to their supported years based on
    the registered downloader plugins.

    Returns
    -------
    dict
        Dictionary with:
        - conferences: list of conference names
        - years: list of all unique years across all plugins
        - conference_years: dict mapping conference names to their supported years
    """
    try:
        from neurips_abstracts.plugins import list_plugins

        # Get all registered plugins
        plugins = list_plugins()

        # Build mapping of conferences to years
        conference_years = {}
        all_years = set()

        for plugin_info in plugins:
            conference_name = plugin_info.get("conference_name")
            supported_years = plugin_info.get("supported_years", [])

            if conference_name and supported_years:
                if conference_name not in conference_years:
                    conference_years[conference_name] = []
                conference_years[conference_name].extend(supported_years)
                all_years.update(supported_years)

        # Sort years and deduplicate
        all_years_sorted = sorted(list(all_years), reverse=True)
        conferences = sorted(conference_years.keys())

        # Sort years for each conference
        for conf in conference_years:
            conference_years[conf] = sorted(conference_years[conf], reverse=True)

        return jsonify({"conferences": conferences, "years": all_years_sorted, "conference_years": conference_years})
    except Exception as e:
        logger.error(f"Error in available-filters endpoint: {e}", exc_info=True)
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
    - years: list[int] - Filter by years (optional)
    - conferences: list[str] - Filter by conferences (optional)

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
        years = data.get("years", [])
        conferences = data.get("conferences", [])

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
            if years:
                # Convert years to integers for ChromaDB
                year_ints = [int(y) for y in years]
                filter_conditions.append({"year": {"$in": year_ints}})
            if conferences:
                filter_conditions.append({"conference": {"$in": conferences}})

            # Use $or operator if multiple conditions, otherwise use single condition
            where_filter = None
            if len(filter_conditions) > 1:
                where_filter = {"$and": filter_conditions}
            elif len(filter_conditions) == 1:
                where_filter = filter_conditions[0]

            logger.info(f"Search filter: sessions={sessions}, years={years}, conferences={conferences}")
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
                keyword=query,
                sessions=sessions,
                years=years,
                conferences=conferences,
                limit=limit,
            )

            # Convert to list of dicts for JSON serialization
            papers = [dict(p) for p in papers]

            # Parse authors from comma-separated string for each paper
            for paper in papers:
                if "authors" in paper and paper["authors"]:
                    paper["authors"] = [a.strip() for a in paper["authors"].split(";")]
                else:
                    paper["authors"] = []

        return jsonify({"papers": papers, "count": len(papers), "query": query, "use_embeddings": use_embeddings})
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/paper/<string:paper_uid>")
def get_paper(paper_uid):
    """
    Get a specific paper by UID.

    Parameters
    ----------
    paper_uid : str
        Paper UID (unique identifier string)

    Returns
    -------
    dict
        Paper details including authors
    """
    try:
        database = get_database()
        paper = get_paper_with_authors(database, paper_uid)
        return jsonify(paper)
    except PaperFormattingError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/papers/batch", methods=["POST"])
def get_papers_batch():
    """
    Get multiple papers by their UIDs.

    Parameters
    ----------
    paper_ids : list of str
        List of paper UIDs to fetch (note: parameter name kept for compatibility)

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

        for paper_uid in paper_ids:
            try:
                # Convert to string if needed (JavaScript might send as string or int)
                paper_uid_str = str(paper_uid)
                paper = get_paper_with_authors(database, paper_uid_str)
                papers.append(paper)
            except PaperFormattingError as e:
                logger.warning(f"Paper {paper_uid} not found: {e}")
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
    - years: list (optional) - Filter by years
    - conferences: list (optional) - Filter by conferences

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
        years = data.get("years", [])
        conferences = data.get("conferences", [])

        if not message:
            return jsonify({"error": "Message is required"}), 400

        rag = get_rag_chat()

        if reset:
            rag.reset_conversation()

        # Build metadata filter
        filter_conditions = []
        if sessions:
            filter_conditions.append({"session": {"$in": sessions}})
        if years:
            # Convert years to integers for ChromaDB
            year_ints = [int(y) for y in years]
            filter_conditions.append({"year": {"$in": year_ints}})
        if conferences:
            filter_conditions.append({"conference": {"$in": conferences}})

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


def natural_sort_key(s):
    """
    Generate a sort key for natural sorting of strings with numbers.

    Parameters
    ----------
    s : str
        String to generate sort key for

    Returns
    -------
    tuple
        Sort key that enables natural number sorting
    """
    import re

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r"(\d+)", s)]


def fetch_conference_info():
    """
    Fetch conference information from NeurIPS website.

    Returns
    -------
    dict or None
        Conference information dictionary with keys: name, dates, location, description
        Returns None if fetching fails
    """
    try:
        # Try to import BeautifulSoup
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed, using fallback conference info")
            return None

        import re

        conference_url = "https://neurips.cc/Conferences/2025"
        response = requests.get(conference_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        conf_info = {
            "name": "38th Conference on Neural Information Processing Systems (NeurIPS 2025)",
            "dates": None,
            "location": None,
            "description": None,
        }

        # Try to find conference title
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if "NeurIPS" in title_text or "Neural Information Processing Systems" in title_text:
                conf_info["name"] = title_text

        # Try to find dates - look for patterns like "December 9-15, 2025"
        page_text = soup.get_text()
        date_patterns = [
            r"(December\s+\d+[-–]\d+,\s+\d{4})",
            r"(Dec\s+\d+[-–]\d+,\s+\d{4})",
            r"(\d{1,2}[-–]\d{1,2}\s+December\s+\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["dates"] = match.group(1)
                break

        # Try to find location - look for "Vancouver" mentions
        location_patterns = [
            r"(Vancouver\s+Convention\s+Centre(?:,\s+(?:Vancouver|BC|British\s+Columbia))?(?:,\s+Canada)?)",
            r"(Vancouver(?:,\s+(?:BC|British\s+Columbia))?(?:,\s+Canada)?)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["location"] = match.group(1)
                break

        # Try to find conference description/about text
        about_section = soup.find(["div", "p"], class_=re.compile(r"about|description", re.I))
        if about_section:
            desc_text = about_section.get_text(strip=True)
            # Limit description length
            if len(desc_text) > 500:
                desc_text = desc_text[:500] + "..."
            conf_info["description"] = desc_text

        logger.info(f"Fetched conference info: {conf_info}")
        return conf_info

    except Exception as e:
        logger.warning(f"Failed to fetch conference info from website: {e}")
        return None


def generate_all_papers_markdown(papers, title):
    """
    Generate markdown for all papers in a single file.

    Parameters
    ----------
    papers : list
        List of papers
    title : str
        Title for the markdown file

    Returns
    -------
    str
        Markdown content
    """
    markdown = f"# {title}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    # Group by session
    sessions = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(paper)

    # Write each session
    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            stars = "⭐" * paper.get("priority", 0)
            markdown += f"### {paper.get('title', 'Untitled')}\n\n"
            markdown += f"**Rating:** {stars} ({paper.get('priority', 0)}/5)\n\n"

            if paper.get("searchTerm"):
                markdown += f"**Search Term:** {paper.get('searchTerm')}\n\n"

            if paper.get("authors"):
                authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
                markdown += f"**Authors:** {authors}\n\n"

            if paper.get("poster_position"):
                markdown += f"**Poster:** {paper['poster_position']}\n\n"

            # Link to PDF on OpenReview
            paper_id = paper["uid"]
            pdf_url = paper.get("paper_pdf_url")
            if not pdf_url and paper.get("paper_url"):
                pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
            if pdf_url:
                markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

            if paper.get("paper_url"):
                markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

            if paper.get("url"):
                markdown += f"**Source URL:** {paper['url']}\n\n"

            if paper.get("abstract"):
                markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

            # Link to poster image
            poster_url = get_poster_url(paper)
            if poster_url:
                markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

            markdown += "---\n\n"

    return markdown


def generate_folder_structure_export(papers, search_query, sort_order="search-rating-poster"):
    """
    Generate a zip file with folder structure respecting the sort order.

    File organization based on first sort priority:
    - search-rating-poster: Separate files per search term
    - rating-poster-search: Separate files per rating level
    - poster-search-rating: Single file with all papers (poster # is first)

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    search_query : str
        Search query context
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    BytesIO
        Buffer containing zip file
    """
    import re

    # Create in-memory zip file
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Generate main README.md with conference information
        readme_content = generate_main_readme(papers, search_query, sort_order)
        zipf.writestr("README.md", readme_content)

        if sort_order == "poster-search-rating":
            # Poster number is first priority - all papers in one single file
            all_papers_markdown = generate_all_papers_markdown(papers, "All Papers (by Poster #)")
            zipf.writestr("all_papers.md", all_papers_markdown)

        elif sort_order == "rating-poster-search":
            # Rating is first priority - split by rating
            priority_groups = {}
            for paper in papers:
                priority = paper.get("priority", 0)
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append(paper)

            # Create a markdown file for each priority rating
            for priority in sorted(priority_groups.keys(), reverse=True):  # Higher priorities first
                priority_papers = priority_groups[priority]
                priority_stars = "⭐" * priority
                priority_name = f"{priority}_stars" if priority > 0 else "0_stars"

                # Generate markdown for this priority level
                priority_markdown = generate_all_papers_markdown(
                    priority_papers, f"{priority_stars} {priority} Stars ({len(priority_papers)} papers)"
                )

                # Write to file named by priority
                zipf.writestr(f"{priority_name}.md", priority_markdown)

        else:  # sort_order == "search-rating-poster"
            # Search term is first priority - split by search term
            search_terms = {}
            for paper in papers:
                search_term = paper.get("searchTerm") or "Unknown"
                if search_term not in search_terms:
                    search_terms[search_term] = []
                search_terms[search_term].append(paper)

            # Create a markdown file for each search term
            for search_term, term_papers in search_terms.items():
                # Sanitize search term for filename
                safe_name = re.sub(r"[^\w\s-]", "", search_term).strip().replace(" ", "_")
                safe_name = safe_name[:50]  # Limit length

                if not safe_name:
                    safe_name = "unknown"

                # Generate markdown for this search term
                term_markdown = generate_search_term_markdown(search_term, term_papers)

                # Write to file named by search term
                zipf.writestr(f"{safe_name}.md", term_markdown)

    zip_buffer.seek(0)
    return zip_buffer


def generate_main_readme(papers, search_query, sort_order="search-rating-poster"):
    """
    Generate main README.md with conference overview and links to search term files.

    Parameters
    ----------
    papers : list
        List of paper dictionaries (already sorted)
    search_query : str
        Search query context
    sort_order : str
        Sort order used ('search-rating-poster', 'rating-poster-search', 'poster-search-rating')

    Returns
    -------
    str
        Markdown content for main README
    """
    from datetime import datetime
    import re

    markdown = "# NeurIPS 2025 - Interesting Papers\n\n"

    markdown += "Generated by [NeurIPS Abstracts](https://github.com/thawn/neurips-abstracts)\n\n"

    # Conference information - fetch from website
    markdown += "## Conference Information\n\n"

    # Try to fetch conference info from neurips.cc
    conf_info = fetch_conference_info()

    if conf_info:
        markdown += f"**Conference:** {conf_info.get('name', '38th Conference on Neural Information Processing Systems (NeurIPS 2025)')}\n\n"
        if conf_info.get("dates"):
            markdown += f"**Dates:** {conf_info['dates']}\n\n"
        if conf_info.get("location"):
            markdown += f"**Location:** {conf_info['location']}\n\n"
        markdown += "**Website:** [https://neurips.cc/](https://neurips.cc/)\n\n"
        if conf_info.get("description"):
            markdown += f"**About:** {conf_info['description']}\n\n"
    else:
        # Fallback to static information if scraping fails
        markdown += "**Conference:** 38th Conference on Neural Information Processing Systems (NeurIPS 2025)\n\n"
        markdown += "**Dates:** December 9-15, 2025\n\n"
        markdown += "**Location:** Vancouver Convention Centre, Vancouver, Canada\n\n"
        markdown += "**Website:** [https://neurips.cc/](https://neurips.cc/)\n\n"
        markdown += (
            "**Topic:** Neural Information Processing Systems - Machine Learning and Artificial Intelligence\n\n"
        )

    markdown += "---\n\n"

    # Export metadata
    markdown += "## Export Information\n\n"
    markdown += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"**Total Papers:** {len(papers)}\n\n"

    # Document sort order
    sort_order_descriptions = {
        "search-rating-poster": "Search Term → Rating → Poster #",
        "rating-poster-search": "Rating → Poster # → Search Term",
        "poster-search-rating": "Poster # → Search Term → Rating",
    }
    sort_desc = sort_order_descriptions.get(sort_order, sort_order)
    markdown += f"**Sort Order:** {sort_desc}\n\n"

    markdown += "---\n\n"

    # Group papers by search term and session
    search_terms = {}
    for paper in papers:
        search_term = paper.get("searchTerm") or "Unknown"
        if search_term not in search_terms:
            search_terms[search_term] = {"count": 0, "sessions": set(), "avg_priority": 0, "priorities": []}
        search_terms[search_term]["count"] += 1
        search_terms[search_term]["priorities"].append(paper.get("priority", 0))
        if paper.get("session"):
            search_terms[search_term]["sessions"].add(paper["session"])

    # Calculate averages
    for term_data in search_terms.values():
        if term_data["priorities"]:
            term_data["avg_priority"] = sum(term_data["priorities"]) / len(term_data["priorities"])

    # Generate table of contents based on sort order
    if sort_order == "poster-search-rating":
        # Single file with all papers
        markdown += "## Papers\n\n"
        markdown += "All papers are organized in a single file: [View All Papers](all_papers.md)\n\n"
        markdown += f"**Total Papers:** {len(papers)}\n\n"

        # Still show search terms summary
        markdown += "### Search Terms Summary\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating |\n"
        markdown += "|-------------|--------|----------|------------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * round(term_data["avg_priority"])
            markdown += f"| {search_term} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    elif sort_order == "rating-poster-search":
        # Files organized by rating
        markdown += "## Papers by Rating\n\n"

        # Group by priority
        priority_groups = {}
        for paper in papers:
            priority = paper.get("priority", 0)
            if priority not in priority_groups:
                priority_groups[priority] = {"count": 0, "search_terms": set()}
            priority_groups[priority]["count"] += 1
            priority_groups[priority]["search_terms"].add(paper.get("searchTerm") or "Unknown")

        markdown += "| Rating | Papers | Search Terms | File |\n"
        markdown += "|--------|--------|--------------|------|\n"

        for priority in sorted(priority_groups.keys(), reverse=True):
            priority_data = priority_groups[priority]
            priority_stars = "⭐" * priority
            priority_name = f"{priority}_stars" if priority > 0 else "0_stars"
            search_terms_str = ", ".join(sorted(priority_data["search_terms"]))
            if len(search_terms_str) > 50:
                search_terms_str = search_terms_str[:50] + "..."
            markdown += f"| {priority_stars} {priority}/5 | {priority_data['count']} | {search_terms_str} | [{priority_name}.md]({priority_name}.md) |\n"

        markdown += "\n### Search Terms Summary\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating |\n"
        markdown += "|-------------|--------|----------|------------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * round(term_data["avg_priority"])
            markdown += f"| {search_term} | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) |\n"

    else:  # search-rating-poster
        # Files organized by search term
        markdown += "## Papers by Search Term\n\n"
        markdown += "| Search Term | Papers | Sessions | Avg Rating | File |\n"
        markdown += "|-------------|--------|----------|------------|------|\n"

        for search_term in sorted(search_terms.keys()):
            term_data = search_terms[search_term]
            safe_name = re.sub(r"[^\w\s-]", "", search_term).strip().replace(" ", "_")
            safe_name = safe_name[:50]
            if not safe_name:
                safe_name = "unknown"

            sessions_str = f"{len(term_data['sessions'])} session(s)"
            avg_stars = "⭐" * round(term_data["avg_priority"])

            markdown += f"| [{search_term}]({safe_name}.md) | {term_data['count']} | {sessions_str} | {avg_stars} ({term_data['avg_priority']:.1f}/5) | {safe_name}.md |\n"

    markdown += "\n---\n\n"

    # Session overview
    sessions = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = {"count": 0, "search_terms": set()}
        sessions[session]["count"] += 1
        sessions[session]["search_terms"].add(paper.get("searchTerm") or "Unknown")

    markdown += "## Sessions Overview\n\n"
    for session in sorted(sessions.keys()):
        session_data = sessions[session]
        markdown += f"### {session}\n\n"
        markdown += f"- **Papers:** {session_data['count']}\n"
        markdown += f"- **Search Terms:** {', '.join(sorted(session_data['search_terms']))}\n\n"

    return markdown


def generate_search_term_markdown(search_term, papers):
    """
    Generate markdown for a single search term with all its papers.

    Parameters
    ----------
    search_term : str
        The search term
    papers : list
        List of papers for this search term

    Returns
    -------
    str
        Markdown content
    """
    markdown = f"# {search_term}\n\n"
    markdown += f"**Papers:** {len(papers)}\n\n"
    markdown += "---\n\n"

    # Group by session
    sessions = {}
    for paper in papers:
        session = paper.get("session") or "No Session"
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(paper)

    # Write each session
    for session in sorted(sessions.keys()):
        session_papers = sessions[session]
        markdown += f"## {session}\n\n"
        markdown += f"**Papers in this session:** {len(session_papers)}\n\n"

        for paper in session_papers:
            stars = "⭐" * paper.get("priority", 0)
            markdown += f"### {paper.get('title', 'Untitled')}\n\n"
            markdown += f"**Rating:** {stars} ({paper.get('priority', 0)}/5)\n\n"

            if paper.get("authors"):
                authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]
                markdown += f"**Authors:** {authors}\n\n"

            if paper.get("poster_position"):
                markdown += f"**Poster:** {paper['poster_position']}\n\n"

            # Link to PDF on OpenReview
            paper_id = paper["uid"]
            pdf_url = paper.get("paper_pdf_url")
            if not pdf_url and paper.get("paper_url"):
                pdf_url = paper["paper_url"].replace("/forum?id=", "/pdf?id=")
            if pdf_url:
                markdown += f"**PDF:** [View on OpenReview]({pdf_url})\n\n"

            if paper.get("paper_url"):
                markdown += f"**Paper URL:** {paper['paper_url']}\n\n"

            if paper.get("url"):
                markdown += f"**Source URL:** {paper['url']}\n\n"

            if paper.get("abstract"):
                markdown += f"**Abstract:**\n\n{paper['abstract']}\n\n"

            # Link to poster image
            poster_url = get_poster_url(paper)
            if poster_url:
                markdown += f"**Poster Image:** ![Poster]({poster_url})\n\n"

            markdown += "---\n\n"

    return markdown


@app.route("/api/export/interesting-papers", methods=["POST"])
def export_interesting_papers():
    """
    Export interesting papers to a zip file with folder structure.

    Parameters
    ----------
    paper_ids : list of int
        List of paper IDs to export
    priorities : dict
        Dictionary mapping paper IDs to priority ratings (1-5)
    search_query : str, optional
        Search query context

    Returns
    -------
    file
        Zip file containing folder structure with README.md and search term markdown files
    """
    try:
        data = request.json
        paper_ids = data.get("paper_ids", [])
        priorities = data.get("priorities", {})
        search_query = data.get("search_query", "")
        sort_order = data.get("sort_order", "search-rating-poster")

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

        # Sort papers based on the selected sort order
        if sort_order == "search-rating-poster":
            # Search term, then priority, then poster position
            papers.sort(
                key=lambda p: (
                    p.get("searchTerm") or "",
                    -p.get("priority", 0),  # Descending priority
                    natural_sort_key(p.get("poster_position") or ""),
                )
            )
        elif sort_order == "rating-poster-search":
            # Priority, then poster position, then search term
            papers.sort(
                key=lambda p: (
                    -p.get("priority", 0),  # Descending priority
                    natural_sort_key(p.get("poster_position") or ""),
                    p.get("searchTerm") or "",
                )
            )
        elif sort_order == "poster-search-rating":
            # Poster position, then search term, then priority
            papers.sort(
                key=lambda p: (
                    natural_sort_key(p.get("poster_position") or ""),
                    p.get("searchTerm") or "",
                    -p.get("priority", 0),  # Descending priority
                )
            )
        else:
            # Default: search term, priority, poster position
            papers.sort(
                key=lambda p: (
                    p.get("searchTerm") or "",
                    -p.get("priority", 0),  # Descending priority
                    natural_sort_key(p.get("poster_position") or ""),
                )
            )

        # Generate zip file with folder structure
        zip_buffer = generate_folder_structure_export(papers, search_query, sort_order)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f'interesting-papers-{papers[0].get("year", "2025")}.zip',
        )

    except Exception as e:
        logger.error(f"Error exporting interesting papers: {e}")
        return jsonify({"error": str(e)}), 500


def get_poster_url(paper):
    """
    Get poster image URL from paper object.

    Parameters
    ----------
    paper : dict
        Paper object containing poster_image_url and original_id fields

    Returns
    -------
    str or None
        Poster image URL if found, None otherwise
    """
    # Use poster_image_url from database if available
    poster_image_url = paper.get("poster_image_url")
    if poster_image_url:
        return poster_image_url

    # Fallback: construct poster URL from original_id
    original_id = paper.get("original_id")
    if original_id:
        return f"https://{paper.get('conference', 'neurips').lower()}.cc/media/PosterPDFs/{paper.get('conference', 'NeurIPS')}%20{paper.get('year', '2025')}/{original_id}.png"

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
