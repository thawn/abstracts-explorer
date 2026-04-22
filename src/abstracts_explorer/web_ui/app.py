"""
Flask web application for Abstracts Explorer.

Provides a web interface for searching papers, chatting with RAG,
and exploring the abstracts database.
"""

import os
import signal
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, render_template, request, jsonify, g, send_file, abort
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat
from abstracts_explorer.config import get_config
from abstracts_explorer.export_utils import export_papers_to_zip
from abstracts_explorer.paper_utils import extract_top_keywords

# Import version
try:
    from abstracts_explorer._version import __version__
except ImportError:
    # Fallback if _version.py doesn't exist (e.g., editable install without build)
    from abstracts_explorer import __version__

logger = logging.getLogger(__name__)

# Default distance threshold for semantic search (L2 distance in embedding space).
# Lower values are stricter (fewer, more similar results).
# Users can adjust this per-search via the web UI settings or --distance-threshold CLI flag.
_SIMILAR_DISTANCE_THRESHOLD = 1.2

# Get the directory where this file is located
PACKAGE_DIR = Path(__file__).parent

# Initialize Flask app with correct template/static folders
app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"), static_folder=str(PACKAGE_DIR / "static"))
CORS(app)

# Apply ProxyFix so Flask correctly interprets X-Forwarded-* headers set by a
# reverse proxy (e.g. nginx).  The hop counts are read from environment variables
# (PROXY_X_FOR, PROXY_X_PROTO, PROXY_X_HOST, PROXY_X_PREFIX) so the number of
# trusted proxy layers can be tuned in docker-compose.yml or the .env file
# without rebuilding the image.  The defaults (1) match the single nginx layer
# in docker-compose.yml.  Set any value to 0 to disable that header entirely.
# When the app is run directly (no proxy), no X-Forwarded-* headers are present
# so ProxyFix is a no-op regardless of the configured values.
_config = get_config()
app.wsgi_app = ProxyFix(  # type: ignore[assignment]
    app.wsgi_app,
    x_for=_config.proxy_x_for,
    x_proto=_config.proxy_x_proto,
    x_host=_config.proxy_x_host,
    x_prefix=_config.proxy_x_prefix,
)

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
        # Database configuration comes from config file
        g.db = DatabaseManager()
        g.db.connect()  # Explicitly connect to the database
        g.db.create_tables()  # Ensure all tables exist (including new ones like clustering_cache)
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


# Known LLM backend definitions: each entry maps a URL fragment to backend metadata.
# Order matters – more specific patterns should come first.
_LLM_BACKENDS: List[Dict[str, Any]] = [
    {
        "url_fragments": ["blablador.fz-juelich.de"],
        "name": "BLABLADOR",
        "homepage": "https://helmholtz-blablador.fz-juelich.de",
        "logo": "blablador-logo.png",
    },
    {
        "url_fragments": ["chat.fz-rossendorf.de"],
        "name": "chat.fz-rossendorf.de",
        "homepage": "https://chat.fz-rossendorf.de",
        "logo": None,
    },
    {
        "url_fragments": ["localhost:1234", "127.0.0.1:1234"],
        "name": "LM Studio",
        "homepage": "https://lmstudio.ai",
        "logo": None,
    },
]


def get_llm_backend_info(backend_url: str) -> Dict[str, Optional[str]]:
    """
    Detect the LLM backend from its URL and return display metadata.

    Parameters
    ----------
    backend_url : str
        The URL of the configured LLM backend (e.g. from ``config.llm_backend_url``).

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``name`` : str or None
            Human-readable backend name, or ``None`` if unknown.
        ``homepage`` : str or None
            URL of the backend's homepage, or ``None`` if unknown.
        ``logo`` : str or None
            Static-file name of the backend logo (relative to the ``static/``
            directory), or ``None`` if no logo is available.
    """
    for backend in _LLM_BACKENDS:
        if any(fragment in backend_url for fragment in backend["url_fragments"]):
            return {
                "name": backend["name"],
                "homepage": backend["homepage"],
                "logo": backend["logo"],
            }
    return {"name": None, "homepage": None, "logo": None}


@app.route("/")
def index():
    """
    Render the main page.

    Returns
    -------
    str
        Rendered HTML template
    """
    llm_backend = get_llm_backend_info(_config.llm_backend_url)
    return render_template(
        "index.html",
        version=__version__,
        imprint_link=_config.imprint_link,
        url_conference=None,
        url_conference_error=None,
        llm_backend=llm_backend,
    )


@app.route("/<conference_name>")
def conference_index(conference_name):
    """
    Render the main page with a specific conference pre-selected.

    Resolves the conference name case-insensitively against known
    conferences from plugins and the database. When no match is found,
    renders the page with an error listing the available conferences.

    Paths starting with a dot (e.g. ``.well-known``) are excluded so
    that ACME challenges and other hidden-path conventions are not
    intercepted.

    Parameters
    ----------
    conference_name : str
        Conference name from the URL path (case-insensitive).

    Returns
    -------
    str
        Rendered HTML template with conference context
    """
    # Don't intercept hidden paths (e.g. .well-known used by Let's Encrypt)
    if conference_name.startswith("."):
        abort(404)

    llm_backend = get_llm_backend_info(_config.llm_backend_url)
    try:
        database = get_database()
        result = database.resolve_conference_for_url(conference_name)
        return render_template(
            "index.html",
            version=__version__,
            imprint_link=_config.imprint_link,
            url_conference=result["conference"],
            url_conference_error=result["error"],
            llm_backend=llm_backend,
        )
    except Exception as e:
        logger.error(f"Error in conference URL route: {e}", exc_info=True)
        return render_template(
            "index.html",
            version=__version__,
            imprint_link=_config.imprint_link,
            url_conference=None,
            url_conference_error=None,
            llm_backend=llm_backend,
        )


@app.route("/health")
def health():
    """
    Health check endpoint for container orchestration.

    Returns
    -------
    tuple
        JSON response with status and HTTP status code
    """
    try:
        # Check if database is accessible
        db = get_database()
        db.get_paper_count()
        return jsonify({"status": "healthy", "service": "abstracts-explorer"}), 200
    except Exception as e:
        # Log the actual error for debugging
        logger.error(f"Health check failed: {e}", exc_info=True)
        # Return generic message for security
        return jsonify({"status": "unhealthy", "error": "Service unavailable"}), 503


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
        stats_data = database.get_stats(year=year, conference=conference)

        return jsonify(stats_data)
    except ValueError as e:
        return jsonify({"error": f"Invalid year parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/embedding-model-check")
def check_embedding_model():
    """
    Check if the embedding model in the configuration matches the one in the database.

    Returns
    -------
    dict
        - compatible: bool - True if models match or no model is stored
        - stored_model: str or None - Model stored in database
        - current_model: str - Current model from configuration
        - warning: str or None - Warning message if incompatible
    """
    try:
        config = get_config()
        database = get_database()

        # Get the stored embedding model from the database
        stored_model = database.get_embedding_model()
        current_model = config.embedding_model

        # Check compatibility
        if stored_model is None:
            # No model stored yet, considered compatible
            compatible = True
            warning = None
        elif stored_model == current_model:
            # Models match
            compatible = True
            warning = None
        else:
            # Models differ
            compatible = False
            warning = (
                f"Embedding model mismatch detected! The embeddings were created with '{stored_model}' "
                f"but the current configuration uses '{current_model}'. Embeddings from different models "
                f"are incompatible. Please recreate the embeddings using: "
                f"neurips-abstracts create-embeddings --force"
            )

        return jsonify(
            {
                "compatible": compatible,
                "stored_model": stored_model,
                "current_model": current_model,
                "warning": warning,
            }
        )
    except Exception as e:
        logger.error(f"Error in embedding-model-check endpoint: {e}", exc_info=True)
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
        Dictionary with sessions list filtered by the given conference/year
    """
    try:
        # Get optional query parameters
        year_param = request.args.get("year")
        conference_param = request.args.get("conference")

        # Convert year to int if provided
        year = int(year_param) if year_param else None
        conference = conference_param if conference_param else None

        database = get_database()
        sessions = database.get_sessions(conference=conference, year=year)
        return jsonify({"sessions": sessions})
    except ValueError as e:
        return jsonify({"error": f"Invalid year parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/available-filters")
def get_available_filters_endpoint():
    """
    Get available conferences and years from the database.

    Returns conferences and years that actually have data in the database,
    along with sensible defaults.

    Returns
    -------
    dict
        Dictionary with:
        - conferences: list of conference names available in the DB
        - years: list of all unique years in the DB (descending)
        - conference_years: dict mapping conference names to years with data in the DB
        - default_conference: conference to pre-select (guaranteed to have DB data, if any exists)
        - default_year: year to pre-select (guaranteed to have DB data for the default conference)
    """
    try:
        config = get_config()
        database = get_database()

        db_conference_years = database.get_conference_years_from_db()

        conferences = sorted(db_conference_years.keys())
        all_years: set = set()
        for years in db_conference_years.values():
            all_years.update(years)
        years_sorted = sorted(all_years, reverse=True)

        configured_conf = config.default_conference or ""
        configured_year = config.default_year if config.default_year else None
        effective_conf, effective_year = database.resolve_default_conference_year(configured_conf, configured_year)

        return jsonify(
            {
                "conferences": conferences,
                "years": years_sorted,
                "conference_years": db_conference_years,
                "default_conference": effective_conf,
                "default_year": effective_year,
                "default_distance_threshold": _SIMILAR_DISTANCE_THRESHOLD,
            }
        )
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
            # Allow per-request override; fall back to the module-level default
            distance_threshold = float(data.get("distance_threshold", _SIMILAR_DISTANCE_THRESHOLD))

            papers = em.search_papers_semantic(
                query=query,
                database=database,
                limit=limit,
                sessions=sessions,
                years=years,
                conferences=conferences,
                distance_threshold=distance_threshold,
            )

            # Count total similar papers within distance threshold.
            # Parse field filters to determine the semantic portion of the query;
            # when the query consists only of field filters there is no meaningful
            # embedding to compare against, so skip the count.
            _, remaining_query = DatabaseManager.parse_field_filters(query)
            if remaining_query:
                try:
                    total_similar = em.count_papers_within_distance(
                        database=database,
                        query=remaining_query,
                        distance_threshold=distance_threshold,
                        conferences=conferences if conferences else None,
                        years=years if years else None,
                    )
                except Exception:
                    total_similar = None
            else:
                total_similar = None
        else:
            # Keyword search in database
            database = get_database()
            papers = database.search_papers_keyword(
                query=query,
                limit=limit,
                sessions=sessions,
                years=years,
                conferences=conferences,
            )
            total_similar = None

        related_topics = extract_top_keywords(papers)

        response_data = {
            "papers": papers,
            "count": len(papers),
            "query": query,
            "use_embeddings": use_embeddings,
            "related_topics": related_topics,
        }
        if total_similar is not None:
            response_data["total_similar"] = total_similar

        return jsonify(response_data)
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
        paper = database.get_paper_by_uid(paper_uid)
        if paper is None:
            return jsonify({"error": f"Paper with uid={paper_uid} not found"}), 404
        return jsonify(paper)
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
                paper = database.get_paper_by_uid(paper_uid_str)
                if paper is not None:
                    papers.append(paper)
                else:
                    logger.warning(f"Paper {paper_uid} not found in database")
            except Exception as e:
                logger.warning(f"Error fetching paper {paper_uid}: {e}")
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
        database = get_database()
        available_conferences = database.get_conferences()
        response = rag.query(
            message,
            n_results=n_papers,
            metadata_filter=metadata_filter,
            conferences=conferences if conferences else None,
            years=[int(y) for y in years] if years else None,
            available_conferences=available_conferences,
        )

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


@app.route("/api/clusters/compute", methods=["POST"])
def compute_clusters():
    """
    Return pre-computed clustering results from the database cache.

    Clustering always uses agglomerative clustering with
    distance_threshold=150, linkage=ward, and t-SNE dimensionality
    reduction.  Results must be pre-generated via the CLI
    ``clustering pre-generate`` command.

    Request Body
    ------------
    {
        "conferences": list[str] (optional, filter by conferences),
        "years": list[int] (optional, filter by years)
    }

    Returns
    -------
    dict
        Clustering results with points, statistics, and metadata
    """
    try:
        data = request.get_json() or {}

        conferences = data.get("conferences") or None  # list[str] or None
        years = data.get("years") or None  # list[int] or None

        # Get config and database
        config = get_config()
        database = get_database()

        # Get current embedding model
        current_model = config.embedding_model

        # Fixed clustering parameters (only true clustering params, not conference/year)
        clustering_params = {"linkage": "ward", "distance_threshold": 150.0}

        # Determine single conference/year for cache lookup
        cache_conference = conferences[0] if conferences and len(conferences) == 1 else None
        cache_year = years[0] if years and len(years) == 1 else None

        # Look up pre-computed results from the cache
        cached = database.get_clustering_cache(
            embedding_model=current_model,
            reduction_method="tsne",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            clustering_params=clustering_params if clustering_params else None,
            conference=cache_conference,
            year=cache_year,
        )

        if cached:
            return jsonify(cached)

        return (
            jsonify(
                {
                    "error": "No pre-computed clustering data available for this conference/year combination. "
                    "Run 'abstracts-explorer clustering pre-generate' to generate clustering data.",
                }
            ),
            404,
        )

    except Exception as e:
        logger.error(f"Error retrieving clusters: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters/cached")
def get_cached_clusters():
    """
    Get previously computed clusters from cache file.

    Query Parameters
    ----------------
    file : str (optional)
        Path to cached clustering results JSON file
        Default: "clusters.json" in current directory

    Returns
    -------
    dict
        Clustering results from cache file
    """
    try:
        cache_file = request.args.get("file", "clusters.json")
        cache_path = Path(cache_file)

        if not cache_path.exists():
            return (
                jsonify(
                    {
                        "error": f"Cached clusters file not found: {cache_file}",
                        "hint": "Run 'abstracts-explorer cluster-embeddings --output clusters.json' first",
                    }
                ),
                404,
            )

        with open(cache_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        return jsonify(results)

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing cached clusters: {e}", exc_info=True)
        return jsonify({"error": f"Invalid JSON in cache file: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error loading cached clusters: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters/default-count")
def get_default_cluster_count():
    """
    Get the recommended default number of clusters based on embeddings count.

    Returns
    -------
    dict
        Dictionary with:
        - n_clusters: Recommended number of clusters
        - n_papers: Number of papers in embeddings collection
    """
    try:
        from abstracts_explorer.clustering import calculate_default_clusters

        em = get_embeddings_manager()

        # Get embeddings count
        collection_stats = em.get_collection_stats()
        n_papers = collection_stats["count"]

        # Calculate default
        n_clusters = calculate_default_clusters(n_papers)

        return jsonify({"n_clusters": n_clusters, "n_papers": n_papers})
    except Exception as e:
        logger.error(f"Error calculating default cluster count: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters/search", methods=["POST"])
def search_custom_cluster():
    """
    Find papers within a specified distance from a custom search query.

    This endpoint treats the search query as a clustering center and returns
    papers within the specified Euclidean distance radius in embedding space.

    Request Body
    ------------
    {
        "query": str (required) - The search query text
        "distance": float (optional, default: 1.1) - Euclidean distance radius
        "conferences": list[str] (optional) - Filter by conferences
        "years": list[int] (optional) - Filter by years
    }

    Returns
    -------
    dict
        {
            "query": str - The search query
            "query_embedding": list[float] - The generated embedding for the query
            "distance": float - The distance threshold used
            "papers": list[dict] - Papers within the distance radius with their distances
            "count": int - Number of papers found
        }
    """
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing required field: query"}), 400

        query = data["query"]
        distance_threshold = data.get("distance", 1.1)
        conferences = data.get("conferences")
        years = data.get("years")

        # Get embeddings manager and database
        em = get_embeddings_manager()
        database = get_database()

        # Call EmbeddingsManager method directly
        results = em.find_papers_within_distance(
            database=database,
            query=query,
            distance_threshold=distance_threshold,
            conferences=conferences,
            years=years,
        )

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error searching custom cluster: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/topic-evolution", methods=["POST"])
def topic_evolution():
    """
    Get topic evolution data for a given topic across years.

    Uses the MCP ``get_topic_evolution`` tool to compute how many papers
    match a topic query per year for the requested conference(s).

    Request Body
    ------------
    {
        "topic_keywords": str (required) - Topic keywords to search for
        "conferences": list[str] (optional) - Conference names
        "distance_threshold": float (optional, default: 1.1) - Distance threshold
    }

    Returns
    -------
    dict
        Topic evolution data with per-conference year_counts and year_relative
    """
    try:
        data = request.get_json()

        if not data or "topic_keywords" not in data:
            return jsonify({"error": "Missing required field: topic_keywords"}), 400

        topic_keywords = data["topic_keywords"]
        conferences = data.get("conferences")
        distance_threshold = data.get("distance_threshold", 1.1)

        from abstracts_explorer.mcp_server import get_topic_evolution as mcp_get_topic_evolution

        result_json = mcp_get_topic_evolution(
            topic_keywords=topic_keywords,
            conferences=conferences,
            distance_threshold=distance_threshold,
        )

        result = json.loads(result_json)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in topic evolution endpoint: {e}", exc_info=True)
        return jsonify({"error": "Failed to compute topic evolution"}), 500


@app.route("/api/papers-per-year")
def papers_per_year():
    """
    Get paper counts per year for a conference.

    Query Parameters
    ----------------
    conference : str (optional)
        Conference name to filter by

    Returns
    -------
    dict
        Dictionary with year_counts mapping year to paper count
    """
    try:
        conference = request.args.get("conference")
        database = get_database()

        if conference:
            years = database.get_years_for_conference(conference)
        else:
            years = sorted(database.get_years())

        year_counts = {}
        for year in years:
            stats = database.get_stats(year=year, conference=conference)
            year_counts[year] = stats["total_papers"]

        return jsonify({"year_counts": year_counts, "conference": conference})

    except Exception as e:
        logger.error(f"Error in papers-per-year endpoint: {e}", exc_info=True)
        return jsonify({"error": "Failed to load papers per year data"}), 500


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
                paper = database.get_paper_by_uid(str(paper_id))
                if paper is None:
                    logger.warning(f"Paper {paper_id} not found")
                    continue
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
            except Exception:
                logger.warning(f"Paper {paper_id} not found")
                continue

        if not papers:
            return jsonify({"error": "No papers found"}), 404

        # Use export utility function to sort and generate ZIP
        zip_buffer = export_papers_to_zip(papers, search_query, sort_order)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f'interesting-papers-{papers[0].get("year", "2025")}.zip',
        )

    except Exception as e:
        logger.error(f"Error exporting interesting papers: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/donate-data", methods=["POST"])
def donate_data():
    """
    Donate anonymized interesting papers data for validation purposes.

    Parameters
    ----------
    paperPriorities : dict
        Dictionary mapping paper UIDs to priority data (dict with priority and searchTerm)

    Returns
    -------
    dict
        Success message with count of donated papers
    """
    try:
        data = request.json
        paper_priorities = data.get("paperPriorities", {})

        if not paper_priorities:
            return jsonify({"error": "No data provided"}), 400

        database = get_database()

        try:
            # Use DatabaseManager's donate_validation_data method
            donated_count = database.donate_validation_data(paper_priorities)

            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully donated {donated_count} paper(s). Thank you for contributing!",
                    "count": donated_count,
                }
            )

        except ValueError as e:
            # Validation errors return 400
            return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Error donating data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/donate-chat", methods=["POST"])
def donate_chat():
    """
    Donate a chat transcript with thumbs up/down feedback.

    Parameters
    ----------
    rating : str
        Feedback rating ('up' or 'down').
    transcript : list
        List of message dicts with 'role' and 'text' keys.

    Returns
    -------
    dict
        Success message with donation ID.
    """
    try:
        data = request.json
        rating = data.get("rating")
        transcript = data.get("transcript")

        if not rating or not transcript:
            return jsonify({"error": "Both 'rating' and 'transcript' are required"}), 400

        database = get_database()

        try:
            donation_id = database.donate_chat_transcript(rating, transcript)

            return jsonify(
                {
                    "success": True,
                    "message": "Thank you for your feedback! 🎉❤️",
                    "id": donation_id,
                }
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Error donating chat transcript: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def run_server(host="127.0.0.1", port=5000, debug=False, dev=False, threads=6):
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
    dev : bool
        Use Flask development server instead of production server (default: False)
    threads : int
        Number of worker threads for Waitress (default: 6). Must be >= 1.

    Raises
    ------
    ValueError
        If threads is less than 1.
    FileNotFoundError
        If the database file does not exist.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

    config = get_config()  # Get config lazily

    # Check if database is accessible before starting server
    # For SQLite databases, check if the file exists
    if config.database_url.startswith("sqlite:///"):
        db_path = config.database_url.replace("sqlite:///", "")
        if not os.path.exists(db_path):
            print("\n❌ Error: Database not found!", file=sys.stderr)
            print(f"\nThe database file does not exist: {db_path}", file=sys.stderr)
            print("\nTo create and populate the database, run one of these commands:", file=sys.stderr)
            print("  # Download NeurIPS papers:", file=sys.stderr)
            print("  neurips-abstracts download --conference neurips --year 2025", file=sys.stderr)
            print("\n  # Or use a different conference/year:", file=sys.stderr)
            print("  neurips-abstracts download --conference iclr --year 2025", file=sys.stderr)
            print("\n  # List available plugins:", file=sys.stderr)
            print("  neurips-abstracts list-plugins", file=sys.stderr)
            print("\nAfter downloading papers, you may also want to create embeddings:", file=sys.stderr)
            print("  neurips-abstracts create-embeddings", file=sys.stderr)
            raise FileNotFoundError(f"Database not found: {db_path}")
    # For PostgreSQL, we can't check file existence - connection will be validated at runtime

    print("Starting Abstracts Explorer Web Interface...")
    print(f"Database: {config.database_url}")

    # Print embeddings configuration
    print(f"Embeddings: {config.embedding_db}")

    print(f"Server: http://{host}:{port}")

    # Register SIGTERM handler for graceful shutdown (e.g., docker stop).
    # Without this, PID 1 in a container ignores SIGTERM by default, forcing Docker
    # to wait the full stop-timeout before sending SIGKILL.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    # Use Flask development server if explicitly requested
    if dev:
        print("\n⚠️  Using Flask development server (not suitable for production)")
        print("   Use without --dev flag for production server")
        print("\nPress CTRL+C to stop the server")
        # LGTM: Debug mode only used when explicitly requested via --dev flag or -vv
        # This is intentionally for development/debugging purposes only
        app.run(host=host, port=port, debug=debug)
    else:
        # Use Waitress production server (debug mode works with Waitress too)
        try:
            from waitress import serve  # type: ignore[import-untyped]

            print("\n✅ Using Waitress production WSGI server")
            if debug:
                print("⚠️  Debug mode enabled (logging level set to DEBUG via -vv)")
            print(f"Waitress threads: {threads}")
            print("\nPress CTRL+C to stop the server")
            # Set Flask debug mode even with Waitress for better error messages
            app.debug = debug
            serve(app, host=host, port=port, threads=threads)
        except ImportError:
            print("\n⚠️  Waitress not installed, falling back to Flask development server", file=sys.stderr)
            print("   Install Waitress with: pip install waitress", file=sys.stderr)
            print("   Or install web extras: pip install abstracts-explorer[web]", file=sys.stderr)
            print("\nPress CTRL+C to stop the server")
            app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Abstracts Explorer Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging and Flask debug features; does not change server type)",
    )
    parser.add_argument(
        "--dev", action="store_true", help="Use Flask development server instead of production server"
    )
    parser.add_argument("--threads", type=int, default=6, help="Number of Waitress worker threads")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug, dev=args.dev, threads=args.threads)
