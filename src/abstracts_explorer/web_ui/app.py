"""
Flask web application for Abstracts Explorer.

Provides a web interface for searching papers, chatting with RAG,
and exploring the abstracts database.
"""

import os
import secrets
import signal
import sys
import logging
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.middleware.proxy_fix import ProxyFix

from abstracts_explorer.database import DatabaseManager, normalize_model_name
from abstracts_explorer.clustering import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_REDUCTION_METHOD,
)
from abstracts_explorer.config import get_config
from abstracts_explorer.export_utils import export_papers_to_zip
from abstracts_explorer.paper_utils import extract_top_keywords
from abstracts_explorer.web_ui.dependencies import (
    _RAG_CHAT_CACHE_MAX,  # noqa: F401  (re-exported for tests)
    _get_session_id,  # noqa: F401  (re-exported for tests)
    _rag_chat_cache,  # noqa: F401  (re-exported for tests)
    _reset_rag_chat_local,  # noqa: F401  (re-exported for tests)
    get_database,
    get_embeddings_manager,
    get_llm_backend_info,
    get_rag_chat,
    teardown_db,
)
from abstracts_explorer.web_ui.api_utils import (
    ApiError,
    api_error_handler,
    coerce_int_list,
    coerce_limit,
    coerce_str_list,
    get_conference_year_params,
    get_json_object,
)

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


def _load_secret_key() -> str:
    """Return a stable Flask secret key from the environment or data directory."""
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    key_path = Path(get_config().data_dir) / ".flask_secret"
    try:
        if key_path.exists():
            existing = key_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        generated = secrets.token_hex(32)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(generated, encoding="utf-8")
        key_path.chmod(0o600)
        return generated
    except OSError:
        logger.warning("Could not persist a Flask secret key; using an ephemeral key", exc_info=True)
        return secrets.token_hex(32)


# Initialize Flask app with correct template/static folders
app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"), static_folder=str(PACKAGE_DIR / "static"))
app.secret_key = _load_secret_key()
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 2 * 1024 * 1024))

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

app.teardown_appcontext(teardown_db)


def _allowed_cors_origins() -> list[str]:
    """Return the configured CORS origins, read at request time."""
    return [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "").split(",") if origin.strip()]


@app.before_request
def _handle_api_preflight():
    """Answer CORS preflight requests for API routes."""
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        return app.make_default_options_response()
    return None


@app.after_request
def _apply_api_headers(response):
    """Apply no-store and opt-in CORS headers to API responses."""
    if not request.path.startswith("/api/"):
        return response
    response.headers["Cache-Control"] = "no-store"
    origins = _allowed_cors_origins()
    if origins:
        origin = request.headers.get("Origin")
        if "*" in origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin in origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


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
@api_error_handler
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
    conference, year = get_conference_year_params(request)

    database = get_database()
    stats_data = database.get_stats(year=year, conference=conference)

    return jsonify(stats_data)


@app.route("/api/embedding-model-check")
@api_error_handler
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
    elif normalize_model_name(stored_model) == normalize_model_name(current_model):
        # Models match (ignoring alias- prefixes)
        compatible = True
        warning = None
    else:
        # Models differ
        compatible = False
        warning = (
            f"Embedding model mismatch detected! The embeddings were created with '{stored_model}' "
            f"but the current configuration uses '{current_model}'. Embeddings from different models "
            f"are incompatible. Please recreate the embeddings using: "
            f"abstracts-explorer create-embeddings --force"
        )

    return jsonify(
        {
            "compatible": compatible,
            "stored_model": stored_model,
            "current_model": current_model,
            "warning": warning,
        }
    )


@app.route("/api/filters")
@api_error_handler
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
    conference, year = get_conference_year_params(request)

    database = get_database()
    sessions = database.get_sessions(conference=conference, year=year)
    return jsonify({"sessions": sessions})


@app.route("/api/available-filters")
@api_error_handler
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
    default_result = database.confirm_default_conference_year_exists(configured_conf, configured_year)

    return jsonify(
        {
            "conferences": conferences,
            "years": years_sorted,
            "conference_years": db_conference_years,
            "default_conference": default_result["conference"],
            "default_year": default_result["year"],
            "default_conference_error": default_result["error"],
            "default_distance_threshold": _SIMILAR_DISTANCE_THRESHOLD,
        }
    )


@app.route("/api/search", methods=["POST"])
@api_error_handler
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
    data = get_json_object()
    query = str(data.get("query", "") or "")
    use_embeddings = bool(data.get("use_embeddings", False))
    limit = coerce_limit(data.get("limit", 10))
    sessions = coerce_str_list(data.get("sessions", []), "sessions")
    years = coerce_int_list(data.get("years", []), "years")
    conferences = coerce_str_list(data.get("conferences", []), "conferences")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    database = get_database()

    if use_embeddings:
        # Semantic search using embeddings
        em = get_embeddings_manager()
        distance_threshold = float(data.get("distance_threshold", _SIMILAR_DISTANCE_THRESHOLD))

        papers = database.search_papers_keyword(
            query=query,
            limit=limit,
            sessions=sessions,
            years=years,
            conferences=conferences,
            use_embeddings=True,
            distance_threshold=distance_threshold,
            embeddings_manager=em,
        )

        # Count total similar papers within distance threshold.
        # Parse field filters to determine the semantic portion of the query;
        # when the query consists only of field filters there is no meaningful
        # embedding to compare against, so skip the count.
        field_filters, remaining_query = DatabaseManager.parse_field_filters(query)
        if remaining_query:
            try:
                candidate_ids = database.resolve_paper_uids(
                    conferences=conferences if conferences else None,
                    years=years if years else None,
                    sessions=sessions if sessions else None,
                    field_filters=field_filters or None,
                )
                total_similar = len(
                    em.query_ids_within_distance(
                        query=remaining_query,
                        distance_threshold=distance_threshold,
                        ids=candidate_ids,
                    )["ids"]
                )
            except Exception:
                total_similar = None
        else:
            total_similar = None
    else:
        # Keyword search in database
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


@app.route("/api/paper/<string:paper_uid>")
@api_error_handler
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
    database = get_database()
    paper = database.get_paper_by_uid(paper_uid)
    if paper is None:
        return jsonify({"error": f"Paper with uid={paper_uid} not found"}), 404
    return jsonify(paper)


@app.route("/api/papers/batch", methods=["POST"])
@api_error_handler
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
    data = get_json_object()
    paper_ids = data.get("paper_ids", [])

    if not isinstance(paper_ids, list) or not paper_ids:
        return jsonify({"error": "No paper IDs provided"}), 400
    if len(paper_ids) > 1000:
        raise ApiError("Too many paper IDs (maximum 1000)", 400)

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


@app.route("/api/chat", methods=["POST"])
@api_error_handler
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
    - paper_priorities: dict (optional) - Interesting-paper ratings from the web UI

    Returns
    -------
    dict
        Chat response with papers used
    """
    config = get_config()  # Get config lazily
    data = get_json_object()
    message = str(data.get("message", "") or "")
    n_papers = coerce_limit(
        data.get("n_papers", config.max_context_papers), default=config.max_context_papers, maximum=100
    )
    reset = bool(data.get("reset", False))
    paper_priorities = data.get("paper_priorities", {}) or {}
    if not isinstance(paper_priorities, dict):
        raise ApiError("paper_priorities must be an object", 400)

    # Get filters
    years = coerce_int_list(data.get("years", []), "years")
    conferences = coerce_str_list(data.get("conferences", []), "conferences")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    rag = get_rag_chat()

    if reset:
        rag.reset_conversation()

    # Get response with filters
    database = get_database()
    available_conferences = database.get_conferences()
    response = rag.query(
        message,
        n_results=n_papers,
        conferences=conferences if conferences else None,
        years=years if years else None,
        available_conferences=available_conferences,
        paper_priorities=paper_priorities,
    )

    return jsonify(
        {
            "response": response,
            "message": message,
        }
    )


@app.route("/api/chat/reset", methods=["POST"])
@api_error_handler
def reset_chat():
    """
    Reset the chat conversation.

    Returns
    -------
    dict
        Success message
    """
    session_id = _get_session_id()
    if session_id in _rag_chat_cache:
        _rag_chat_cache[session_id].reset_conversation()
    return jsonify({"success": True, "message": "Conversation reset"})


@app.route("/api/clusters/compute", methods=["POST"])
@api_error_handler
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
    data = get_json_object()

    conferences = coerce_str_list(data.get("conferences"), "conferences") or None
    years = coerce_int_list(data.get("years"), "years") or None

    # Get config and database
    config = get_config()
    database = get_database()

    # Get current embedding model
    current_model = config.embedding_model

    # Fixed clustering parameters (only true clustering params, not conference/year)
    clustering_params = dict(DEFAULT_CLUSTERING_PARAMS)

    # Determine single conference/year for cache lookup
    cache_conference = conferences[0] if conferences and len(conferences) == 1 else None
    cache_year = years[0] if years and len(years) == 1 else None

    # Look up pre-computed results from the cache
    cached = database.get_clustering_cache(
        embedding_model=current_model,
        reduction_method=DEFAULT_REDUCTION_METHOD,
        n_components=DEFAULT_N_COMPONENTS,
        clustering_method=DEFAULT_CLUSTERING_METHOD,
        n_clusters=None,
        clustering_params=clustering_params if clustering_params else None,
        conference=cache_conference,
        year=cache_year,
        collection_name=config.collection_name,
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


@app.route("/api/clusters/search", methods=["POST"])
@api_error_handler
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
    data = get_json_object()

    if not data.get("query"):
        return jsonify({"error": "Missing required field: query"}), 400

    query = str(data["query"])
    try:
        distance_threshold = float(data.get("distance", 1.1))
    except (TypeError, ValueError) as exc:
        raise ApiError("distance must be a number", 400) from exc
    conferences = coerce_str_list(data.get("conferences"), "conferences") or None
    years = coerce_int_list(data.get("years"), "years") or None

    # Get embeddings manager and database
    em = get_embeddings_manager()
    database = get_database()

    # Call DatabaseManager orchestration method
    results = database.find_papers_within_distance(
        embeddings_manager=em,
        query=query,
        distance_threshold=distance_threshold,
        conferences=conferences,
        years=years,
    )

    if isinstance(results, dict) and isinstance(results.get("papers"), list):
        results["papers"] = results["papers"][:500]
        results["count"] = len(results["papers"])
    return jsonify(results)


@app.route("/api/topic-evolution", methods=["POST"])
@api_error_handler(error_message="Failed to compute topic evolution")
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
    data = get_json_object()

    if not data.get("topic_keywords"):
        return jsonify({"error": "Missing required field: topic_keywords"}), 400

    topic_keywords = str(data["topic_keywords"])
    conferences = coerce_str_list(data.get("conferences"), "conferences") or None
    try:
        distance_threshold = float(data.get("distance_threshold", 1.1))
    except (TypeError, ValueError) as exc:
        raise ApiError("distance_threshold must be a number", 400) from exc

    from abstracts_explorer.mcp_server import get_topic_evolution as mcp_get_topic_evolution

    result_json = mcp_get_topic_evolution(
        topic_keywords=topic_keywords,
        conferences=conferences,
        distance_threshold=distance_threshold,
    )

    result = json.loads(result_json)
    return jsonify(result)


@app.route("/api/papers-per-year")
@api_error_handler(error_message="Failed to load papers per year data")
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
    conference = request.args.get("conference")
    database = get_database()

    if conference:
        years = sorted(database.get_years(conference=conference))
    else:
        years = sorted(database.get_years())

    year_counts = {}
    for year in years:
        stats = database.get_stats(year=year, conference=conference)
        year_counts[year] = stats["total_papers"]

    return jsonify({"year_counts": year_counts, "conference": conference})


@app.route("/api/export/interesting-papers", methods=["POST"])
@api_error_handler
def export_interesting_papers():
    """
    Export interesting papers to a zip file with folder structure.

    Parameters
    ----------
    paper_ids : list of int
        List of paper IDs to export
    priorities : dict
        Dictionary mapping paper IDs to dicts with ``priority`` and
        ``searchTerm`` fields
    search_query : str, optional
        Search query context

    Returns
    -------
    file
        Zip file containing folder structure with README.md and search term markdown files
    """
    data = get_json_object()
    paper_ids = data.get("paper_ids", [])
    priorities = data.get("priorities", {})
    if not isinstance(paper_ids, list):
        raise ApiError("paper_ids must be a list", 400)
    if len(paper_ids) > 1000:
        raise ApiError("Too many paper IDs (maximum 1000)", 400)
    sort_order = data.get("sort_order", "search-rating-poster")

    if not paper_ids:
        return jsonify({"error": "No paper IDs provided"}), 400

    if not isinstance(priorities, dict):
        return jsonify({"error": "Invalid priorities format. Expected a dict keyed by paper UID."}), 400

    for paper_uid, priority_data in priorities.items():
        if not isinstance(priority_data, dict):
            return (
                jsonify(
                    {
                        "error": (
                            f"Invalid rating format for paper '{paper_uid}': expected a dict with "
                            "'priority' and optional 'searchTerm'."
                        )
                    }
                ),
                400,
            )

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
            try:
                priority_value = int(priority_data.get("priority", 0))
            except (TypeError, ValueError) as exc:
                raise ApiError(f"Invalid priority for paper '{paper_id}'", 400) from exc
            paper["priority"] = max(0, min(priority_value, 5))
            paper["searchTerm"] = str(priority_data.get("searchTerm", "Unknown"))

            papers.append(paper)
        except Exception:
            logger.warning(f"Paper {paper_id} not found")
            continue

    if not papers:
        return jsonify({"error": "No papers found"}), 404

    # Use export utility function to sort and generate ZIP
    zip_buffer = export_papers_to_zip(papers, sort_order)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f'interesting-papers-{papers[0].get("year", "2025")}.zip',
    )


@app.route("/api/donate-data", methods=["POST"])
@api_error_handler
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
    data = get_json_object()
    paper_priorities = data.get("paperPriorities", {})

    if not paper_priorities:
        return jsonify({"error": "No data provided"}), 400

    database = get_database()

    try:
        # Use DatabaseManager's donate_validation_data method
        donated_count = database.donate_validation_data(paper_priorities)
    except ValueError as e:
        # Validation errors return 400
        raise ApiError(str(e), 400) from e

    return jsonify(
        {
            "success": True,
            "message": f"Successfully donated {donated_count} paper(s). Thank you for contributing!",
            "count": donated_count,
        }
    )


@app.route("/api/donate-chat", methods=["POST"])
@api_error_handler
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
    data = get_json_object()
    rating = data.get("rating")
    transcript = data.get("transcript")

    if not rating or not transcript:
        return jsonify({"error": "Both 'rating' and 'transcript' are required"}), 400

    database = get_database()

    try:
        donation_id = database.donate_chat_transcript(rating, transcript)
    except ValueError as e:
        raise ApiError(str(e), 400) from e

    return jsonify(
        {
            "success": True,
            "message": "Thank you for your feedback! 🎉❤️",
            "id": donation_id,
        }
    )


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

    if debug and host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "⚠️  Debug mode is disabled for non-loopback hosts; use -v for INFO logging instead.",
            file=sys.stderr,
        )
        debug = False

    # Check if database is accessible before starting server
    # For SQLite databases, check if the file exists
    if config.database_url.startswith("sqlite:///"):
        db_path = config.database_url.replace("sqlite:///", "")
        if not os.path.exists(db_path):
            print("\n❌ Error: Database not found!", file=sys.stderr)
            print(f"\nThe database file does not exist: {db_path}", file=sys.stderr)
            print("\nTo create and populate the database, run one of these commands:", file=sys.stderr)
            print("  # Download NeurIPS papers:", file=sys.stderr)
            print("  abstracts-explorer download --conference neurips --year 2025", file=sys.stderr)
            print("\n  # Or use a different conference/year:", file=sys.stderr)
            print("  abstracts-explorer download --conference iclr --year 2025", file=sys.stderr)
            print("\n  # List available plugins:", file=sys.stderr)
            print("  abstracts-explorer download --list-plugins", file=sys.stderr)
            print("\nAfter downloading papers, you may also want to create embeddings:", file=sys.stderr)
            print("  abstracts-explorer create-embeddings", file=sys.stderr)
            raise FileNotFoundError(f"Database not found: {db_path}")
    # For PostgreSQL, we can't check file existence - connection will be validated at runtime

    print("Starting Abstracts Explorer Web Interface...")
    print(f"Database: {config.masked_database_url}")

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
