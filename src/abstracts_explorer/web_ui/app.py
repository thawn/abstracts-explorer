"""
Flask web application for Abstracts Explorer.

Provides a web interface for searching papers, chatting with RAG,
and exploring the abstracts database.
"""

import os
import secrets
import signal
import socket
import sys
import logging
import json
from pathlib import Path
from typing import List
from flask import Flask, Response, render_template, request, jsonify, send_file, abort
from werkzeug.middleware.proxy_fix import ProxyFix

from abstracts_explorer.database import normalize_model_name
from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.clustering import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_N_COMPONENTS,
    DEFAULT_NR_LEVELS,
    DEFAULT_REDUCTION_METHOD,
    build_cluster_centers_from_points,
    derive_hierarchy_levels,
)
from abstracts_explorer.embeddings import QueryEmbedding
from abstracts_explorer.export_utils import export_papers_to_zip
from abstracts_explorer.paper_utils import extract_top_keywords
from abstracts_explorer.profiling import profile_span
from abstracts_explorer.web_ui import request_services as services
from abstracts_explorer.web_ui.request_services import (
    _CHAT_SESSION_CACHE_MAX,  # noqa: F401  (re-exported for tests)
    _chat_sessions,  # noqa: F401  (re-exported for tests)
    _get_session_id,  # noqa: F401  (re-exported for tests)
    _reset_rag_chat_local,  # noqa: F401  (re-exported for tests)
    get_llm_backend_info,  # noqa: F401  (re-exported for tests)
)
from abstracts_explorer.web_ui.api_utils import (
    ApiError,
    api_error_handler,
    coerce_int_list,
    coerce_limit,
    coerce_number,
    coerce_str_list,
    coerce_text,
    get_conference_year_params,
    get_json_object,
)
from abstracts_explorer.web_ui.rate_limit import enforce_client_rate_limit
from abstracts_explorer.web_ui.profiling import init_profiling
from abstracts_explorer.web_ui.warmup import warm_up

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

_MAX_FILTER_ITEMS = 100
_MAX_QUERY_LENGTH = 2000
_MAX_CHAT_MESSAGE_LENGTH = 8000
_MAX_PRIORITY_ENTRIES = 1000
_MAX_SEARCH_TERM_LENGTH = 500
_MAX_TRANSCRIPT_MESSAGES = 200
_MAX_TRANSCRIPT_CHARS = 20000
_MAX_CLUSTER_RESULTS = 500
_MAX_DISTANCE = 10000.0
_EXPORT_SORT_ORDERS = {"search-rating-poster", "rating-poster-search", "poster-search-rating"}


def _client_id() -> str:
    """Return a stable identifier for the current client."""
    return request.remote_addr or "unknown"


# Get the directory where this file is located
PACKAGE_DIR = Path(__file__).parent


def _load_secret_key() -> str:
    """Return a stable Flask secret key from the environment or data directory."""
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    key_path = Path(services.get_config().data_dir) / ".flask_secret"
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
_config = services.get_config()
app.wsgi_app = ProxyFix(  # type: ignore[assignment]
    app.wsgi_app,
    x_for=_config.proxy_x_for,
    x_proto=_config.proxy_x_proto,
    x_host=_config.proxy_x_host,
    x_prefix=_config.proxy_x_prefix,
)

app.teardown_appcontext(services.teardown_db)


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
    llm_backend = services.get_llm_backend_info(_config.llm_backend_url)
    return render_template(
        "index.html",
        version=__version__,
        read_only=_config.read_only,
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
    conferences from plugins and the database. Unknown paths and
    conferences without downloaded data return 404 while still rendering
    the helpful error banner.

    Paths starting with a dot (e.g. ``.well-known``) are excluded so
    that ACME challenges and other hidden-path conventions are not
    intercepted.

    Parameters
    ----------
    conference_name : str
        Conference name from the URL path (case-insensitive).

    Returns
    -------
    tuple
        Rendered HTML template and HTTP status code
    """
    # Don't intercept hidden paths (e.g. .well-known used by Let's Encrypt)
    if conference_name.startswith("."):
        abort(404)

    llm_backend = services.get_llm_backend_info(_config.llm_backend_url)
    try:
        database = services.get_database()
        result = database.resolve_conference_for_url(conference_name)
    except DatabaseError as exc:
        logger.error("Database error in conference URL route: %s", exc, exc_info=True)
        return (
            render_template(
                "index.html",
                version=__version__,
                read_only=_config.read_only,
                imprint_link=_config.imprint_link,
                url_conference=None,
                url_conference_error={"message": "Conference data is temporarily unavailable."},
                llm_backend=llm_backend,
            ),
            503,
        )
    except Exception as exc:
        logger.error("Error in conference URL route: %s", exc, exc_info=True)
        return (
            render_template(
                "index.html",
                version=__version__,
                read_only=_config.read_only,
                imprint_link=_config.imprint_link,
                url_conference=None,
                url_conference_error={"message": "Unable to load conference information."},
                llm_backend=llm_backend,
            ),
            500,
        )

    status = 200 if result.get("reason") == "found" else 404
    return (
        render_template(
            "index.html",
            version=__version__,
            read_only=_config.read_only,
            imprint_link=_config.imprint_link,
            url_conference=result["conference"],
            url_conference_error=result["error"],
            llm_backend=llm_backend,
        ),
        status,
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
        db = services.get_database()
        db.get_paper_count()
        return jsonify({"status": "healthy", "service": "abstracts-explorer"}), 200
    except Exception as e:
        # Log the actual error for debugging
        logger.error(f"Health check failed: {e}", exc_info=True)
        # Return generic message for security
        return jsonify({"status": "unhealthy", "error": "Service unavailable"}), 503


def _filter_cluster_results_by_locations(cached: dict, locations: List[str]) -> dict:
    """
    Filter cached clustering points to the selected presentation locations.

    Recomputes cluster statistics and centers for the visible subset. Cluster
    labels and hierarchy are kept because they describe the full clustering.
    """
    selected = set(locations)
    points = []
    for point in cached.get("points", []):
        point_locations = point.get("locations")
        if not point_locations:
            single = point.get("location")
            point_locations = [single] if single else []
        if selected & set(point_locations):
            points.append(point)

    cached["points"] = points
    counts: dict = {}
    for point in points:
        cluster_id = point.get("cluster")
        if cluster_id is None:
            continue
        counts[cluster_id] = counts.get(cluster_id, 0) + 1

    cluster_sizes = {int(key): int(value) for key, value in counts.items() if int(key) >= 0}
    n_noise = sum(1 for point in points if point.get("cluster") == -1)
    statistics = dict(cached.get("statistics") or {})
    statistics.update(
        {
            "cluster_sizes": cluster_sizes,
            "n_clusters": len(cluster_sizes),
            "n_noise": n_noise,
            "total_papers": len(points),
        }
    )
    cached["statistics"] = statistics

    centers = build_cluster_centers_from_points(points)
    cached["cluster_centers"] = centers
    return cached


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
    location = request.args.get("location")

    database = services.get_database()
    stats_data = database.get_stats(
        year=year,
        conference=conference,
        locations=[location] if location else None,
    )

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
    config = services.get_config()
    database = services.get_database()

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
            f"abstracts-explorer embeddings create --force"
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
    location = request.args.get("location")

    database = services.get_database()
    sessions = database.get_sessions(conference=conference, year=year, location=location)
    return jsonify({"sessions": sessions})


@app.route("/api/available-filters")
@api_error_handler
def get_available_filters_endpoint() -> Response:
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
    config = services.get_config()
    database = services.get_database()

    db_conference_years = database.get_conference_years_from_db()
    conference_year_locations = database.get_conference_year_locations()

    conferences = sorted(db_conference_years.keys())
    all_years: set[int] = set()
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
            "conference_year_locations": conference_year_locations,
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
    query = coerce_text(data.get("query"), "query", _MAX_QUERY_LENGTH)
    use_embeddings = bool(data.get("use_embeddings", False))
    limit = coerce_limit(data.get("limit", 10))
    sessions = coerce_str_list(data.get("sessions", []), "sessions", max_items=_MAX_FILTER_ITEMS)
    years = coerce_int_list(data.get("years", []), "years", max_items=_MAX_FILTER_ITEMS)
    conferences = coerce_str_list(data.get("conferences", []), "conferences", max_items=_MAX_FILTER_ITEMS)
    locations = coerce_str_list(data.get("locations", []), "locations", max_items=_MAX_FILTER_ITEMS)
    distance_threshold = coerce_number(
        data.get("distance_threshold"),
        "distance_threshold",
        default=_SIMILAR_DISTANCE_THRESHOLD,
        minimum=0.0,
        maximum=_MAX_DISTANCE,
    )

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if use_embeddings:
        enforce_client_rate_limit(_client_id())

    database = services.get_database()
    embeddings_manager = services.get_embeddings_manager() if use_embeddings else None
    papers, total_similar = database.search(
        query=query,
        limit=limit,
        sessions=sessions,
        years=years,
        conferences=conferences,
        locations=locations,
        use_embeddings=use_embeddings,
        distance_threshold=distance_threshold,
        embeddings_manager=embeddings_manager,
        include_total=use_embeddings,
    )

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
    database = services.get_database()
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

    database = services.get_database()
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
    config = services.get_config()  # Get config lazily
    data = get_json_object()
    message = coerce_text(data.get("message"), "message", _MAX_CHAT_MESSAGE_LENGTH)
    n_papers = coerce_limit(
        data.get("n_papers", config.max_context_papers), default=config.max_context_papers, maximum=100
    )
    reset = bool(data.get("reset", False))
    paper_priorities = data.get("paper_priorities", {}) or {}
    if not isinstance(paper_priorities, dict):
        raise ApiError("paper_priorities must be an object", 400)
    if len(paper_priorities) > _MAX_PRIORITY_ENTRIES:
        raise ApiError(f"paper_priorities must contain at most {_MAX_PRIORITY_ENTRIES} entries", 400)

    # Get filters
    sessions = coerce_str_list(data.get("sessions", []), "sessions", max_items=_MAX_FILTER_ITEMS)
    years = coerce_int_list(data.get("years", []), "years", max_items=_MAX_FILTER_ITEMS)
    conferences = coerce_str_list(data.get("conferences", []), "conferences", max_items=_MAX_FILTER_ITEMS)

    if not message:
        return jsonify({"error": "Message is required"}), 400

    enforce_client_rate_limit(_client_id())

    # Get response with filters
    database = services.get_database()
    available_conferences = database.get_conferences()
    with services.chat_session(reset=reset) as rag:
        response = rag.query(
            message,
            limit=n_papers,
            conferences=conferences if conferences else None,
            years=years if years else None,
            sessions=sessions if sessions else None,
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
    services.reset_chat_history()
    return jsonify({"success": True, "message": "Conversation reset"})


@app.route("/api/clusters/compute", methods=["POST"])
@api_error_handler
def compute_clusters():
    """
    Return pre-computed clustering results from the database cache.

    Clustering always uses BERTopic with Agglomerative clustering
    (distance_threshold=0.5, linkage=ward) on 5-D UMAP embeddings and
    2-D UMAP visualization.  Results must be populated via the CLI
    ``clustering cache populate`` command.

    Request Body
    ------------
    {
        "conferences": list[str] (optional, filter by conferences),
        "years": list[int] (optional, filter by years)
    }

    Returns
    -------
    dict
        Clustering results with points, statistics, hierarchy levels derived
        from the cached tree, and metadata
    """
    data = get_json_object()

    conferences = coerce_str_list(data.get("conferences"), "conferences", max_items=_MAX_FILTER_ITEMS) or None
    years = coerce_int_list(data.get("years"), "years", max_items=_MAX_FILTER_ITEMS) or None
    locations = coerce_str_list(data.get("locations"), "locations", max_items=_MAX_FILTER_ITEMS) or None
    hierarchy_levels = coerce_limit(
        data.get("hierarchy_levels"),
        default=DEFAULT_NR_LEVELS,
        minimum=1,
        maximum=100,
    )
    level_scale = (
        coerce_text(
            data.get("hierarchy_level_scale"),
            "hierarchy_level_scale",
            max_length=32,
            default=DEFAULT_LEVEL_SCALE,
        ).strip()
        or DEFAULT_LEVEL_SCALE
    )
    if level_scale not in ("linear", "lin", "log", "logarithmic"):
        raise ApiError("hierarchy_level_scale must be 'linear' or 'log'", 400)

    # Get config and database
    config = services.get_config()
    database = services.get_database()

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
        hierarchy = cached.get("hierarchy") or {}
        if "tree" not in hierarchy:
            raise ApiError(
                "Cached clustering data is outdated. Regenerate it with "
                "'abstracts-explorer clustering cache populate --force'.",
                409,
            )
        if not locations:
            tree = hierarchy.get("tree") or []
            centers = cached.get("cluster_centers")
            has_centers = isinstance(centers, dict) and all(node.get("center") for node in tree)
            if not has_centers:
                raise ApiError(
                    "Cached clustering data is missing precomputed centers. "
                    "Run 'python scripts/backfill_cluster_centers_median.py'.",
                    409,
                )
        if locations:
            cached = _filter_cluster_results_by_locations(cached, locations)
        cached["hierarchy"] = derive_hierarchy_levels(
            cached,
            nr_levels=hierarchy_levels,
            level_scale=level_scale,
            recompute_centers=bool(locations),
        )
        return jsonify(cached)

    return (
        jsonify(
            {
                "error": "No pre-computed clustering data available for this conference/year combination. "
                "Run 'abstracts-explorer clustering cache populate' to generate clustering data.",
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

    query = coerce_text(data["query"], "query", _MAX_QUERY_LENGTH)
    distance_threshold = coerce_number(
        data.get("distance"),
        "distance",
        default=1.1,
        minimum=0.0,
        maximum=_MAX_DISTANCE,
    )
    conferences = coerce_str_list(data.get("conferences"), "conferences", max_items=_MAX_FILTER_ITEMS) or None
    years = coerce_int_list(data.get("years"), "years", max_items=_MAX_FILTER_ITEMS) or None
    locations = coerce_str_list(data.get("locations"), "locations", max_items=_MAX_FILTER_ITEMS) or None

    # Get embeddings manager and database
    enforce_client_rate_limit(_client_id())
    em = services.get_embeddings_manager()
    database = services.get_database()

    query_embedding = QueryEmbedding(query, em)
    matches = database.find_paper_ids_within_distance(
        query_embedding,
        distance_threshold,
        conferences=conferences,
        years=years,
        locations=locations,
    )
    papers = database.group_papers_by_content(database.get_result_details(matches, slice(0, _MAX_CLUSTER_RESULTS)))
    return jsonify(
        {
            "query": query,
            "query_embedding": query_embedding.embedding,
            "distance": distance_threshold,
            "papers": papers,
            "count": len(papers),
        }
    )


@app.route("/api/topic-evolution", methods=["POST"])
@api_error_handler(error_message="Could not compute topic evolution. Please try again.")
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

    topic_keywords = coerce_text(data["topic_keywords"], "topic_keywords", _MAX_QUERY_LENGTH)
    conferences = coerce_str_list(data.get("conferences"), "conferences", max_items=_MAX_FILTER_ITEMS) or None
    distance_threshold = coerce_number(
        data.get("distance_threshold"),
        "distance_threshold",
        default=1.1,
        minimum=0.0,
        maximum=_MAX_DISTANCE,
    )

    from abstracts_explorer.mcp.errors import ToolError
    from abstracts_explorer.mcp_server import get_topic_evolution as mcp_get_topic_evolution

    enforce_client_rate_limit(_client_id())
    try:
        with profile_span("mcp.get_topic_evolution", "tool", conferences=conferences):
            result_json = mcp_get_topic_evolution(
                topic_keywords=topic_keywords,
                conferences=conferences,
                distance_threshold=distance_threshold,
            )
    except ToolError as exc:
        return jsonify({"error": str(exc)})

    result = json.loads(result_json)
    return jsonify(result)


@app.route("/api/papers-per-year")
@api_error_handler(error_message="Could not load paper counts. Please try again.")
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
    database = services.get_database()

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
    sort_order = coerce_text(data.get("sort_order"), "sort_order", 50, default="search-rating-poster")
    if sort_order not in _EXPORT_SORT_ORDERS:
        raise ApiError("Invalid sort_order", 400)

    if not paper_ids:
        return jsonify({"error": "No paper IDs provided"}), 400

    if not isinstance(priorities, dict):
        return jsonify({"error": "Invalid priorities format. Expected a dict keyed by paper UID."}), 400
    if len(priorities) > _MAX_PRIORITY_ENTRIES:
        raise ApiError(f"priorities must contain at most {_MAX_PRIORITY_ENTRIES} entries", 400)

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
    database = services.get_database()
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
            paper["searchTerm"] = coerce_text(
                priority_data.get("searchTerm", "Unknown"),
                "searchTerm",
                _MAX_SEARCH_TERM_LENGTH,
                default="Unknown",
            )

            papers.append(paper)
        except ApiError:
            raise
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
    if not isinstance(paper_priorities, dict):
        raise ApiError("paperPriorities must be an object", 400)
    if len(paper_priorities) > _MAX_PRIORITY_ENTRIES:
        raise ApiError(f"paperPriorities must contain at most {_MAX_PRIORITY_ENTRIES} entries", 400)
    if any(not isinstance(value, dict) for value in paper_priorities.values()):
        raise ApiError("Invalid data format. Expected dict with priority and searchTerm", 400)

    database = services.get_database()

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
    if not isinstance(transcript, list):
        raise ApiError("transcript must be a list", 400)
    if len(transcript) > _MAX_TRANSCRIPT_MESSAGES:
        raise ApiError(f"transcript must contain at most {_MAX_TRANSCRIPT_MESSAGES} messages", 400)
    total_chars = 0
    for message in transcript:
        if not isinstance(message, dict):
            raise ApiError("transcript entries must be objects", 400)
        text = message.get("text", "")
        total_chars += len(str(text))
    if total_chars > _MAX_TRANSCRIPT_CHARS:
        raise ApiError(f"transcript must contain at most {_MAX_TRANSCRIPT_CHARS} characters", 400)

    database = services.get_database()

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


@app.errorhandler(413)
def payload_too_large(e):
    """Handle requests larger than MAX_CONTENT_LENGTH."""
    return jsonify({"error": "Request body too large"}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def _port_conflict_message(host: str, port: int) -> str:
    """Return an actionable message for an already occupied web UI port."""
    return (
        f"Port {port} on {host} is already in use. On macOS, port 5000 is often "
        "occupied by AirPlay Receiver, which answers with HTTP 403 Forbidden. "
        "Choose another port with --port or disable AirPlay Receiver in "
        "System Settings > General > AirDrop & Handoff."
    )


def _port_in_use(host: str, port: int) -> bool:
    """
    Return True when a TCP listener already accepts connections on host:port.

    Parameters
    ----------
    host : str
        Host the web UI would bind to.
    port : int
        Port the web UI would bind to.

    Returns
    -------
    bool
        ``True`` if something is already listening, otherwise ``False``.
    """
    probe_host = "127.0.0.1" if host in ("", "0.0.0.0", "::", "localhost") else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def run_server(host="127.0.0.1", port=5001, debug=False, dev=False, threads=None):
    """
    Run the Flask web server.

    Parameters
    ----------
    host : str
        Host to bind to (default: 127.0.0.1)
    port : int
        Port to bind to (default: 5001)
    debug : bool
        Enable debug mode (default: False)
    dev : bool
        Use Flask development server instead of production server (default: False)
    threads : int, optional
        Number of worker threads for Waitress. Defaults to ``WEB_UI_THREADS``
        from the configuration (built-in default: 6). Must be >= 1.

    Raises
    ------
    ValueError
        If threads is less than 1 or the port is already occupied.
    FileNotFoundError
        If the database file does not exist.
    """
    config = services.get_config()  # Get config lazily

    if threads is None:
        threads = config.web_ui_threads

    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

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
            print("  abstracts-explorer data download --conference neurips --year 2025", file=sys.stderr)
            print("\n  # Or use a different conference/year:", file=sys.stderr)
            print("  abstracts-explorer data download --conference iclr --year 2025", file=sys.stderr)
            print("\n  # List available plugins:", file=sys.stderr)
            print("  abstracts-explorer data download --list-plugins", file=sys.stderr)
            print("\nAfter downloading papers, you may also want to create embeddings:", file=sys.stderr)
            print("  abstracts-explorer embeddings create", file=sys.stderr)
            raise FileNotFoundError(f"Database not found: {db_path}")
    # For PostgreSQL, we can't check file existence - connection will be validated at runtime

    init_profiling(app)

    # Prime the database connection, schema cache, and metadata queries so the
    # first page load does not pay the cold-start cost.
    warm_up()

    print("Starting Abstracts Explorer Web Interface...")
    print(f"Database: {config.masked_database_url}")

    # Print embeddings configuration
    print(f"Embeddings: {config.embedding_db}")

    if config.read_only:
        print("Mode: read-only (database and embedding writes are disabled)")

    if _port_in_use(host, port):
        print(f"\n⚠️  {_port_conflict_message(host, port)}", file=sys.stderr)

    print(f"Server: http://{host}:{port}")

    # Register SIGTERM handler for graceful shutdown (e.g., docker stop).
    # Without this, PID 1 in a container ignores SIGTERM by default, forcing Docker
    # to wait the full stop-timeout before sending SIGKILL.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    services.start_chat_history_cleanup()

    try:
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
    except OSError as exc:
        raise ValueError(_port_conflict_message(host, port)) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Abstracts Explorer Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging and Flask debug features; does not change server type)",
    )
    parser.add_argument(
        "--dev", action="store_true", help="Use Flask development server instead of production server"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of Waitress worker threads (default: WEB_UI_THREADS or 6)",
    )

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug, dev=args.dev, threads=args.threads)
