"""
Flask web application for Abstracts Explorer.

Provides a web interface for searching papers, chatting with RAG,
and exploring the abstracts database.
"""

import os
import sys
import logging
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, g, send_file
from flask_cors import CORS

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat
from abstracts_explorer.config import get_config
from abstracts_explorer.paper_utils import get_paper_with_authors, PaperFormattingError
from abstracts_explorer.export_utils import export_papers_to_zip
from abstracts_explorer.clustering import compute_clusters_with_cache, ClusteringError, calculate_default_clusters
from abstracts_explorer.plugin import get_available_filters

# Import version
try:
    from abstracts_explorer._version import __version__
except ImportError:
    # Fallback if _version.py doesn't exist (e.g., editable install without build)
    from abstracts_explorer import __version__

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


@app.route("/")
def index():
    """
    Render the main page.

    Returns
    -------
    str
        Rendered HTML template
    """
    return render_template("index.html", version=__version__)


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
        
        return jsonify({
            "compatible": compatible,
            "stored_model": stored_model,
            "current_model": current_model,
            "warning": warning
        })
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
def get_available_filters_endpoint():
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
        filters = get_available_filters()
        return jsonify(filters)
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
            
            papers = em.search_papers_semantic(
                query=query,
                database=database,
                limit=limit,
                sessions=sessions,
                years=years,
                conferences=conferences,
            )
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


@app.route("/api/clusters/compute", methods=["POST"])
def compute_clusters():
    """
    Compute clusters on demand with specified parameters.

    Checks cache first, computes only if needed. Invalidates cache if
    embedding model has changed.

    Request Body
    ------------
    {
        "reduction_method": str (optional, default: "pca"),
        "n_components": int (optional, default: 2),
        "clustering_method": str (optional, default: "kmeans"),
        "n_clusters": int (optional, default: None - auto-calculated),
        "eps": float (optional, default: 0.5, for DBSCAN),
        "min_samples": int (optional, default: 5, for DBSCAN),
        "distance_threshold": float (optional, for agglomerative),
        "linkage": str (optional, for agglomerative, default: "ward"),
        "affinity": str (optional, for agglomerative/spectral),
        "m": float (optional, for fuzzy c-means, default: 2.0),
        "n_neighbors": int (optional, for spectral with nearest_neighbors affinity),
        "limit": int (optional, max embeddings to process),
        "force": bool (optional, default: False, force recompute)
    }

    Returns
    -------
    dict
        Clustering results with points, statistics, and metadata
    """
    try:
        data = request.get_json() or {}

        # Get parameters
        reduction_method = data.get("reduction_method", "pca")
        n_components = data.get("n_components", 2)
        clustering_method = data.get("clustering_method", "kmeans")
        n_clusters = data.get("n_clusters")  # None means auto-calculate
        limit = data.get("limit")
        force = data.get("force", False)

        # Get config and database
        config = get_config()
        database = get_database()
        em = get_embeddings_manager()

        # Get current embedding model
        current_model = config.embedding_model
        
        # Build clustering kwargs for different methods
        clustering_kwargs = {}
        method_lower = clustering_method.lower()
        
        if method_lower == "dbscan":
            clustering_kwargs["eps"] = data.get("eps", 0.5)
            clustering_kwargs["min_samples"] = data.get("min_samples", 5)
        elif method_lower == "agglomerative":
            if "distance_threshold" in data:
                clustering_kwargs["distance_threshold"] = data.get("distance_threshold")
            if "linkage" in data:
                clustering_kwargs["linkage"] = data.get("linkage")
            if "affinity" in data:
                clustering_kwargs["affinity"] = data.get("affinity")
        elif method_lower in ["fuzzy_cmeans", "fuzzy-cmeans"]:
            if "m" in data:
                clustering_kwargs["m"] = data.get("m", 2.0)
            if "maxiter" in data:
                clustering_kwargs["maxiter"] = data.get("maxiter")
            if "error" in data:
                clustering_kwargs["error"] = data.get("error")
        elif method_lower == "spectral":
            if "affinity" in data:
                clustering_kwargs["affinity"] = data.get("affinity")
            if "n_neighbors" in data:
                clustering_kwargs["n_neighbors"] = data.get("n_neighbors")

        # Use shared clustering function
        results = compute_clusters_with_cache(
            embeddings_manager=em,
            database=database,
            embedding_model=current_model,
            reduction_method=reduction_method,
            n_components=n_components,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            limit=limit,
            force=force,
            **clustering_kwargs
        )

        return jsonify(results)

    except ClusteringError as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error computing clusters: {e}", exc_info=True)
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
            return jsonify({
                "error": f"Cached clusters file not found: {cache_file}",
                "hint": "Run 'abstracts-explorer cluster-embeddings --output clusters.json' first"
            }), 404

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
        
        return jsonify({
            "n_clusters": n_clusters,
            "n_papers": n_papers
        })
    except Exception as e:
        logger.error(f"Error calculating default cluster count: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters/precalculate", methods=["POST"])
def precalculate_clusters():
    """
    Pre-calculate clusters in the background for caching.
    
    This endpoint starts a background clustering computation with default settings
    to populate the cache. It returns immediately without waiting for completion.
    
    Request Body
    ------------
    {
        "reduction_method": str (optional, default: "pca"),
        "n_components": int (optional, default: 2),
        "clustering_method": str (optional, default: "kmeans"),
        "n_clusters": int (optional, default: None - auto-calculated)
    }
    
    Returns
    -------
    dict
        Status message indicating the pre-calculation was started
    """
    try:
        import threading
        
        data = request.get_json() or {}
        
        # Get parameters with defaults
        reduction_method = data.get("reduction_method", "pca")
        n_components = data.get("n_components", 2)
        clustering_method = data.get("clustering_method", "kmeans")
        n_clusters = data.get("n_clusters")
        
        # Get config and managers
        config = get_config()
        em = get_embeddings_manager()
        database = get_database()
        
        # Calculate default n_clusters if not provided
        if n_clusters is None:
            collection_stats = em.get_collection_stats()
            n_papers = collection_stats["count"]
            n_clusters = calculate_default_clusters(n_papers)
        
        # Check if cache already exists
        current_model = config.embedding_model
        cached_results = database.get_clustering_cache(
            embedding_model=current_model,
            reduction_method=reduction_method,
            n_components=n_components,
            clustering_method=clustering_method,
            n_clusters=n_clusters if clustering_method.lower() != "dbscan" else None,
        )
        
        if cached_results:
            logger.info("Clustering cache already exists, skipping pre-calculation")
            return jsonify({
                "status": "cache_exists",
                "message": "Clustering cache already exists"
            })
        
        # Define background task
        def background_clustering():
            try:
                logger.info(f"Starting background clustering pre-calculation (n_clusters={n_clusters})")
                
                # Use shared clustering function
                compute_clusters_with_cache(
                    embeddings_manager=em,
                    database=database,
                    embedding_model=current_model,
                    reduction_method=reduction_method,
                    n_components=n_components,
                    clustering_method=clustering_method,
                    n_clusters=n_clusters,
                    limit=None,
                    force=False,
                )
                
                logger.info("Background clustering pre-calculation completed successfully")
                
            except Exception as e:
                logger.error(f"Error in background clustering: {e}", exc_info=True)
        
        # Start background thread
        thread = threading.Thread(target=background_clustering, daemon=True)
        thread.start()
        
        logger.info(f"Started background clustering pre-calculation with n_clusters={n_clusters}")
        
        return jsonify({
            "status": "started",
            "message": "Clustering pre-calculation started in background",
            "n_clusters": n_clusters
        })
        
    except Exception as e:
        logger.error(f"Error starting clustering pre-calculation: {e}", exc_info=True)
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
            years=years
        )
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error searching custom cluster: {e}", exc_info=True)
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


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def run_server(host="127.0.0.1", port=5000, debug=False, dev=False):
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
    
    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
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
            print("\nPress CTRL+C to stop the server")
            # Set Flask debug mode even with Waitress for better error messages
            app.debug = debug
            serve(app, host=host, port=port)
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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (uses Flask dev server)")
    parser.add_argument("--dev", action="store_true", help="Use Flask development server instead of production server")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug, dev=args.dev)
