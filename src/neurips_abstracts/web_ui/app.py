"""
Flask web application for NeurIPS Abstracts.

Provides a web interface for searching papers, chatting with RAG,
and exploring the NeurIPS abstracts database.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS

from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.embeddings import EmbeddingsManager
from neurips_abstracts.rag import RAGChat
from neurips_abstracts.config import get_config

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
    if rag_chat is None:
        config = get_config()  # Get config lazily
        em = get_embeddings_manager()
        rag_chat = RAGChat(
            embeddings_manager=em,
            database=None,  # Will be set per-request
            lm_studio_url=config.llm_backend_url,
            model=config.chat_model,
        )
    # Update database reference for this request
    rag_chat.database = get_database()
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
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """
    Search for papers.

    Expected JSON body:
    - query: str - Search query
    - use_embeddings: bool - Use semantic search
    - limit: int - Maximum results (default: 10)

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

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if use_embeddings:
            # Semantic search using embeddings
            em = get_embeddings_manager()
            results = em.search_similar(query, n_results=limit)

            # Transform ChromaDB results to paper format
            papers = []
            if results.get("ids") and results["ids"][0]:
                database = get_database()
                for i in range(len(results["ids"][0])):
                    paper_id = results["ids"][0][i]

                    # Get full paper details from database
                    paper_rows = database.query("SELECT * FROM papers WHERE id = ?", (paper_id,))
                    if paper_rows:
                        paper_dict = dict(paper_rows[0])
                        # Add similarity score
                        if "distances" in results and results["distances"][0]:
                            distance = results["distances"][0][i]
                            paper_dict["similarity"] = 1 - distance if distance <= 1 else 0

                        # Get authors from authors table
                        authors = database.get_paper_authors(paper_id)
                        paper_dict["authors"] = [a["fullname"] for a in authors]

                        papers.append(paper_dict)
        else:
            # Keyword search in database
            database = get_database()
            papers = database.search_papers(keyword=query, limit=limit)

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
        papers = database.query("SELECT * FROM papers WHERE id = ?", (paper_id,))

        if not papers:
            return jsonify({"error": "Paper not found"}), 404

        paper = dict(papers[0])

        # Get authors from authors table
        authors = database.get_paper_authors(paper_id)
        paper["authors"] = [a["fullname"] for a in authors]

        return jsonify(paper)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat with RAG system.

    Expected JSON body:
    - message: str - User message
    - n_papers: int (optional) - Number of papers for context
    - reset: bool (optional) - Reset conversation

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

        if not message:
            return jsonify({"error": "Message is required"}), 400

        rag = get_rag_chat()

        if reset:
            rag.reset_conversation()

        # Get response
        response = rag.query(message, n_results=n_papers)

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
    print(f"Starting NeurIPS Abstracts Web Interface...")
    print(f"Database: {config.paper_db_path}")
    print(f"Embeddings: {config.embedding_db_path}")
    print(f"Server: http://{host}:{port}")
    print(f"\nPress CTRL+C to stop the server")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeurIPS Abstracts Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
