"""
MCP Server for Cluster Analysis
================================

This module provides a Model Context Protocol (MCP) server that exposes
tools for analyzing clustered embeddings. The server enables LLM-based
assistants to answer questions about conference paper topics, trends,
and developments.

Features:
- Get most frequently mentioned topics from clusters
- Analyze topic evolution over years
- Find recent developments in specific topics
- Generate cluster visualizations
"""

import logging
import json
from typing import Any, Dict, Optional
from collections import defaultdict, Counter

from mcp.server.fastmcp import FastMCP

from .embeddings import EmbeddingsManager
from .database import DatabaseManager
from .clustering import ClusteringManager, perform_clustering
from .config import get_config

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Abstracts Explorer Cluster Analysis")


class ClusterAnalysisError(Exception):
    """Exception raised for cluster analysis errors."""
    pass


def load_clustering_data(
    embeddings_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> tuple[ClusteringManager, DatabaseManager]:
    """
    Load clustering data and database.

    Parameters
    ----------
    embeddings_path : str, optional
        Path to ChromaDB embeddings database
    collection_name : str, optional
        Name of the ChromaDB collection
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    tuple[ClusteringManager, DatabaseManager]
        Clustering manager and database manager instances

    Raises
    ------
    ClusterAnalysisError
        If loading fails
    """
    config = get_config()
    
    # Use config defaults if not provided
    embeddings_path = embeddings_path or config.embedding_db_path
    collection_name = collection_name or config.collection_name
    db_path = db_path or config.paper_db_path
    
    try:
        # Initialize embeddings manager
        em = EmbeddingsManager(
            chroma_path=embeddings_path,
            collection_name=collection_name,
        )
        em.connect()
        em.create_collection()
        
        # Initialize database manager
        db = DatabaseManager(db_path)
        db.connect()
        
        # Initialize clustering manager
        cm = ClusteringManager(em, db)
        
        return cm, db
        
    except Exception as e:
        raise ClusterAnalysisError(f"Failed to load clustering data: {str(e)}") from e


def analyze_cluster_topics(
    cm: ClusteringManager,
    db: DatabaseManager,
    cluster_id: int,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Analyze topics in a specific cluster.

    Parameters
    ----------
    cm : ClusteringManager
        Clustering manager with loaded data
    db : DatabaseManager
        Database manager for paper metadata
    cluster_id : int
        Cluster ID to analyze
    use_llm : bool, optional
        Whether to use LLM for topic extraction (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - cluster_id: Cluster ID
        - paper_count: Number of papers in cluster
        - keywords: Most common keywords
        - sessions: Most common sessions
        - years: Distribution by year
        - sample_titles: Sample paper titles
    """
    if cm.cluster_labels is None or cm.paper_ids is None or cm.metadatas is None:
        raise ClusterAnalysisError("Clustering data not loaded. Call load_embeddings() and cluster() first.")
    
    # Find papers in this cluster
    cluster_indices = [i for i, label in enumerate(cm.cluster_labels) if label == cluster_id]
    
    if not cluster_indices:
        return {
            "cluster_id": cluster_id,
            "paper_count": 0,
            "keywords": [],
            "sessions": [],
            "years": {},
            "sample_titles": [],
        }
    
    # Extract metadata for papers in this cluster
    keywords = []
    sessions = []
    years = []
    sample_titles: list[str] = []
    
    for idx in cluster_indices:
        metadata = cm.metadatas[idx]
        
        # Collect keywords
        if metadata.get("keywords"):
            keywords.extend(metadata.get("keywords", "").split(","))
        
        # Collect sessions
        if metadata.get("session"):
            sessions.append(metadata.get("session"))
        
        # Collect years
        if metadata.get("year"):
            years.append(metadata.get("year"))
        
        # Collect sample titles (first 5)
        if len(sample_titles) < 5:
            sample_titles.append(metadata.get("title", ""))
    
    # Count frequencies
    keyword_counts = Counter([k.strip().lower() for k in keywords if k.strip()])
    session_counts = Counter(sessions)
    year_counts = Counter(years)
    
    return {
        "cluster_id": cluster_id,
        "paper_count": len(cluster_indices),
        "keywords": [{"keyword": k, "count": c} for k, c in keyword_counts.most_common(10)],
        "sessions": [{"session": s, "count": c} for s, c in session_counts.most_common(5)],
        "years": dict(sorted(year_counts.items())),
        "sample_titles": sample_titles,
    }


@mcp.tool()
def get_cluster_topics(
    n_clusters: int = 8,
    reduction_method: str = "pca",
    clustering_method: str = "kmeans",
    embeddings_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """
    Get the most frequently mentioned topics from clustered embeddings.

    This tool clusters paper embeddings and analyzes the topics in each cluster
    based on keywords, sessions, and paper titles.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to create (default: 8)
    reduction_method : str, optional
        Dimensionality reduction method: 'pca' or 'tsne' (default: 'pca')
    clustering_method : str, optional
        Clustering method: 'kmeans', 'dbscan', or 'agglomerative' (default: 'kmeans')
    embeddings_path : str, optional
        Path to ChromaDB embeddings database (uses config default if not provided)
    collection_name : str, optional
        Name of ChromaDB collection (uses config default if not provided)
    db_path : str, optional
        Path to SQLite database (uses config default if not provided)

    Returns
    -------
    str
        JSON string containing cluster topics analysis
    """
    try:
        config = get_config()
        embeddings_path = embeddings_path or config.embedding_db_path
        collection_name = collection_name or config.collection_name
        db_path = db_path or config.paper_db_path
        
        # Load clustering data
        cm, db = load_clustering_data(embeddings_path, collection_name, db_path)
        
        # Load embeddings
        logger.info("Loading embeddings...")
        cm.load_embeddings()
        
        # Perform clustering on full embeddings
        logger.info(f"Clustering using {clustering_method}...")
        cm.cluster(
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=42,
            use_reduced=False,
        )
        
        # Reduce dimensions for visualization (needed for some methods)
        logger.info(f"Reducing dimensions using {reduction_method}...")
        cm.reduce_dimensions(
            method=reduction_method,
            n_components=2,
            random_state=42,
        )
        
        # Get cluster statistics
        stats = cm.get_cluster_statistics()
        
        # Analyze topics for each cluster
        cluster_topics = []
        for cluster_id in range(stats["n_clusters"]):
            topics = analyze_cluster_topics(cm, db, cluster_id)
            cluster_topics.append(topics)
        
        result = {
            "statistics": stats,
            "clusters": cluster_topics,
        }
        
        # Clean up
        cm.embeddings_manager.close()
        db.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to get cluster topics: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


def merge_where_clause_with_conference(
    where: Optional[Dict[str, Any]],
    conference: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Merge a WHERE clause with a conference filter.

    This helper function properly combines custom WHERE clauses with conference
    filters, avoiding duplicates and handling nested operators correctly.

    Parameters
    ----------
    where : dict, optional
        Custom WHERE clause from user
    conference : str, optional
        Conference name to filter by

    Returns
    -------
    dict or None
        Merged WHERE clause, or None if both inputs are None

    Raises
    ------
    ValueError
        If WHERE clause is not a dict
    """
    # Validate where parameter
    if where is not None and not isinstance(where, dict):
        raise ValueError(f"WHERE clause must be a dict, got {type(where).__name__}")
    
    # If no conference, just return the WHERE clause (or None)
    if not conference:
        return where.copy() if where else None
    
    # If no WHERE clause, just return conference filter
    if not where:
        return {"conference": conference}
    
    # Check if conference already exists anywhere in WHERE clause
    def has_conference_filter(obj: Any) -> bool:
        """Recursively check if conference filter exists in nested structure."""
        if isinstance(obj, dict):
            if "conference" in obj:
                return True
            # Check nested values
            for value in obj.values():
                if has_conference_filter(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if has_conference_filter(item):
                    return True
        return False
    
    # If conference already in WHERE clause, don't add again
    if has_conference_filter(where):
        return where.copy()
    
    # Need to merge conference with WHERE clause
    where_filter = where.copy()
    
    # If WHERE already has $and, append to it
    if "$and" in where_filter:
        where_filter["$and"].append({"conference": conference})
    else:
        # Create new $and with existing filter and conference
        where_filter = {"$and": [where_filter, {"conference": conference}]}
    
    return where_filter


@mcp.tool()
def get_topic_evolution(
    topic_keywords: str,
    conference: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    embeddings_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """
    Analyze how topics have evolved over the years for a conference.

    This tool searches for papers related to specific topic keywords and
    analyzes their distribution and trends over time.

    Parameters
    ----------
    topic_keywords : str
        Keywords describing the topic to analyze (e.g., "transformers attention")
    conference : str, optional
        Filter by conference name (e.g., "neurips", "iclr")
    start_year : int, optional
        Start year for analysis (inclusive)
    end_year : int, optional
        End year for analysis (inclusive)
    where : dict, optional
        Custom ChromaDB WHERE clause for filtering results by metadata.
        Supports ChromaDB query operators like $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
        Logical operators $and, $or are also supported.
        Examples:
          {"year": 2025}  # Filter by specific year
          {"session": {"$in": ["Oral Session 1", "Oral Session 2"]}}  # Multiple sessions
          {"$and": [{"year": {"$gte": 2024}}, {"conference": "NeurIPS"}]}  # Multiple conditions
        Note: If 'conference' parameter is provided, it will be merged with this WHERE clause.
    embeddings_path : str, optional
        Path to ChromaDB embeddings database
    collection_name : str, optional
        Name of ChromaDB collection
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    str
        JSON string containing topic evolution analysis
    """
    try:
        config = get_config()
        embeddings_path = embeddings_path or config.embedding_db_path
        collection_name = collection_name or config.collection_name
        db_path = db_path or config.paper_db_path
        
        # Initialize embeddings manager
        em = EmbeddingsManager(
            chroma_path=embeddings_path,
            collection_name=collection_name,
        )
        em.connect()
        em.create_collection()
        
        # Initialize database
        db = DatabaseManager(db_path)
        db.connect()
        
        # Build metadata filter using helper function
        try:
            where_filter = merge_where_clause_with_conference(where, conference)
        except ValueError as e:
            logger.error(f"Invalid WHERE clause: {str(e)}")
            return json.dumps({"error": f"Invalid WHERE clause: {str(e)}"}, indent=2)
        
        # Search for papers related to topic
        logger.info(f"Searching for papers about: {topic_keywords}")
        if where_filter:
            logger.info(f"Applying WHERE filter: {where_filter}")
        results = em.search_similar(
            query=topic_keywords,
            n_results=100,  # Get more results for trend analysis
            where=where_filter,
        )
        
        # Analyze results by year
        year_distribution = defaultdict(list)
        year_counts: Counter[Any] = Counter()
        
        if results["ids"] and results["ids"][0]:
            for idx, paper_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                year = metadata.get("year")
                
                # Filter by year range if specified
                if year:
                    if start_year and year < start_year:
                        continue
                    if end_year and year > end_year:
                        continue
                    
                    year_counts[year] += 1
                    year_distribution[year].append({
                        "title": metadata.get("title", ""),
                        "session": metadata.get("session", ""),
                        "distance": results["distances"][0][idx] if "distances" in results else None,
                    })
        
        # Sort by year
        sorted_years = sorted(year_distribution.keys())
        
        # Build result
        result = {
            "topic": topic_keywords,
            "conference": conference,
            "total_papers": len(results["ids"][0]) if results["ids"] else 0,
            "year_range": {
                "start": min(sorted_years) if sorted_years else None,
                "end": max(sorted_years) if sorted_years else None,
            },
            "year_counts": dict(sorted(year_counts.items())),
            "papers_by_year": {
                year: {
                    "count": len(papers),
                    "sample_papers": papers[:3],  # Top 3 most relevant
                }
                for year, papers in sorted(year_distribution.items())
            },
        }
        
        # Clean up
        em.close()
        db.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to get topic evolution: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def get_recent_developments(
    topic_keywords: str,
    n_years: int = 2,
    n_results: int = 10,
    conference: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
    embeddings_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """
    Find the most important recent developments in a specific topic.

    This tool searches for the most relevant papers about a topic from
    recent years.

    Parameters
    ----------
    topic_keywords : str
        Keywords describing the topic (e.g., "large language models")
    n_years : int, optional
        Number of recent years to consider (default: 2)
    n_results : int, optional
        Number of papers to return (default: 10)
    conference : str, optional
        Filter by conference name
    where : dict, optional
        Custom ChromaDB WHERE clause for filtering results by metadata.
        Supports ChromaDB query operators like $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
        Logical operators $and, $or are also supported.
        Examples:
          {"year": 2025}  # Filter by specific year
          {"session": {"$in": ["Oral Session 1", "Oral Session 2"]}}  # Multiple sessions
          {"$and": [{"year": {"$gte": 2024}}, {"conference": "NeurIPS"}]}  # Multiple conditions
        Note: If 'conference' parameter is provided, it will be merged with this WHERE clause.
    embeddings_path : str, optional
        Path to ChromaDB embeddings database
    collection_name : str, optional
        Name of ChromaDB collection
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    str
        JSON string containing recent developments
    """
    try:
        config = get_config()
        embeddings_path = embeddings_path or config.embedding_db_path
        collection_name = collection_name or config.collection_name
        db_path = db_path or config.paper_db_path
        
        # Initialize embeddings manager
        em = EmbeddingsManager(
            chroma_path=embeddings_path,
            collection_name=collection_name,
        )
        em.connect()
        em.create_collection()
        
        # Initialize database
        db = DatabaseManager(db_path)
        db.connect()
        
        # Calculate year cutoff
        from datetime import datetime
        current_year = datetime.now().year
        year_cutoff = current_year - n_years
        
        # Build metadata filter using helper function
        try:
            where_filter = merge_where_clause_with_conference(where, conference)
        except ValueError as e:
            logger.error(f"Invalid WHERE clause: {str(e)}")
            return json.dumps({"error": f"Invalid WHERE clause: {str(e)}"}, indent=2)
        
        # Search for papers
        logger.info(f"Searching for recent papers about: {topic_keywords}")
        if where_filter:
            logger.info(f"Applying WHERE filter: {where_filter}")
        results = em.search_similar(
            query=topic_keywords,
            n_results=n_results * 3,  # Get more to filter by year
            where=where_filter,
        )
        
        # Filter and format results
        recent_papers = []
        if results["ids"] and results["ids"][0]:
            for idx, paper_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                year = metadata.get("year")
                
                # Only include recent papers
                if year and year >= year_cutoff:
                    recent_papers.append({
                        "id": paper_id,
                        "title": metadata.get("title", ""),
                        "year": year,
                        "conference": metadata.get("conference", ""),
                        "session": metadata.get("session", ""),
                        "abstract": results["documents"][0][idx] if "documents" in results and results["documents"][0] else "",
                        "relevance_score": 1.0 - results["distances"][0][idx] if "distances" in results else None,
                    })
                    
                    if len(recent_papers) >= n_results:
                        break
        
        result = {
            "topic": topic_keywords,
            "conference": conference,
            "year_cutoff": year_cutoff,
            "papers_found": len(recent_papers),
            "papers": recent_papers,
        }
        
        # Clean up
        em.close()
        db.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to get recent developments: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def get_cluster_visualization(
    n_clusters: int = 8,
    reduction_method: str = "tsne",
    clustering_method: str = "kmeans",
    n_components: int = 2,
    output_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """
    Generate visualization data for clustered embeddings.

    This tool performs clustering and dimensionality reduction on paper
    embeddings and returns data suitable for visualization.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters (default: 8)
    reduction_method : str, optional
        Reduction method: 'pca' or 'tsne' (default: 'tsne')
    clustering_method : str, optional
        Clustering method: 'kmeans', 'dbscan', or 'agglomerative' (default: 'kmeans')
    n_components : int, optional
        Number of dimensions for visualization: 2 or 3 (default: 2)
    output_path : str, optional
        Path to save visualization JSON file (optional)
    embeddings_path : str, optional
        Path to ChromaDB embeddings database
    collection_name : str, optional
        Name of ChromaDB collection
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    str
        JSON string containing visualization data with points, clusters, and statistics
    """
    try:
        config = get_config()
        embeddings_path = embeddings_path or config.embedding_db_path
        collection_name = collection_name or config.collection_name
        db_path = db_path or config.paper_db_path
        
        # Perform clustering
        logger.info("Performing clustering for visualization...")
        results = perform_clustering(
            embeddings_path=embeddings_path,
            collection_name=collection_name,
            reduction_method=reduction_method,
            n_components=n_components,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            output_path=output_path,
            random_state=42,
        )
        
        # Format result
        result = {
            "n_dimensions": n_components,
            "n_points": len(results["points"]),
            "statistics": results["statistics"],
            "points": results["points"][:1000],  # Limit for MCP response size
            "visualization_saved": output_path is not None,
            "output_path": output_path if output_path else None,
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to generate cluster visualization: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


def run_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "sse",
) -> None:
    """
    Run the MCP server.

    Parameters
    ----------
    host : str, optional
        Host address to bind to (default: "127.0.0.1")
    port : int, optional
        Port to listen on (default: 8000)
    transport : str, optional
        Transport method: 'sse' or 'stdio' (default: 'sse')

    Examples
    --------
    >>> run_mcp_server(host="0.0.0.0", port=8000)
    """
    logger.info(f"Starting MCP server on {host}:{port} with {transport} transport")
    
    if transport == "stdio":
        # Run with stdio transport (for local CLI integration)
        import asyncio
        asyncio.run(mcp.run_stdio_async())
    else:
        # Run with SSE transport (for HTTP integration)
        mcp.run(host=host, port=port)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run server
    run_mcp_server()
