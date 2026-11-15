"""
``web-ui`` and ``mcp-server`` commands

"""

import argparse
import logging
import sys


from abstracts_explorer.mcp_server import run_mcp_server


from abstracts_explorer.cli._common import (
    _print_header,
    _unexpected_error,
)

logger = logging.getLogger(__name__)


def web_ui_command(args: argparse.Namespace) -> int:
    """
    Start the web UI server.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - host: Host to bind to
        - port: Port to bind to
        - verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        - dev: Use Flask development server
        - threads: Number of Waitress worker threads

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Try to import Flask - if it fails, give helpful error
        try:
            from abstracts_explorer.web_ui import run_server
        except ImportError:
            print("\n❌ Web UI dependencies not installed!", file=sys.stderr)
            print("\nThe web UI requires Waitress and Flask. Install them with:", file=sys.stderr)
            print("  uv sync --extra web", file=sys.stderr)
            print("\nOr install Flask manually:", file=sys.stderr)
            print("  pip install flask flask-cors", file=sys.stderr)
            return 1

        # Determine debug mode from verbosity level (2+ = DEBUG)
        debug = getattr(args, "verbose", 0) >= 2

        # Start the server (dev defaults to False for production server)
        run_server(
            host=args.host,
            port=args.port,
            debug=debug,
            dev=getattr(args, "dev", False),
            threads=getattr(args, "threads", 6),
        )
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except ValueError as e:
        print(f"\n❌ Invalid configuration: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        # Database not found - error message already printed by run_server
        return 1
    except Exception as e:
        return _unexpected_error(e, label="Error starting web server")


def mcp_server_command(args: argparse.Namespace) -> int:
    """
    Start the MCP server for cluster analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - host: Host to bind to
        - port: Port to bind to
        - transport: Transport method (streamable-http or stdio)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        _print_header(
            "MCP Server",
            f"Host: {args.host}",
            f"Port: {args.port}",
            f"Transport: {args.transport}",
        )
        print("\nStarting MCP server...")
        print("\nAvailable tools:")
        print("  - get_conference_topics: Get the main research topics of a conference")
        print("  - get_topic_evolution: Analyze how topics evolved over years")
        print("  - search_papers: Find abstracts about topics")
        print("  - get_cluster_visualization: Generate cluster visualization data")
        print("\nPress Ctrl+C to stop the server\n")

        # Start the MCP server
        run_mcp_server(host=args.host, port=args.port, transport=args.transport)
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except Exception as e:
        return _unexpected_error(e, label="Error starting MCP server")
