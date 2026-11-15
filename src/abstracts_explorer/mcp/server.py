"""
MCP server object and runner.
"""

import logging

from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = MCPServer("Abstracts Explorer Cluster Analysis")


def run_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "streamable-http",
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
        Transport method: 'streamable-http' or 'stdio' (default: 'streamable-http')

    Examples
    --------
    >>> run_mcp_server(host="0.0.0.0", port=8000)
    """
    logger.info(f"Starting MCP server on {host}:{port} with {transport} transport")

    if transport == "stdio":
        # Run with stdio transport (for local CLI integration)
        mcp.run(transport="stdio")
    else:
        # Run with streamable HTTP transport (for HTTP integration)
        mcp.run(transport="streamable-http", host=host, port=port)
