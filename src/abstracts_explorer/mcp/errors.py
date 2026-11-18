"""
MCP tool errors.
"""

from mcp.server.mcpserver.exceptions import ToolError

__all__ = ["ClusterAnalysisError", "ToolError"]


class ClusterAnalysisError(Exception):
    """Exception raised for cluster analysis errors."""

    pass
