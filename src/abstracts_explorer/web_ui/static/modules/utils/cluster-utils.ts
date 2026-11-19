/**
 * Clustering Utility Functions
 *
 * This module provides utility functions for clustering visualization.
 */

/**
 * Get cluster label with paper count.
 *
 * Returns formatted string like "Cluster Name (N)" or "Cluster N (N)".
 */
export function getClusterLabelWithCount(
    clusterId: string,
    labels: Record<string, string>,
    paperCount: number
): string {
    const label = labels[clusterId] || `Cluster ${clusterId}`;
    return `${label} (${paperCount})`;
}
