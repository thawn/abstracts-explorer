/**
 * Clustering Utility Functions
 * 
 * This module provides utility functions for clustering visualization.
 */

/**
 * Get cluster label with paper count
 * Returns formatted string like "Cluster Name (N)" or "Cluster N (N)"
 * @param {string} clusterId - Cluster ID
 * @param {Object} labels - Map of cluster IDs to labels
 * @param {number} paperCount - Number of papers in cluster
 * @returns {string} Formatted cluster label
 */
export function getClusterLabelWithCount(clusterId, labels, paperCount) {
    const label = labels[clusterId] || `Cluster ${clusterId}`;
    return `${label} (${paperCount})`;
}
