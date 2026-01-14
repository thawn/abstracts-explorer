/**
 * Sorting Utility Functions
 * 
 * This module provides utility functions for sorting data.
 */

/**
 * Natural sort comparison for poster positions
 * Handles strings like "Board 123" or "123" correctly (99 < 100 < 1000)
 * @param {string} a - First position string
 * @param {string} b - Second position string
 * @returns {number} Comparison result (-1, 0, or 1)
 */
export function naturalSortPosterPosition(a, b) {
    const aPos = a || '';
    const bPos = b || '';
    
    // Extract numbers from the strings
    const aMatch = aPos.match(/\d+/);
    const bMatch = bPos.match(/\d+/);
    
    if (aMatch && bMatch) {
        const aNum = parseInt(aMatch[0], 10);
        const bNum = parseInt(bMatch[0], 10);
        
        if (aNum !== bNum) {
            return aNum - bNum;
        }
    }
    
    // Fallback to string comparison if no numbers or numbers are equal
    return aPos.localeCompare(bPos);
}

/**
 * Sort clusters by size (descending), then by ID (ascending) as tiebreaker
 * @param {Array<[string, Array|number]>} clusterEntries - Array of [clusterId, items] entries
 * @returns {Array<[string, Array|number]>} Sorted cluster entries
 */
export function sortClustersBySizeDesc(clusterEntries) {
    return clusterEntries.sort((a, b) => {
        // Get size - either length of array or the value itself if it's a number
        const sizeA = Array.isArray(a[1]) ? a[1].length : a[1];
        const sizeB = Array.isArray(b[1]) ? b[1].length : b[1];
        
        if (sizeB !== sizeA) {
            return sizeB - sizeA;  // Sort by size descending
        }
        return parseInt(a[0]) - parseInt(b[0]);  // Tiebreaker: sort by ID ascending
    });
}
