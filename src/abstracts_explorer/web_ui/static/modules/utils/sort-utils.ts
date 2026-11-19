/**
 * Sorting Utility Functions
 *
 * This module provides utility functions for sorting data.
 */

/**
 * Natural sort comparison for poster positions.
 * Handles strings like "Board 123" or "123" correctly (99 < 100 < 1000).
 */
export function naturalSortPosterPosition(
    a: string | null | undefined,
    b: string | null | undefined
): number {
    const aPos = a || '';
    const bPos = b || '';

    const aMatch = aPos.match(/\d+/);
    const bMatch = bPos.match(/\d+/);

    if (aMatch && bMatch) {
        const aNum = parseInt(aMatch[0], 10);
        const bNum = parseInt(bMatch[0], 10);

        if (aNum !== bNum) {
            return aNum - bNum;
        }
    }

    return aPos.localeCompare(bPos);
}

/**
 * Sort clusters by size (descending), then by ID (ascending) as tiebreaker.
 */
export function sortClustersBySizeDesc(
    clusterEntries: Array<[string, Array<unknown> | number]>
): Array<[string, Array<unknown> | number]> {
    return clusterEntries.sort((a, b) => {
        const sizeA = Array.isArray(a[1]) ? a[1].length : a[1];
        const sizeB = Array.isArray(b[1]) ? b[1].length : b[1];

        if (sizeB !== sizeA) {
            return sizeB - sizeA;
        }
        return parseInt(a[0], 10) - parseInt(b[0], 10);
    });
}
