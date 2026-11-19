/**
 * Application State Management
 *
 * This module manages the global state of the application including
 * current tab, paper priorities, and sort preferences.
 */

import type { PaperPriority } from './types.js';

// State variables
export let currentTab = 'search';
export let chatHistory: unknown[] = [];
export let paperPriorities: Record<string, PaperPriority> = {};
export let currentSearchTerm = '';
export let currentInterestingSession: string | null = null;
export let interestingPapersSortOrder = 'search-rating-poster';

/**
 * Reset all state to initial values (for testing).
 */
export function resetState(): void {
    currentTab = 'search';
    chatHistory = [];
    paperPriorities = {};
    currentSearchTerm = '';
    currentInterestingSession = null;
    interestingPapersSortOrder = 'search-rating-poster';
}

/**
 * Set the current tab.
 */
export function setCurrentTab(tab: string): void {
    currentTab = tab;
}

/**
 * Get the current tab.
 */
export function getCurrentTab(): string {
    return currentTab;
}

/**
 * Set the current search term.
 */
export function setCurrentSearchTerm(term: string): void {
    currentSearchTerm = term;
}

/**
 * Get the current search term.
 */
export function getCurrentSearchTerm(): string {
    return currentSearchTerm;
}

/**
 * Set the current interesting session.
 */
export function setCurrentInterestingSession(session: string | null): void {
    currentInterestingSession = session;
}

/**
 * Get the current interesting session.
 */
export function getCurrentInterestingSession(): string | null {
    return currentInterestingSession;
}

/**
 * Set the interesting papers sort order.
 */
export function setInterestingPapersSortOrder(order: string): void {
    interestingPapersSortOrder = order;
    localStorage.setItem('interestingPapersSortOrder', order);
}

/**
 * Get the interesting papers sort order.
 */
export function getInterestingPapersSortOrder(): string {
    return interestingPapersSortOrder;
}

/**
 * Load priorities from localStorage.
 */
export function loadPriorities(): void {
    const stored = localStorage.getItem('paperPriorities');
    if (stored) {
        try {
            paperPriorities = JSON.parse(stored) as Record<string, PaperPriority>;
        } catch (e) {
            console.error('Error loading priorities:', e);
            paperPriorities = {};
        }
    }

    // Load sort order preference
    const storedSortOrder = localStorage.getItem('interestingPapersSortOrder');
    if (storedSortOrder) {
        interestingPapersSortOrder = storedSortOrder;
    }
}

/**
 * Save priorities to localStorage.
 */
export function savePriorities(): void {
    try {
        localStorage.setItem('paperPriorities', JSON.stringify(paperPriorities));
    } catch (e) {
        console.error('Error saving priorities:', e);
    }
}

/**
 * Set paper priority.
 *
 * @returns True if priority was set/updated, false if removed
 */
export function setPaperPriority(paperId: string, priority: number): boolean {
    const currentPriority = paperPriorities[paperId]?.priority || 0;
    const existingSearchTerm = paperPriorities[paperId]?.searchTerm;

    // If clicking the same star, remove the rating
    if (currentPriority === priority) {
        delete paperPriorities[paperId];
        savePriorities();
        return false;
    } else if (priority === 0) {
        delete paperPriorities[paperId];
        savePriorities();
        return false;
    } else {
        paperPriorities[paperId] = {
            priority: priority,
            // Preserve the existing search term when updating
            searchTerm: existingSearchTerm || currentSearchTerm || 'Unknown'
        };
        savePriorities();
        return true;
    }
}

/**
 * Get paper priority (0 if not set).
 */
export function getPaperPriority(paperId: string): number {
    return paperPriorities[paperId]?.priority || 0;
}

/**
 * Get all paper priorities.
 */
export function getAllPaperPriorities(): Record<string, PaperPriority> {
    return paperPriorities;
}

/**
 * Get paper IDs with priorities.
 */
export function getPaperIdsWithPriorities(): string[] {
    return Object.keys(paperPriorities);
}

/**
 * Update search term for a paper.
 */
export function updatePaperSearchTerm(paperId: string, searchTerm: string): void {
    if (paperPriorities[paperId]) {
        paperPriorities[paperId].searchTerm = searchTerm;
        savePriorities();
    }
}

/**
 * Update search term for all papers with a specific old search term.
 *
 * @returns Number of papers updated
 */
export function updateSearchTermForMultiplePapers(oldSearchTerm: string, newSearchTerm: string): number {
    let updatedCount = 0;
    for (const paperId in paperPriorities) {
        const paperData = paperPriorities[paperId];
        if (paperData.searchTerm === oldSearchTerm) {
            paperData.searchTerm = newSearchTerm;
            updatedCount++;
        }
    }
    if (updatedCount > 0) {
        savePriorities();
    }
    return updatedCount;
}
