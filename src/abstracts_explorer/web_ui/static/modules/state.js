/**
 * Application State Management
 * 
 * This module manages the global state of the application including
 * current tab, paper priorities, and sort preferences.
 */

// State variables
export let currentTab = 'search';
export let chatHistory = [];
export let paperPriorities = {}; // Store paper priorities: { paperId: { priority: number, searchTerm: string } }
export let currentSearchTerm = ''; // Track the current search term
export let currentInterestingSession = null; // Track the current selected session in interesting papers tab
export let interestingPapersSortOrder = 'search-rating-poster'; // Sort order

/**
 * Set the current tab
 * @param {string} tab - Tab name
 */
export function setCurrentTab(tab) {
    currentTab = tab;
}

/**
 * Get the current tab
 * @returns {string} Current tab name
 */
export function getCurrentTab() {
    return currentTab;
}

/**
 * Set the current search term
 * @param {string} term - Search term
 */
export function setCurrentSearchTerm(term) {
    currentSearchTerm = term;
}

/**
 * Get the current search term
 * @returns {string} Current search term
 */
export function getCurrentSearchTerm() {
    return currentSearchTerm;
}

/**
 * Set the current interesting session
 * @param {string} session - Session name
 */
export function setCurrentInterestingSession(session) {
    currentInterestingSession = session;
}

/**
 * Get the current interesting session
 * @returns {string|null} Current session or null
 */
export function getCurrentInterestingSession() {
    return currentInterestingSession;
}

/**
 * Set the interesting papers sort order
 * @param {string} order - Sort order
 */
export function setInterestingPapersSortOrder(order) {
    interestingPapersSortOrder = order;
    localStorage.setItem('interestingPapersSortOrder', order);
}

/**
 * Get the interesting papers sort order
 * @returns {string} Sort order
 */
export function getInterestingPapersSortOrder() {
    return interestingPapersSortOrder;
}

/**
 * Load priorities from localStorage
 */
export function loadPriorities() {
    const stored = localStorage.getItem('paperPriorities');
    if (stored) {
        try {
            paperPriorities = JSON.parse(stored);
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
 * Save priorities to localStorage
 */
export function savePriorities() {
    try {
        localStorage.setItem('paperPriorities', JSON.stringify(paperPriorities));
    } catch (e) {
        console.error('Error saving priorities:', e);
    }
}

/**
 * Set paper priority
 * @param {string} paperId - Paper ID
 * @param {number} priority - Priority (0-5, 0 = remove)
 * @returns {boolean} True if priority was set/updated, false if removed
 */
export function setPaperPriority(paperId, priority) {
    const currentPriority = paperPriorities[paperId]?.priority || 0;
    const existingSearchTerm = paperPriorities[paperId]?.searchTerm;

    // If clicking the same star, remove the rating
    if (currentPriority === priority) {
        delete paperPriorities[paperId];
        savePriorities();
        return false;
    } else if (priority === 0) {
        // Remove priority if set to 0
        delete paperPriorities[paperId];
        savePriorities();
        return false;
    } else {
        paperPriorities[paperId] = {
            priority: priority,
            // Preserve the existing search term when updating, only use currentSearchTerm for new ratings
            searchTerm: existingSearchTerm || currentSearchTerm || 'Unknown'
        };
        savePriorities();
        return true;
    }
}

/**
 * Get paper priority
 * @param {string} paperId - Paper ID
 * @returns {number} Priority (0 if not set)
 */
export function getPaperPriority(paperId) {
    return paperPriorities[paperId]?.priority || 0;
}

/**
 * Get all paper priorities
 * @returns {Object} All paper priorities
 */
export function getAllPaperPriorities() {
    return paperPriorities;
}

/**
 * Get paper IDs with priorities
 * @returns {Array<string>} Array of paper IDs
 */
export function getPaperIdsWithPriorities() {
    return Object.keys(paperPriorities);
}

/**
 * Update search term for a paper
 * @param {string} paperId - Paper ID
 * @param {string} searchTerm - New search term
 */
export function updatePaperSearchTerm(paperId, searchTerm) {
    if (paperPriorities[paperId]) {
        paperPriorities[paperId].searchTerm = searchTerm;
        savePriorities();
    }
}

/**
 * Update search term for all papers with a specific old search term
 * @param {string} oldSearchTerm - Old search term
 * @param {string} newSearchTerm - New search term
 * @returns {number} Number of papers updated
 */
export function updateSearchTermForMultiplePapers(oldSearchTerm, newSearchTerm) {
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
