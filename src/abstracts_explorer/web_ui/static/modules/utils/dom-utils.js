/**
 * DOM Utility Functions
 * 
 * This module provides utility functions for DOM manipulation and HTML rendering.
 */

/**
 * Escape HTML special characters to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML string
 */
export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Render an empty state message
 * @param {string} message - Main message to display
 * @param {string} subtext - Additional context text
 * @param {string} icon - FontAwesome icon class (default: fa-inbox)
 * @returns {string} HTML string for empty state
 */
export function renderEmptyState(message, subtext, icon = 'fa-inbox') {
    return `
        <div class="text-center text-gray-500 py-12">
            <i class="fas ${icon} text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">${message}</p>
            <p class="text-sm">${subtext}</p>
        </div>
    `;
}

/**
 * Render an error message block
 * @param {string} message - Error message to display
 * @returns {string} HTML string for error block
 */
export function renderErrorBlock(message) {
    return `
        <div class="bg-red-50 border border-red-200 rounded-lg p-6">
            <div class="flex items-center">
                <i class="fas fa-exclamation-circle text-red-500 text-2xl mr-3"></i>
                <div>
                    <h3 class="text-red-800 font-semibold">Error</h3>
                    <p class="text-red-700 text-sm mt-1">${escapeHtml(message)}</p>
                </div>
            </div>
        </div>
    `;
}

/**
 * Show loading message in a specific element
 * @param {string} elementId - ID of the element to show loading in
 * @param {string} message - Loading message to display
 */
export function showLoading(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element with id '${elementId}' not found`);
        return;
    }
    element.innerHTML = `
        <div class="text-center text-gray-500 py-12">
            <i class="fas fa-spinner fa-spin text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">${escapeHtml(message)}</p>
            <p class="text-sm mt-2">This may take a moment</p>
        </div>
    `;
}

/**
 * Show error message in a specific element
 * @param {string} elementId - ID of the element to show error in
 * @param {string} message - Error message to display
 */
export function showErrorInElement(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element with id '${elementId}' not found`);
        return;
    }
    element.innerHTML = renderErrorBlock(message);
}

/**
 * Show error in search results area
 * @param {string} message - Error message to display
 */
export function showError(message) {
    const resultsDiv = document.getElementById('search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = renderErrorBlock(message);
    }
}
