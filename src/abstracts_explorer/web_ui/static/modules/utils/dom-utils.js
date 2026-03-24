/**
 * DOM Utility Functions
 * 
 * This module provides utility functions for DOM manipulation.
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
 * Get the currently selected conference from the header selector.
 * Returns an empty string when no conference is selected.
 * @returns {string} Selected conference name, or '' if none
 */
export function getSelectedConference() {
    const conferenceSelect = document.getElementById('conference-selector');
    return conferenceSelect ? conferenceSelect.value : '';
}

/**
 * Get the currently selected years from the header multi-select.
 * "All Years" (value="") is treated as no filter (returns empty array).
 * @returns {number[]} Array of selected years, or [] for all years
 */
export function getSelectedYears() {
    const yearSelect = document.getElementById('year-selector');
    if (!yearSelect) return [];
    const selected = Array.from(yearSelect.selectedOptions)
        .map(o => o.value)
        .filter(v => v !== '');  // exclude "All Years" sentinel
    return selected.map(Number);
}
