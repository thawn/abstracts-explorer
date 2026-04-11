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
 * Get the currently selected year from the header dropdown.
 * Returns an empty array when no year is selected (empty dropdown).
 * @returns {number[]} Array with one selected year, or [] when no year is available
 */
export function getSelectedYears() {
    const yearSelect = document.getElementById('year-selector');
    if (!yearSelect) return [];
    const value = yearSelect.value;
    if (!value) return [];  // no year selected / empty dropdown
    return [Number(value)];
}
