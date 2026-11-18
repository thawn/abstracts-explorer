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
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

/**
 * Escape a value for use inside a quoted HTML attribute or inline handler.
 * Unlike escapeHtml, this also escapes quotes so the value cannot break out
 * of the attribute.
 * @param {string} text - Text to escape
 * @returns {string} Escaped attribute-safe string
 */
export function escapeAttr(text) {
    return escapeHtml(text)
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}

/**
 * Return a safe URL for an href, or '' when the scheme is not allowed.
 * @param {string} url - Candidate URL
 * @returns {string} Safe absolute/relative URL or ''
 */
export function sanitizeUrl(url) {
    const value = String(url == null ? '' : url).trim();
    if (!value) return '';
    if (/^(https?:\/\/|\/|#)/i.test(value)) return value;
    return '';
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
