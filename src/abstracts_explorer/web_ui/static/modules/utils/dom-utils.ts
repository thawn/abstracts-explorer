/**
 * DOM Utility Functions
 *
 * This module provides utility functions for DOM manipulation.
 */

/**
 * Escape HTML special characters to prevent XSS attacks.
 *
 * @param text - Text to escape
 * @returns Escaped HTML string
 */
export function escapeHtml(text: string | null | undefined): string {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

/**
 * Escape a value for use inside a quoted HTML attribute or inline handler.
 * Unlike escapeHtml, this also escapes quotes so the value cannot break out
 * of the attribute.
 *
 * @param text - Text to escape
 * @returns Escaped attribute-safe string
 */
export function escapeAttr(text: string | null | undefined): string {
    return escapeHtml(text)
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}

/**
 * Return a safe URL for an href, or '' when the scheme is not allowed.
 *
 * @param url - Candidate URL
 * @returns Safe absolute/relative URL or ''
 */
export function sanitizeUrl(url: string | null | undefined): string {
    const value = String(url == null ? '' : url).trim();
    if (!value) return '';
    if (/^(https?:\/\/|\/|#)/i.test(value)) return value;
    return '';
}

/**
 * Get the currently selected conference from the header selector.
 * Returns an empty string when no conference is selected.
 *
 * @returns Selected conference name, or '' if none
 */
export function getSelectedConference(): string {
    const conferenceSelect = document.getElementById('conference-selector') as HTMLSelectElement | null;
    return conferenceSelect ? conferenceSelect.value : '';
}

/**
 * Get the currently selected year from the header dropdown.
 * Returns an empty array when no year is selected (empty dropdown).
 *
 * @returns Array with one selected year, or [] when no year is available
 */
export function getSelectedYears(): number[] {
    const yearSelect = document.getElementById('year-selector') as HTMLSelectElement | null;
    if (!yearSelect) return [];
    const value = yearSelect.value;
    if (!value) return [];  // no year selected / empty dropdown
    return [Number(value)];
}
