/**
 * UI Component Utilities
 * 
 * This module provides reusable UI components and patterns.
 */

import { escapeHtml } from './dom-utils.js';

/**
 * Render an empty state message
 * @param {string} message - Main message
 * @param {string} subtext - Subtext
 * @param {string} icon - FontAwesome icon class
 * @returns {string} HTML string
 */
export function renderEmptyState(message, subtext, icon = 'fa-inbox') {
    return `
        <div class="text-center text-gray-500 py-12">
            <i class="fas ${icon} text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">${escapeHtml(message)}</p>
            <p class="text-sm">${escapeHtml(subtext)}</p>
        </div>
    `;
}

/**
 * Render error block
 * @param {string} message - Error message
 * @returns {string} HTML string
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
 * Render loading spinner
 * @param {string} message - Loading message
 * @returns {string} HTML string
 */
export function renderLoadingSpinner(message) {
    return `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
            ${message ? `<p class="ml-4 text-gray-600">${escapeHtml(message)}</p>` : ''}
        </div>
    `;
}

/**
 * Show loading in element
 * @param {string} elementId - Element ID
 * @param {string} message - Loading message
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
 * Show error in element
 * @param {string} elementId - Element ID
 * @param {string} message - Error message
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
 * Show error in search results
 * @param {string} message - Error message
 */
export function showError(message) {
    const resultsDiv = document.getElementById('search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = renderErrorBlock(message);
    }
}
