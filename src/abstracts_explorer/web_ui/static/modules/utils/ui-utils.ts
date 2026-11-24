/**
 * UI Component Utilities
 *
 * This module provides reusable UI components and patterns.
 */

import { escapeHtml } from './dom-utils.js';

/**
 * Render an empty state message.
 */
export function renderEmptyState(
    message: string,
    subtext: string,
    icon = 'fa-inbox'
): string {
    return `
        <div class="text-center text-gray-500 py-12">
            <i class="fas ${icon} text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">${escapeHtml(message)}</p>
            <p class="text-sm">${escapeHtml(subtext)}</p>
        </div>
    `;
}

/**
 * Render error block.
 */
export function renderErrorBlock(message: string): string {
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
 * Show loading in element.
 */
export function showLoading(elementId: string, message: string): void {
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
 * Show error in element.
 */
export function showErrorInElement(elementId: string, message: string): void {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element with id '${elementId}' not found`);
        return;
    }
    element.innerHTML = renderErrorBlock(message);
}

/**
 * Show error in search results.
 */
export function showError(message: string): void {
    const resultsDiv = document.getElementById('search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = renderErrorBlock(message);
    }
}
