/**
 * Search Module
 *
 * Handles paper search functionality including query submission
 * and results display.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { showError, renderEmptyState } from './utils/ui-utils.js';
import { setCurrentSearchTerm } from './state.js';
import { formatPaperCard } from './paper-card.js';
import type { Paper } from './types.js';

let latestSearchRequest = 0;

/**
 * Invalidate any in-flight search request so its response is ignored.
 */
export function cancelPendingSearch(): void {
    latestSearchRequest += 1;
}

/**
 * Search papers using the search API.
 */
export async function searchPapers(): Promise<void> {
    const query = (document.getElementById('search-input') as HTMLInputElement).value.trim();
    const limit = parseInt((document.getElementById('limit-select') as HTMLSelectElement).value, 10);
    const distanceThresholdInput = document.getElementById('distance-threshold-input') as HTMLInputElement | null;
    const distanceThreshold = distanceThresholdInput ? parseFloat(distanceThresholdInput.value) : undefined;

    const sessionSelect = document.getElementById('session-filter') as HTMLSelectElement;
    const sessions = Array.from(sessionSelect.selectedOptions).map(opt => opt.value);

    const selectedYears = getSelectedYears();
    const selectedConference = getSelectedConference();

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    const requestId = ++latestSearchRequest;

    setCurrentSearchTerm(query);

    const resultsDiv = document.getElementById('search-results') as HTMLElement;
    resultsDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        const requestBody: Record<string, any> = {
            query,
            use_embeddings: true,
            limit
        };

        if (distanceThreshold !== undefined && !isNaN(distanceThreshold) && distanceThreshold > 0) {
            requestBody.distance_threshold = distanceThreshold;
        }

        if (sessions.length > 0 && sessions.length < sessionSelect.options.length) {
            requestBody.sessions = sessions;
        }

        if (selectedYears.length > 0) {
            requestBody.years = selectedYears;
        }

        if (selectedConference) {
            requestBody.conferences = [selectedConference];
        }

        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (requestId !== latestSearchRequest) return;

        const data = await parseJsonResponse(response);

        if (requestId !== latestSearchRequest) return;

        if (data.error) {
            showError(data.error);
            return;
        }

        displaySearchResults(data);
    } catch (error) {
        if (requestId !== latestSearchRequest) return;
        console.error('Search error:', error);
        showError(error && (error as Error).message ? (error as Error).message : 'We could not complete the search. Please try again.');
    }
}

/**
 * Display search results.
 */
export function displaySearchResults(data: any): void {
    const resultsDiv = document.getElementById('search-results') as HTMLElement;

    if (!data.papers || data.papers.length === 0) {
        resultsDiv.innerHTML = renderEmptyState(
            'No papers found',
            'Try different keywords or search terms',
            'fa-inbox'
        );
        return;
    }

    let html = `
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4">
            <div class="flex items-center justify-between">
                <div>
                    <span class="text-sm text-gray-600 dark:text-gray-400">${data.total_similar != null ? `Showing the <strong>${data.count}</strong> best matches out of <strong>${data.total_similar}</strong> similar papers` : `Found <strong>${data.count}</strong> papers`}</span>
                    ${data.use_embeddings ? '<span class="ml-2 px-2 py-1 bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 text-xs rounded-full">LLM-Powered</span>' : ''}
                </div>
            </div>
            ${data.related_topics && data.related_topics.length > 0 ? `
            <div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
                <span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Related Topics</span>
                <div class="flex flex-wrap gap-2 mt-2">
                    ${data.related_topics.map((kw: string) => `<button
                        class="px-3 py-1 bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 text-sm rounded-full border border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-800 transition-colors cursor-pointer"
                        data-topic="${escapeHtml(kw)}"
                        onclick="document.getElementById('search-input').value = this.dataset.topic; searchPapers();"
                    >${escapeHtml(kw)}</button>`).join('')}
                </div>
            </div>` : ''}
        </div>
    `;

    try {
        (data.papers as Paper[]).forEach(paper => {
            html += formatPaperCard(paper, { compact: false });
        });
    } catch (error) {
        console.error('Error formatting papers:', error);
        showError(`Error displaying search results: ${(error as Error).message}`);
        return;
    }

    resultsDiv.innerHTML = html;
}

/**
 * Advanced search field definitions.
 */
const ADVANCED_FIELDS: Array<{ id: string; field: string }> = [
    { id: 'adv-authors', field: 'authors' },
    { id: 'adv-title', field: 'title' },
    { id: 'adv-keywords', field: 'keywords' },
    { id: 'adv-abstract', field: 'abstract' },
    { id: 'adv-award', field: 'award' },
];

/**
 * Open the advanced search modal.
 */
export function openAdvancedSearch(): void {
    const searchInput = document.getElementById('search-input') as HTMLInputElement | null;
    const query = searchInput ? searchInput.value.trim() : '';

    let remaining = query;
    for (const { id, field } of ADVANCED_FIELDS) {
        const input = document.getElementById(id) as HTMLInputElement | null;
        if (!input) continue;
        const regex = new RegExp(`${field}:"([^"]+)"`, 'i');
        const match = remaining.match(regex);
        if (match) {
            input.value = match[1];
            remaining = remaining.replace(match[0], '');
        } else {
            input.value = '';
        }
    }

    const authorsInput = document.getElementById('adv-authors') as HTMLInputElement | null;
    if (authorsInput && !authorsInput.value) {
        const aliasMatch = remaining.match(/author:"([^"]+)"/i);
        if (aliasMatch) {
            authorsInput.value = aliasMatch[1];
            remaining = remaining.replace(aliasMatch[0], '');
        }
    }

    const topicInput = document.getElementById('adv-topic') as HTMLInputElement | null;
    if (topicInput) {
        topicInput.value = remaining.replace(/\s+/g, ' ').trim();
    }

    const modal = document.getElementById('advanced-search-modal') as HTMLElement;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    if (topicInput) topicInput.focus();
}

/**
 * Close the advanced search modal without applying changes.
 */
export function closeAdvancedSearch(): void {
    const modal = document.getElementById('advanced-search-modal') as HTMLElement;
    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Build the search query from advanced search fields and trigger a search.
 */
export function applyAdvancedSearch(): void {
    const parts: string[] = [];

    for (const { id, field } of ADVANCED_FIELDS) {
        const input = document.getElementById(id) as HTMLInputElement | null;
        if (input && input.value.trim()) {
            parts.push(`${field}:"${input.value.trim()}"`);
        }
    }

    const topicInput = document.getElementById('adv-topic') as HTMLInputElement | null;
    if (topicInput && topicInput.value.trim()) {
        parts.push(topicInput.value.trim());
    }

    const query = parts.join(' ');
    const searchInput = document.getElementById('search-input') as HTMLInputElement | null;
    if (searchInput) {
        searchInput.value = query;
    }

    closeAdvancedSearch();

    if (query) {
        searchPapers();
    }
}
