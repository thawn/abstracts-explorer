/**
 * Search Module
 * 
 * Handles paper search functionality including query submission
 * and results display.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { showError, renderEmptyState } from './utils/ui-utils.js';
import { setCurrentSearchTerm } from './state.js';
import { formatPaperCard } from './paper-card.js';

/**
 * Search papers using the search API
 * @async
 */
export async function searchPapers() {
    const query = document.getElementById('search-input').value.trim();
    const limit = parseInt(document.getElementById('limit-select').value);
    const distanceThresholdInput = document.getElementById('distance-threshold-input');
    const distanceThreshold = distanceThresholdInput ? parseFloat(distanceThresholdInput.value) : undefined;

    // Get multiple selected values from multi-select dropdowns
    const sessionSelect = document.getElementById('session-filter');
    const sessions = Array.from(sessionSelect.selectedOptions).map(opt => opt.value);

    // Get year and conference from header selectors
    const selectedYears = getSelectedYears();
    const selectedConference = getSelectedConference();

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    // Store the current search term for rating papers
    setCurrentSearchTerm(query);

    // Show loading
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        const requestBody = {
            query,
            use_embeddings: true,
            limit
        };

        // Include distance threshold if available and valid
        if (distanceThreshold !== undefined && !isNaN(distanceThreshold) && distanceThreshold > 0) {
            requestBody.distance_threshold = distanceThreshold;
        }

        // Add filters only if NOT all options are selected (all selected = no filter)
        if (sessions.length > 0 && sessions.length < sessionSelect.options.length) {
            requestBody.sessions = sessions;
        }

        // Add year filter if selected
        if (selectedYears.length > 0) {
            requestBody.years = selectedYears;
        }

        // Add conference filter if selected
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

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        displaySearchResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showError('An error occurred while searching. Please try again.');
    }
}

/**
 * Display search results
 * @param {Object} data - Search results data
 */
export function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');

    if (!data.papers || data.papers.length === 0) {
        resultsDiv.innerHTML = renderEmptyState(
            'No papers found',
            'Try different keywords or search terms',
            'fa-inbox'
        );
        return;
    }

    // Display results header
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
                    ${data.related_topics.map(kw => `<button
                        class="px-3 py-1 bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 text-sm rounded-full border border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-800 transition-colors cursor-pointer"
                        data-topic="${escapeHtml(kw)}"
                        onclick="document.getElementById('search-input').value = this.dataset.topic; searchPapers();"
                    >${escapeHtml(kw)}</button>`).join('')}
                </div>
            </div>` : ''}
        </div>
    `;

    // Display papers using the shared formatting function
    try {
        data.papers.forEach(paper => {
            html += formatPaperCard(paper, { compact: false });
        });
    } catch (error) {
        console.error('Error formatting papers:', error);
        showError(`Error displaying search results: ${error.message}`);
        return;
    }

    resultsDiv.innerHTML = html;
}

/**
 * Advanced search field definitions.
 * Each entry maps a DOM input id to the field:"value" syntax name.
 * @type {Array<{id: string, field: string}>}
 */
const ADVANCED_FIELDS = [
    { id: 'adv-authors', field: 'authors' },
    { id: 'adv-title', field: 'title' },
    { id: 'adv-keywords', field: 'keywords' },
    { id: 'adv-abstract', field: 'abstract' },
    { id: 'adv-award', field: 'award' },
];

/**
 * Open the advanced search modal.
 * Parses the current search input to pre-populate modal fields.
 */
export function openAdvancedSearch() {
    const searchInput = document.getElementById('search-input');
    const query = searchInput ? searchInput.value.trim() : '';

    // Parse existing query into topic + field filters
    let remaining = query;
    for (const { id, field } of ADVANCED_FIELDS) {
        const input = document.getElementById(id);
        if (!input) continue;
        // Match field:"value" (case-insensitive field name)
        const regex = new RegExp(`${field}:"([^"]+)"`, 'i');
        const match = remaining.match(regex);
        if (match) {
            input.value = match[1];
            remaining = remaining.replace(match[0], '');
        } else {
            input.value = '';
        }
    }
    // Also try the "author" alias
    const authorsInput = document.getElementById('adv-authors');
    if (authorsInput && !authorsInput.value) {
        const aliasMatch = remaining.match(/author:"([^"]+)"/i);
        if (aliasMatch) {
            authorsInput.value = aliasMatch[1];
            remaining = remaining.replace(aliasMatch[0], '');
        }
    }

    const topicInput = document.getElementById('adv-topic');
    if (topicInput) {
        topicInput.value = remaining.replace(/\s+/g, ' ').trim();
    }

    const modal = document.getElementById('advanced-search-modal');
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Focus the topic field
    if (topicInput) topicInput.focus();
}

/**
 * Close the advanced search modal without applying changes.
 */
export function closeAdvancedSearch() {
    const modal = document.getElementById('advanced-search-modal');
    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Build the search query from advanced search fields and trigger a search.
 */
export function applyAdvancedSearch() {
    const parts = [];

    // Collect field filters
    for (const { id, field } of ADVANCED_FIELDS) {
        const input = document.getElementById(id);
        if (input && input.value.trim()) {
            parts.push(`${field}:"${input.value.trim()}"`);
        }
    }

    // Append the free-text topic at the end
    const topicInput = document.getElementById('adv-topic');
    if (topicInput && topicInput.value.trim()) {
        parts.push(topicInput.value.trim());
    }

    const query = parts.join(' ');
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.value = query;
    }

    closeAdvancedSearch();

    if (query) {
        searchPapers();
    }
}
