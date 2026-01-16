/**
 * Search Module
 * 
 * Handles paper search functionality including query submission
 * and results display.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml } from './utils/dom-utils.js';
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

    // Get multiple selected values from multi-select dropdowns
    const sessionSelect = document.getElementById('session-filter');
    const sessions = Array.from(sessionSelect.selectedOptions).map(opt => opt.value);

    // Get year and conference from header selectors
    const yearSelect = document.getElementById('year-selector');
    const conferenceSelect = document.getElementById('conference-selector');
    const selectedYear = yearSelect.value;
    const selectedConference = conferenceSelect.value;

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

        // Add filters only if NOT all options are selected (all selected = no filter)
        if (sessions.length > 0 && sessions.length < sessionSelect.options.length) {
            requestBody.sessions = sessions;
        }

        // Add year filter if selected
        if (selectedYear) {
            requestBody.years = [parseInt(selectedYear)];
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
        <div class="bg-white rounded-lg shadow-md p-4 mb-4">
            <div class="flex items-center justify-between">
                <div>
                    <span class="text-sm text-gray-600">Found <strong>${data.count}</strong> papers</span>
                    ${data.use_embeddings ? '<span class="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">AI-Powered</span>' : ''}
                </div>
            </div>
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
