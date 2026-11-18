/**
 * Tabs Module
 * 
 * Handles tab switching and tab-specific initialization.
 */

import { API_BASE } from './utils/constants.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { setCurrentTab, getCurrentTab } from './state.js';
import { getInterestingPapersSortOrder } from './state.js';
import { areClustersLoaded } from './clustering.js';

/**
 * Switch to a different tab
 * @param {string} tab - Tab name
 */
export function switchTab(tab) {
    setCurrentTab(tab);

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-purple-600', 'text-gray-700');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    document.getElementById(`tab-${tab}`).classList.remove('border-transparent', 'text-gray-500');
    document.getElementById(`tab-${tab}`).classList.add('border-purple-600', 'text-gray-700');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    document.getElementById(`${tab}-tab`).classList.remove('hidden');

    // Load interesting papers when switching to that tab
    if (tab === 'interesting') {
        const sortDropdown = document.getElementById('sort-order');
        if (sortDropdown) {
            sortDropdown.value = getInterestingPapersSortOrder();
        }
        if (window.loadInterestingPapers) {
            window.loadInterestingPapers();
        }
    }

    // Load clusters when switching to that tab (only if not already loaded)
    if (tab === 'clusters' && !areClustersLoaded()) {
        if (window.loadClusters) {
            window.loadClusters();
        }
    }

    // Always load papers-per-year when switching to clusters tab
    if (tab === 'clusters') {
        if (window.loadPapersPerYear) {
            window.loadPapersPerYear();
        }
    }
}

/**
 * Load statistics
 * @async
 */
export async function loadStats() {
    try {
        const selectedYears = getSelectedYears();
        const selectedConference = getSelectedConference();

        const statsParams = new URLSearchParams();
        // Pass a single year to stats only when exactly one is selected
        if (selectedYears.length === 1) statsParams.append('year', selectedYears[0]);
        if (selectedConference) statsParams.append('conference', selectedConference);

        const response = await fetch(`${API_BASE}/api/stats?${statsParams.toString()}`);
        const data = await parseJsonResponse(response);

        if (data.error) {
            const errorBox = document.createElement('div');
            errorBox.className = 'text-sm text-red-200';
            errorBox.textContent = data.error;
            const statsBox = document.getElementById('stats');
            statsBox.innerHTML = '';
            statsBox.appendChild(errorBox);
            return;
        }

        let displayText = '';
        if (selectedYears.length > 1) {
            // Multiple years selected: show conference + year range
            const yearRange = selectedYears.join(', ');
            displayText = selectedConference ? `${selectedConference} (${yearRange})` : yearRange;
        } else if (data.year && data.conference) {
            displayText = `${data.conference} ${data.year}`;
        } else if (data.year) {
            displayText = `Year ${data.year}`;
        } else if (data.conference) {
            displayText = data.conference;
        } else {
            displayText = 'All Papers';
        }

        const statsBox = document.getElementById('stats');
        statsBox.innerHTML = '';
        const totalLine = document.createElement('div');
        totalLine.className = 'text-sm font-semibold';
        totalLine.textContent = `${data.total_papers.toLocaleString()} Abstracts`;
        const contextLine = document.createElement('div');
        contextLine.className = 'text-xs opacity-90';
        contextLine.textContent = displayText;
        statsBox.appendChild(totalLine);
        statsBox.appendChild(contextLine);
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('stats').innerHTML = `
            <div class="text-sm text-red-200">${escapeHtml(error.message || 'Could not load stats. Please try again.')}</div>
        `;
    }
}

/**
 * Check embedding model compatibility
 * @async
 */
export async function checkEmbeddingModelCompatibility() {
    try {
        const response = await fetch(`${API_BASE}/api/embedding-model-check`);
        const data = await parseJsonResponse(response);
        
        if (!data.compatible && data.warning) {
            const banner = document.getElementById('embedding-warning-banner');
            const messageEl = document.getElementById('embedding-warning-message');
            const currentModelEl = document.getElementById('warning-current-model');
            const storedModelEl = document.getElementById('warning-stored-model');
            
            messageEl.textContent = data.warning;
            currentModelEl.textContent = data.current_model || 'N/A';
            storedModelEl.textContent = data.stored_model || 'N/A';
            
            banner.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Error checking embedding model compatibility:', error);
    }
}

/**
 * Dismiss embedding warning banner
 */
export function dismissEmbeddingWarning() {
    const banner = document.getElementById('embedding-warning-banner');
    banner.classList.add('hidden');
}
