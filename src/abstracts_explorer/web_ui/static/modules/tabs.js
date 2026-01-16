/**
 * Tabs Module
 * 
 * Handles tab switching and tab-specific initialization.
 */

import { API_BASE } from './utils/constants.js';
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
}

/**
 * Load statistics
 * @async
 */
export async function loadStats() {
    try {
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedYear = yearSelect ? yearSelect.value : '';
        const selectedConference = conferenceSelect ? conferenceSelect.value : '';

        const statsParams = new URLSearchParams();
        if (selectedYear) statsParams.append('year', selectedYear);
        if (selectedConference) statsParams.append('conference', selectedConference);

        const response = await fetch(`${API_BASE}/api/stats?${statsParams.toString()}`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('stats').innerHTML = `
                <div class="text-sm text-red-200">${data.error}</div>
            `;
            return;
        }

        let displayText = '';
        if (data.year && data.conference) {
            displayText = `${data.conference} ${data.year}`;
        } else if (data.year) {
            displayText = `Year ${data.year}`;
        } else if (data.conference) {
            displayText = data.conference;
        } else {
            displayText = 'All Conferences';
        }

        document.getElementById('stats').innerHTML = `
            <div class="text-sm font-semibold">${data.total_papers.toLocaleString()} Abstracts</div>
            <div class="text-xs opacity-90">${displayText}</div>
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('stats').innerHTML = `
            <div class="text-sm text-red-200">Error loading stats</div>
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
        if (!response.ok) {
            console.error('Failed to check embedding model compatibility');
            return;
        }
        
        const data = await response.json();
        
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
