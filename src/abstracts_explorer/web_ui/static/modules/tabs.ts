/**
 * Tabs Module
 *
 * Handles tab switching and tab-specific initialization.
 */

import { API_BASE } from './utils/constants.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { escapeHtml, getSelectedConference, getSelectedLocation, getSelectedYears } from './utils/dom-utils.js';
import { setCurrentTab } from './state.js';
import { getInterestingPapersSortOrder } from './state.js';
import { areClustersLoaded } from './clustering.js';

/**
 * Switch to a different tab.
 */
export function switchTab(tab: string): void {
    setCurrentTab(tab);

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-purple-600', 'text-gray-700');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    document.getElementById(`tab-${tab}`)!.classList.remove('border-transparent', 'text-gray-500');
    document.getElementById(`tab-${tab}`)!.classList.add('border-purple-600', 'text-gray-700');

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    document.getElementById(`${tab}-tab`)!.classList.remove('hidden');

    if (tab === 'interesting') {
        const sortDropdown = document.getElementById('sort-order') as HTMLSelectElement | null;
        if (sortDropdown) {
            sortDropdown.value = getInterestingPapersSortOrder();
        }
        if (window.loadInterestingPapers) {
            window.loadInterestingPapers();
        }
    }

    if (tab === 'clusters' && !areClustersLoaded()) {
        if (window.loadClusters) {
            window.loadClusters();
        }
    }

    if (tab === 'clusters') {
        if (window.loadPapersPerYear) {
            window.loadPapersPerYear();
        }
    }
}

/**
 * Load statistics.
 */
export async function loadStats(): Promise<void> {
    try {
        const selectedYears = getSelectedYears();
        const selectedConference = getSelectedConference();
        const selectedLocation = getSelectedLocation();

        const statsParams = new URLSearchParams();
        if (selectedYears.length === 1) statsParams.append('year', String(selectedYears[0]));
        if (selectedConference) statsParams.append('conference', selectedConference);
        if (selectedLocation) statsParams.append('location', selectedLocation);

        const response = await fetch(`${API_BASE}/api/stats?${statsParams.toString()}`);
        const data = await parseJsonResponse(response);

        if (data.error) {
            const errorBox = document.createElement('div');
            errorBox.className = 'text-sm text-red-200';
            errorBox.textContent = data.error;
            const statsBox = document.getElementById('stats') as HTMLElement;
            statsBox.innerHTML = '';
            statsBox.appendChild(errorBox);
            return;
        }

        let displayText = '';
        if (selectedYears.length > 1) {
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
        if (selectedLocation) {
            displayText = `${displayText} · ${selectedLocation}`;
        }

        const statsBox = document.getElementById('stats') as HTMLElement;
        statsBox.innerHTML = '';
        const totalLine = document.createElement('div');
        totalLine.className = 'text-sm font-semibold';
        const uniquePapers = data.unique_papers ?? data.total_papers;
        totalLine.textContent = `${uniquePapers.toLocaleString()} Papers`;
        const contextLine = document.createElement('div');
        contextLine.className = 'text-xs opacity-90';
        const presentations = data.total_papers && data.total_papers !== uniquePapers
            ? `${data.total_papers.toLocaleString()} presentations · `
            : '';
        contextLine.textContent = `${presentations}${displayText}`;
        statsBox.appendChild(totalLine);
        statsBox.appendChild(contextLine);
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('stats')!.innerHTML = `
            <div class="text-sm text-red-200">${escapeHtml((error as Error).message || 'Could not load stats. Please try again.')}</div>
        `;
    }
}

/**
 * Check embedding model compatibility.
 */
export async function checkEmbeddingModelCompatibility(): Promise<void> {
    try {
        const response = await fetch(`${API_BASE}/api/embedding-model-check`);
        const data = await parseJsonResponse(response);

        if (!data.compatible && data.warning) {
            const banner = document.getElementById('embedding-warning-banner') as HTMLElement;
            const messageEl = document.getElementById('embedding-warning-message') as HTMLElement;
            const currentModelEl = document.getElementById('warning-current-model') as HTMLElement;
            const storedModelEl = document.getElementById('warning-stored-model') as HTMLElement;

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
 * Dismiss embedding warning banner.
 */
export function dismissEmbeddingWarning(): void {
    const banner = document.getElementById('embedding-warning-banner') as HTMLElement;
    banner.classList.add('hidden');
}
