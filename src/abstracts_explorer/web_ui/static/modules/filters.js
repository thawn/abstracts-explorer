/**
 * Filters Module
 * 
 * Handles filter options, settings modals, and filter synchronization.
 */

import { API_BASE } from './utils/constants.js';

/**
 * Load filter options from API
 * @async
 */
export async function loadFilterOptions() {
    try {
        // Get selected year and conference from header
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedYear = yearSelect ? yearSelect.value : '';
        const selectedConference = conferenceSelect ? conferenceSelect.value : '';

        // Build query params for filters
        const filterParams = new URLSearchParams();
        if (selectedYear) filterParams.append('year', selectedYear);
        if (selectedConference) filterParams.append('conference', selectedConference);

        // Load session, topic, eventtype filters from database
        const filtersResponse = await fetch(`${API_BASE}/api/filters?${filterParams.toString()}`);
        const filtersData = await filtersResponse.json();

        if (filtersData.error) {
            console.error('Error loading filters:', filtersData.error);
            return;
        }

        // Load available conferences and years from plugins
        const availableResponse = await fetch(`${API_BASE}/api/available-filters`);
        const availableData = await availableResponse.json();

        if (availableData.error) {
            console.error('Error loading available filters:', availableData.error);
        } else {
            // Store conference_years mapping and all years for future use
            window.conferenceYearsMap = availableData.conference_years || {};
            window.allYears = availableData.years || [];

            // Populate year selector in header
            if (yearSelect && yearSelect.options.length === 1) {
                if (availableData.years && availableData.years.length > 0) {
                    availableData.years.forEach(year => {
                        const option = document.createElement('option');
                        option.value = year;
                        option.textContent = year;
                        yearSelect.appendChild(option);
                    });
                }
            }

            // Populate conference selector in header
            if (conferenceSelect && conferenceSelect.options.length === 1) {
                if (availableData.conferences && availableData.conferences.length > 0) {
                    availableData.conferences.forEach(conference => {
                        const option = document.createElement('option');
                        option.value = conference;
                        option.textContent = conference;
                        conferenceSelect.appendChild(option);
                    });
                }
            }
        }

        // Clear and repopulate search session filter
        const sessionSelect = document.getElementById('session-filter');
        sessionSelect.innerHTML = '';
        filtersData.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true;
            sessionSelect.appendChild(option);
        });

        // Clear and repopulate chat session filter
        const chatSessionSelect = document.getElementById('chat-session-filter');
        chatSessionSelect.innerHTML = '';
        filtersData.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true;
            chatSessionSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading filter options:', error);
    }
}

/**
 * Select all options in a filter
 * @param {string} filterId - Filter element ID
 */
export function selectAllFilter(filterId) {
    const select = document.getElementById(filterId);
    Array.from(select.options).forEach(option => {
        option.selected = true;
    });
}

/**
 * Deselect all options in a filter
 * @param {string} filterId - Filter element ID
 */
export function deselectAllFilter(filterId) {
    const select = document.getElementById(filterId);
    Array.from(select.options).forEach(option => {
        option.selected = false;
    });
}

/**
 * Open search settings modal
 */
export function openSearchSettings() {
    syncFiltersToModal('search');

    document.getElementById('modal-title').textContent = 'Search Settings';

    document.getElementById('search-settings-section').classList.remove('hidden');
    document.getElementById('chat-settings-section').classList.add('hidden');

    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    modal.dataset.context = 'search';
}

/**
 * Open chat settings modal
 */
export function openChatSettings() {
    syncFiltersToModal('chat');

    document.getElementById('modal-title').textContent = 'Chat Settings';

    document.getElementById('search-settings-section').classList.add('hidden');
    document.getElementById('chat-settings-section').classList.remove('hidden');

    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    modal.dataset.context = 'chat';
}

/**
 * Close settings modal
 */
export function closeSettings() {
    const modal = document.getElementById('settings-modal');
    const context = modal.dataset.context;

    if (context) {
        syncFiltersFromModal(context);
    }

    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Sync filters from search/chat to modal
 * @param {string} context - 'search' or 'chat'
 */
export function syncFiltersToModal(context) {
    const prefix = context === 'search' ? '' : 'chat-';

    const sessionFilter = document.getElementById(prefix + 'session-filter');
    const modalSessionFilter = document.getElementById('modal-session-filter');
    if (sessionFilter && modalSessionFilter) {
        if (modalSessionFilter.options.length === 0) {
            modalSessionFilter.innerHTML = sessionFilter.innerHTML;
        }
        Array.from(modalSessionFilter.options).forEach((opt, idx) => {
            opt.selected = sessionFilter.options[idx]?.selected || false;
        });
    }
}

/**
 * Sync filters from modal back to search/chat
 * @param {string} context - 'search' or 'chat'
 */
export function syncFiltersFromModal(context) {
    const prefix = context === 'search' ? '' : 'chat-';

    const sessionFilter = document.getElementById(prefix + 'session-filter');
    const modalSessionFilter = document.getElementById('modal-session-filter');
    if (sessionFilter && modalSessionFilter) {
        Array.from(sessionFilter.options).forEach((opt, idx) => {
            opt.selected = modalSessionFilter.options[idx]?.selected || false;
        });
    }
}

/**
 * Handle year selector change
 */
export function handleYearChange() {
    // Reload stats
    if (window.loadStats) window.loadStats();

    // Reload filter options
    loadFilterOptions();

    // Refresh interesting papers or update count
    const currentTab = document.querySelector('.tab-btn.border-purple-600')?.id?.replace('tab-', '');
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) window.loadInterestingPapers();
    } else {
        if (window.updateInterestingPapersCount) window.updateInterestingPapersCount();
    }

    // Clear search results
    const resultsDiv = document.getElementById('search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-search text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">Enter a search query to find abstracts</p>
                <p class="text-sm">Try "transformer architecture" or "reinforcement learning"</p>
            </div>
        `;
    }
}

/**
 * Handle conference selector change
 */
export function handleConferenceChange() {
    updateYearsForConference();

    // Reload stats
    if (window.loadStats) window.loadStats();

    // Reload filter options
    loadFilterOptions();

    // Refresh interesting papers or update count
    const currentTab = document.querySelector('.tab-btn.border-purple-600')?.id?.replace('tab-', '');
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) window.loadInterestingPapers();
    } else {
        if (window.updateInterestingPapersCount) window.updateInterestingPapersCount();
    }

    // Clear search results
    const resultsDiv = document.getElementById('search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-search text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">Enter a search query to find abstracts</p>
                <p class="text-sm">Try "transformer architecture" or "reinforcement learning"</p>
            </div>
        `;
    }
}

/**
 * Update years dropdown based on selected conference
 */
export function updateYearsForConference() {
    const conferenceSelect = document.getElementById('conference-selector');
    const yearSelect = document.getElementById('year-selector');

    if (!conferenceSelect || !yearSelect || !window.conferenceYearsMap) {
        return;
    }

    const selectedConference = conferenceSelect.value;
    const currentYear = yearSelect.value;

    yearSelect.innerHTML = '<option value="">All Years</option>';

    if (selectedConference) {
        const yearsForConference = window.conferenceYearsMap[selectedConference] || [];
        yearsForConference.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });
    } else {
        if (window.allYears && window.allYears.length > 0) {
            window.allYears.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearSelect.appendChild(option);
            });
        }
    }

    // Restore previous selection if available
    const availableYears = Array.from(yearSelect.options).map(opt => opt.value);
    if (currentYear && availableYears.includes(currentYear)) {
        yearSelect.value = currentYear;
    } else {
        yearSelect.value = '';
    }
}
