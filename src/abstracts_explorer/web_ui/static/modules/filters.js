/**
 * Filters Module
 * 
 * Handles filter options, settings modals, and filter synchronization.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';

/**
 * Show the conference URL error banner if window.urlConferenceError is set.
 * Clears the global after showing so it only appears once.
 */
function showConferenceErrorBanner() {
    if (!window.urlConferenceError) return;

    const error = window.urlConferenceError;
    const banner = document.getElementById('conference-error-banner');
    const messageEl = document.getElementById('conference-error-message');
    const availableEl = document.getElementById('conference-error-available');

    if (banner && messageEl) {
        messageEl.textContent = error.message || 'Conference not found.';

        if (availableEl && error.available_conferences && error.available_conferences.length > 0) {
            const confLinks = error.available_conferences.map(conf => {
                const safeName = escapeHtml(conf);
                const safeHref = encodeURIComponent(conf);
                return `<a href="/${safeHref}" class="underline hover:text-red-900">${safeName}</a>`;
            });
            availableEl.innerHTML = `<strong>Available conferences:</strong> ${confLinks.join(', ')}`;
        } else if (availableEl) {
            availableEl.innerHTML = '';
        }

        banner.classList.remove('hidden');
    }

    delete window.urlConferenceError;
}

/**
 * Dismiss the conference error banner.
 */
export function dismissConferenceError() {
    const banner = document.getElementById('conference-error-banner');
    if (banner) {
        banner.classList.add('hidden');
    }
}

/**
 * Load filter options from API
 * @async
 */
export async function loadFilterOptions() {
    try {
        // Show conference URL error banner if present (only on first call)
        showConferenceErrorBanner();

        // Get selected year and conference from header
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedConference = getSelectedConference();
        const selectedYears = getSelectedYears();

        // Build query params for filters
        const filterParams = new URLSearchParams();
        if (selectedYears.length === 1) filterParams.append('year', selectedYears[0]);
        if (selectedConference) filterParams.append('conference', selectedConference);

        // Load session, topic, eventtype filters from database
        const filtersResponse = await fetch(`${API_BASE}/api/filters?${filterParams.toString()}`);
        const filtersData = await filtersResponse.json();

        if (filtersData.error) {
            console.error('Error loading filters:', filtersData.error);
            return;
        }

        // Load available conferences and years from database
        const availableResponse = await fetch(`${API_BASE}/api/available-filters`);
        const availableData = await availableResponse.json();

        if (availableData.error) {
            console.error('Error loading available filters:', availableData.error);
        } else {
            // Store conference_years mapping and all years for future use
            window.conferenceYearsMap = availableData.conference_years || {};
            window.allYears = availableData.years || [];

            // Populate year selector in header (only on initial load when empty)
            // Track if we just populated it to apply defaults afterward.
            let yearsJustPopulated = false;
            if (yearSelect && yearSelect.options.length === 0) {
                if (availableData.years && availableData.years.length > 0) {
                    availableData.years.forEach(year => {
                        const option = document.createElement('option');
                        option.value = year;
                        option.textContent = year;
                        yearSelect.appendChild(option);
                    });
                    yearsJustPopulated = true;
                }
            }

            // Populate conference selector in header (only on initial load when empty)
            // Track if we just populated it to apply defaults afterward.
            let conferencesJustPopulated = false;
            if (conferenceSelect && conferenceSelect.options.length === 0) {
                if (availableData.conferences && availableData.conferences.length > 0) {
                    availableData.conferences.forEach(conference => {
                        const option = document.createElement('option');
                        option.value = conference;
                        option.textContent = conference;
                        conferenceSelect.appendChild(option);
                    });
                    conferencesJustPopulated = true;
                }
            }

            // Apply defaults only on initial load (right after the selectors were populated)
            let defaultApplied = false;

            // URL-specified conference takes highest priority
            if (conferencesJustPopulated && window.urlConference) {
                const urlConf = String(window.urlConference);
                const confOption = Array.from(conferenceSelect.options).find(opt => opt.value === urlConf);
                if (confOption) {
                    conferenceSelect.value = urlConf;
                    updateYearsForConference();
                    defaultApplied = true;
                }
                // Clear so it's only applied once
                delete window.urlConference;
            }

            if (!defaultApplied && conferencesJustPopulated && availableData.default_conference) {
                const defaultConf = String(availableData.default_conference);
                const confOption = Array.from(conferenceSelect.options).find(opt => opt.value === defaultConf);
                if (confOption) {
                    conferenceSelect.value = defaultConf;
                    updateYearsForConference();
                    defaultApplied = true;
                } else if (conferenceSelect.options.length > 0) {
                    // Fall back to first available conference when default not found
                    conferenceSelect.selectedIndex = 0;
                    updateYearsForConference();
                    defaultApplied = true;
                }
            } else if (!defaultApplied && conferencesJustPopulated && conferenceSelect && conferenceSelect.options.length > 0) {
                // No specific default but conferences were just populated — select the first one
                conferenceSelect.selectedIndex = 0;
                updateYearsForConference();
                defaultApplied = true;
            }

            if ((yearsJustPopulated || conferencesJustPopulated) && yearSelect && getSelectedYears().length === 0 && availableData.default_year != null) {
                const defaultYear = String(availableData.default_year);
                const yearOption = Array.from(yearSelect.options).find(opt => opt.value === defaultYear);
                if (yearOption) {
                    yearSelect.value = defaultYear;
                    defaultApplied = true;
                }
            }

            // Refresh stats to reflect the newly applied defaults
            if (defaultApplied && window.loadStats) {
                window.loadStats();
            }

            // Set distance threshold input: prefer localStorage, then API default
            const dtInput = document.getElementById('distance-threshold-input');
            if (dtInput) {
                const stored = localStorage.getItem('searchDistanceThreshold');
                if (stored !== null) {
                    dtInput.value = stored;
                } else if (availableData.default_distance_threshold != null) {
                    dtInput.value = availableData.default_distance_threshold;
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
    
    // Reset cluster data and refresh if currently viewing clusters
    if (window.resetClusters) {
        window.resetClusters();
        if (currentTab === 'clusters' && window.loadClusters) {
            window.loadClusters();
        }
    }

    // Refresh papers-per-year chart if on clusters tab
    if (currentTab === 'clusters' && window.loadPapersPerYear) {
        window.loadPapersPerYear();
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

    // Reset cluster data and refresh if currently viewing clusters
    if (window.resetClusters) {
        window.resetClusters();
        if (currentTab === 'clusters' && window.loadClusters) {
            window.loadClusters();
        }
    }

    // Refresh papers-per-year chart if on clusters tab
    if (currentTab === 'clusters' && window.loadPapersPerYear) {
        window.loadPapersPerYear();
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
    // Preserve currently selected year (single-select)
    const currentYear = yearSelect.value;

    yearSelect.innerHTML = '';

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

    // Restore previous selection if still available; fall back to first year
    const availableYears = Array.from(yearSelect.options).map(opt => opt.value);
    if (currentYear && availableYears.includes(currentYear)) {
        yearSelect.value = currentYear;
    } else if (availableYears.length > 0) {
        yearSelect.value = availableYears[0];
    }
}
