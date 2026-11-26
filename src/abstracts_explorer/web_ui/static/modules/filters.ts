/**
 * Filters Module
 *
 * Handles filter options, settings modals, and filter synchronization.
 */

import { API_BASE } from './utils/constants.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { escapeHtml, getSelectedConference, getSelectedLocation, getSelectedYears } from './utils/dom-utils.js';
import { cancelPendingSearch } from './search.js';

let latestFilterRequest = 0;
let conferenceYearLocations: Record<string, Record<string, string[]>> = {};

/**
 * Update the location selector for the active conference/year.
 *
 * The selector is only shown when the selected conference-year has more than
 * one presentation location.
 */
export function updateLocationsForConferenceYear(): void {
    const container = document.getElementById('location-selector-container');
    const select = document.getElementById('location-selector') as HTMLSelectElement | null;
    if (!container || !select) return;

    const conference = getSelectedConference();
    const years = getSelectedYears();
    const year = years.length === 1 ? String(years[0]) : '';
    const locations = (conferenceYearLocations[conference] || {})[year] || [];

    if (locations.length <= 1) {
        container.classList.add('hidden');
        container.classList.remove('flex');
        select.innerHTML = '<option value="">All locations</option>';
        return;
    }

    const previous = select.value;
    select.innerHTML = '<option value="">All locations</option>';
    locations.forEach((location) => {
        const option = document.createElement('option');
        option.value = location;
        option.textContent = location;
        select.appendChild(option);
    });
    if (previous && locations.includes(previous)) {
        select.value = previous;
    }
    container.classList.remove('hidden');
    container.classList.add('flex');
}

/**
 * Show the conference URL error banner if window.urlConferenceError is set.
 */
function showConferenceErrorBanner(): void {
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
export function dismissConferenceError(): void {
    const banner = document.getElementById('conference-error-banner');
    if (banner) {
        banner.classList.add('hidden');
    }
}

/**
 * Show the configured default conference/year error banner.
 */
function showDefaultFilterErrorBanner(error: any): void {
    const banner = document.getElementById('default-filter-error-banner') as HTMLElement | null;
    const messageEl = document.getElementById('default-filter-error-message');
    const availableEl = document.getElementById('default-filter-error-available');
    const guidanceEl = document.getElementById('default-filter-error-guidance');

    if (!banner || !messageEl) return;

    if (!error) {
        banner.classList.add('hidden');
        return;
    }

    messageEl.textContent = error.message || 'The configured default filter is not available.';

    if (availableEl) {
        const parts: string[] = [];
        if (error.available_conferences && error.available_conferences.length > 0) {
            parts.push(`<strong>Available conferences:</strong> ${error.available_conferences.map(escapeHtml).join(', ')}`);
        }
        if (error.available_years && error.available_years.length > 0) {
            parts.push(`<strong>Available years:</strong> ${error.available_years.map(escapeHtml).join(', ')}`);
        }
        availableEl.innerHTML = parts.join('<br>');
    }

    if (guidanceEl) {
        guidanceEl.textContent = error.guidance || '';
    }

    banner.classList.remove('hidden');
}

/**
 * Dismiss the configured default filter error banner.
 */
export function dismissDefaultFilterError(): void {
    const banner = document.getElementById('default-filter-error-banner');
    if (banner) {
        banner.classList.add('hidden');
    }
}

/**
 * Populate a multi-select with session names, selecting every option.
 *
 * @param select - Target select element, or null to skip
 * @param sessions - Session names to populate
 */
function populateSessionSelect(select: HTMLSelectElement | null, sessions: string[]): void {
    if (!select) return;

    select.innerHTML = '';
    sessions.forEach((session) => {
        const option = document.createElement('option');
        option.value = session;
        option.textContent = session;
        option.selected = true;
        option.hidden = false;
        option.style.display = '';
        select.appendChild(option);
    });
}

/**
 * Determine whether a session option is currently visible after filtering.
 *
 * @param option - Option element to inspect
 * @returns True when the option is visible
 */
function isSessionOptionVisible(option: HTMLOptionElement): boolean {
    return !option.hidden && option.style.display !== 'none';
}

/**
 * Filter session options live based on a search query.
 *
 * Matching is a case-insensitive substring search over each option's value and
 * label. Non-matching options are hidden but remain in the DOM so their
 * selection state is preserved.
 *
 * @param selectId - ID of the multi-select to filter
 * @param query - Search query entered by the user
 */
export function filterSessionOptions(selectId: string, query: string): void {
    const select = document.getElementById(selectId) as HTMLSelectElement | null;
    if (!select) return;

    const normalized = query.trim().toLowerCase();
    Array.from(select.options).forEach((option) => {
        const matches =
            normalized === '' ||
            option.value.toLowerCase().includes(normalized) ||
            (option.textContent?.toLowerCase().includes(normalized) ?? false);
        option.hidden = !matches;
        option.style.display = matches ? '' : 'none';
    });
}

/**
 * Clear the session search field and reveal all session options.
 *
 * @param focus - Whether to move focus back to the search input
 */
export function clearSessionSearch(focus = true): void {
    const input = document.getElementById('modal-session-search') as HTMLInputElement | null;
    if (input) {
        input.value = '';
        if (focus) {
            input.focus();
        }
    }

    filterSessionOptions('modal-session-filter', '');
}

/**
 * Clear the session search field without stealing focus.
 */
function resetSessionSearch(): void {
    clearSessionSearch(false);
}

/**
 * Load filter options from API.
 */
export async function loadFilterOptions(): Promise<void> {
    const requestId = ++latestFilterRequest;
    try {
        showConferenceErrorBanner();

        const yearSelect = document.getElementById('year-selector') as HTMLSelectElement | null;
        const conferenceSelect = document.getElementById('conference-selector') as HTMLSelectElement | null;

        const availableResponse = await fetch(`${API_BASE}/api/available-filters`);
        const availableData = await parseJsonResponse(availableResponse);

        if (requestId !== latestFilterRequest) return;

        if (availableData.error) {
            console.error('Error loading available filters:', availableData.error);
        } else {
            showDefaultFilterErrorBanner(availableData.default_conference_error);
            const hasDefaultError = Boolean(availableData.default_conference_error);

            window.conferenceYearsMap = availableData.conference_years || {};
            window.allYears = availableData.years || [];
            conferenceYearLocations = availableData.conference_year_locations || {};

            let yearsJustPopulated = false;
            if (yearSelect && yearSelect.options.length === 0) {
                if (availableData.years && availableData.years.length > 0) {
                    availableData.years.forEach((year: number | string) => {
                        const option = document.createElement('option');
                        option.value = String(year);
                        option.textContent = String(year);
                        yearSelect.appendChild(option);
                    });
                    yearsJustPopulated = true;
                }
            }

            let conferencesJustPopulated = false;
            if (conferenceSelect && conferenceSelect.options.length === 0) {
                if (availableData.conferences && availableData.conferences.length > 0) {
                    availableData.conferences.forEach((conference: string) => {
                        const option = document.createElement('option');
                        option.value = conference;
                        option.textContent = conference;
                        conferenceSelect.appendChild(option);
                    });
                    conferencesJustPopulated = true;
                }
            }

            let defaultApplied = false;

            if (conferencesJustPopulated && window.urlConference) {
                const urlConf = String(window.urlConference);
                const confOption = Array.from(conferenceSelect!.options).find(opt => opt.value === urlConf);
                if (confOption) {
                    conferenceSelect!.value = urlConf;
                    updateYearsForConference();
                    defaultApplied = true;
                }
                delete window.urlConference;
            }

            if (!defaultApplied && conferencesJustPopulated && availableData.default_conference) {
                const defaultConf = String(availableData.default_conference);
                const confOption = Array.from(conferenceSelect!.options).find(opt => opt.value === defaultConf);
                if (confOption) {
                    conferenceSelect!.value = defaultConf;
                    updateYearsForConference();
                    defaultApplied = true;
                }
            } else if (
                !defaultApplied &&
                !hasDefaultError &&
                conferencesJustPopulated &&
                conferenceSelect &&
                conferenceSelect.options.length > 0
            ) {
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

            // When the configured default is unavailable, the browser selects the
            // first option implicitly. Make that selection explicit so the years
            // list and the stats query agree with the visible filters.
            if (!defaultApplied && conferencesJustPopulated && conferenceSelect && conferenceSelect.options.length > 0) {
                if (!conferenceSelect.value) {
                    conferenceSelect.selectedIndex = 0;
                }
                updateYearsForConference();
            }

            updateLocationsForConferenceYear();

            if (window.loadStats) {
                window.loadStats();
            }

            const dtInput = document.getElementById('distance-threshold-input') as HTMLInputElement | null;
            if (dtInput) {
                const stored = localStorage.getItem('searchDistanceThreshold');
                if (stored !== null) {
                    dtInput.value = stored;
                } else if (availableData.default_distance_threshold != null) {
                    dtInput.value = String(availableData.default_distance_threshold);
                }
            }
        }

        const selectedConference = getSelectedConference();
        const selectedYears = getSelectedYears();

        const filterParams = new URLSearchParams();
        if (selectedYears.length === 1) filterParams.append('year', String(selectedYears[0]));
        if (selectedConference) filterParams.append('conference', selectedConference);
        const selectedLocation = getSelectedLocation();
        if (selectedLocation) filterParams.append('location', selectedLocation);

        const filtersResponse = await fetch(`${API_BASE}/api/filters?${filterParams.toString()}`);
        const filtersData = await parseJsonResponse(filtersResponse);

        if (requestId !== latestFilterRequest) return;

        if (filtersData.error) {
            console.error('Error loading filters:', filtersData.error);
            return;
        }

        const sessionSelect = document.getElementById('session-filter') as HTMLSelectElement | null;
        const chatSessionSelect = document.getElementById('chat-session-filter') as HTMLSelectElement | null;
        const modalSessionSelect = document.getElementById('modal-session-filter') as HTMLSelectElement | null;

        populateSessionSelect(sessionSelect, filtersData.sessions);
        populateSessionSelect(chatSessionSelect, filtersData.sessions);
        populateSessionSelect(modalSessionSelect, filtersData.sessions);
    } catch (error) {
        console.error('Error loading filter options:', error);
        if (window.loadStats) {
            window.loadStats();
        }
    }
}

/**
 * Select all options in a filter.
 */
export function selectAllFilter(filterId: string): void {
    const select = document.getElementById(filterId) as HTMLSelectElement;
    Array.from(select.options).forEach(option => {
        option.selected = isSessionOptionVisible(option);
    });
}

/**
 * Deselect all visible options in a filter.
 *
 * Options hidden by the session search keep their current selection state.
 */
export function deselectAllFilter(filterId: string): void {
    const select = document.getElementById(filterId) as HTMLSelectElement;
    Array.from(select.options).forEach(option => {
        if (isSessionOptionVisible(option)) {
            option.selected = false;
        }
    });
}

/**
 * Open search settings modal.
 */
export function openSearchSettings(): void {
    syncFiltersToModal('search');
    resetSessionSearch();

    document.getElementById('modal-title')!.textContent = 'Search Settings';

    document.getElementById('search-settings-section')!.classList.remove('hidden');
    document.getElementById('chat-settings-section')!.classList.add('hidden');

    const modal = document.getElementById('settings-modal') as HTMLElement;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    modal.dataset.context = 'search';
}

/**
 * Open chat settings modal.
 */
export function openChatSettings(): void {
    syncFiltersToModal('chat');
    resetSessionSearch();

    document.getElementById('modal-title')!.textContent = 'Chat Settings';

    document.getElementById('search-settings-section')!.classList.add('hidden');
    document.getElementById('chat-settings-section')!.classList.remove('hidden');

    const modal = document.getElementById('settings-modal') as HTMLElement;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    modal.dataset.context = 'chat';
}

/**
 * Close settings modal.
 */
export function closeSettings(): void {
    const modal = document.getElementById('settings-modal') as HTMLElement;
    const context = modal.dataset.context;

    if (context) {
        syncFiltersFromModal(context);
    }

    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Sync filters from search/chat to modal.
 */
export function syncFiltersToModal(context: string): void {
    const prefix = context === 'search' ? '' : 'chat-';

    const sessionFilter = document.getElementById(prefix + 'session-filter') as HTMLSelectElement | null;
    const modalSessionFilter = document.getElementById('modal-session-filter') as HTMLSelectElement | null;
    if (sessionFilter && modalSessionFilter) {
        const sourceValues = Array.from(sessionFilter.options).map((opt) => opt.value);
        const modalValues = Array.from(modalSessionFilter.options).map((opt) => opt.value);
        const needsRebuild =
            sourceValues.length !== modalValues.length ||
            sourceValues.some((value, idx) => value !== modalValues[idx]);

        if (needsRebuild) {
            populateSessionSelect(modalSessionFilter, sourceValues);
        }

        Array.from(modalSessionFilter.options).forEach((opt, idx) => {
            opt.selected = sessionFilter.options[idx]?.selected || false;
        });
    }
}

/**
 * Sync filters from modal back to search/chat.
 */
export function syncFiltersFromModal(context: string): void {
    const prefix = context === 'search' ? '' : 'chat-';

    const sessionFilter = document.getElementById(prefix + 'session-filter') as HTMLSelectElement | null;
    const modalSessionFilter = document.getElementById('modal-session-filter') as HTMLSelectElement | null;
    if (sessionFilter && modalSessionFilter) {
        Array.from(sessionFilter.options).forEach((opt, idx) => {
            opt.selected = modalSessionFilter.options[idx]?.selected || false;
        });
    }
}

function currentActiveTab(): string | undefined {
    return document.querySelector('.tab-btn.border-purple-600')?.id?.replace('tab-', '');
}

/**
 * Handle year selector change.
 */
export function handleYearChange(): void {
    cancelPendingSearch();

    loadFilterOptions();

    const currentTab = currentActiveTab();
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) window.loadInterestingPapers();
    } else {
        if (window.updateInterestingPapersCount) window.updateInterestingPapersCount();
    }

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

    if (window.resetClusters) {
        window.resetClusters();
        if (currentTab === 'clusters' && window.loadClusters) {
            window.loadClusters();
        }
    }

    if (currentTab === 'clusters' && window.loadPapersPerYear) {
        window.loadPapersPerYear();
    }
}

/**
 * Handle conference selector change.
 */
export function handleConferenceChange(): void {
    updateYearsForConference();
    cancelPendingSearch();

    loadFilterOptions();

    const currentTab = currentActiveTab();
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) window.loadInterestingPapers();
    } else {
        if (window.updateInterestingPapersCount) window.updateInterestingPapersCount();
    }

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

    if (window.resetClusters) {
        window.resetClusters();
        if (currentTab === 'clusters' && window.loadClusters) {
            window.loadClusters();
        }
    }

    if (currentTab === 'clusters' && window.loadPapersPerYear) {
        window.loadPapersPerYear();
    }
}

/**
 * Handle location selector change.
 */
export function handleLocationChange(): void {
    cancelPendingSearch();

    loadFilterOptions();

    const currentTab = currentActiveTab();
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) window.loadInterestingPapers();
    } else if (window.updateInterestingPapersCount) {
        window.updateInterestingPapersCount();
    }

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

    if (window.resetClusters) {
        window.resetClusters();
        if (currentTab === 'clusters' && window.loadClusters) {
            window.loadClusters();
        }
    }
}

/**
 * Update years dropdown based on selected conference.
 */
export function updateYearsForConference(): void {
    const conferenceSelect = document.getElementById('conference-selector') as HTMLSelectElement | null;
    const yearSelect = document.getElementById('year-selector') as HTMLSelectElement | null;

    if (!conferenceSelect || !yearSelect || !window.conferenceYearsMap) {
        return;
    }

    const selectedConference = conferenceSelect.value;
    const currentYear = yearSelect.value;

    yearSelect.innerHTML = '';

    if (selectedConference) {
        const yearsForConference = window.conferenceYearsMap[selectedConference] || [];
        yearsForConference.forEach((year) => {
            const option = document.createElement('option');
            option.value = String(year);
            option.textContent = String(year);
            yearSelect.appendChild(option);
        });
    } else {
        if (window.allYears && window.allYears.length > 0) {
            window.allYears.forEach((year) => {
                const option = document.createElement('option');
                option.value = String(year);
                option.textContent = String(year);
                yearSelect.appendChild(option);
            });
        }
    }

    const availableYears = Array.from(yearSelect.options).map(opt => opt.value);
    if (currentYear && availableYears.includes(currentYear)) {
        yearSelect.value = currentYear;
    } else if (availableYears.length > 0) {
        yearSelect.value = availableYears[0];
    }

    updateLocationsForConferenceYear();
}
