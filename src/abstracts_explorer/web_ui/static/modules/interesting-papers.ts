/**
 * Interesting Papers Module
 *
 * Handles the display, management, and export of rated/interesting papers.
 */

import { API_BASE } from './utils/constants.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { escapeAttr, escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { renderEmptyState } from './utils/ui-utils.js';
import { naturalSortPosterPosition } from './utils/sort-utils.js';
import {
    getAllPaperPriorities,
    getPaperIdsWithPriorities,
    getInterestingPapersSortOrder,
    setInterestingPapersSortOrder,
    getCurrentInterestingSession,
    setCurrentInterestingSession,
    updateSearchTermForMultiplePapers,
    updatePaperSearchTerm,
    savePriorities,
    paperPriorities
} from './state.js';
import { formatPaperCard } from './paper-card.js';
import { updateYearsForConference } from './filters.js';
import type { Paper, PaperPriority } from './types.js';

interface InterestingConferenceCombo {
    conference: string;
    year: number;
    count: number;
}

const COMBO_VALUE_SEPARATOR = '|';

let donationPromptShownThisSession = false;

const DONATION_CONFIRMATION_MESSAGE =
    'Would you like to donate your paper ratings to help improve our service?\n\n' +
    '✓ Your data will be fully anonymized\n' +
    '✓ No personal information will be collected\n' +
    '✓ Data will only be used to improve paper recommendations\n' +
    '✓ You can donate again, after rating more papers\n\n' +
    'Thank you for contributing to research!';

function hasDataBeenDonated(): boolean {
    const currentPriorities = JSON.stringify(getAllPaperPriorities());
    const donatedData = localStorage.getItem('donatedPrioritiesHash');

    if (!donatedData) {
        return false;
    }

    return donatedData === currentPriorities;
}

function markDataAsDonated(): void {
    const currentPriorities = JSON.stringify(getAllPaperPriorities());
    localStorage.setItem('donatedPrioritiesHash', currentPriorities);
}

/**
 * Update visibility of control buttons based on whether papers are rated
 * and whether data has been donated.
 */
export function updateControlsVisibility(): void {
    const hasPapers = Object.keys(getAllPaperPriorities()).length > 0;
    const sortControl = document.getElementById('interesting-controls-sort');
    const actionsControl = document.getElementById('interesting-controls-actions');
    const donateBtn = document.getElementById('donate-data-btn');

    if (sortControl) {
        sortControl.classList.toggle('hidden', !hasPapers);
    }

    if (actionsControl) {
        actionsControl.classList.toggle('hidden', !hasPapers);
    }

    if (donateBtn) {
        if (window.readOnly) {
            donateBtn.classList.add('hidden');
        } else if (hasPapers) {
            donateBtn.classList.toggle('hidden', hasDataBeenDonated());
        }
    }
}

/**
 * Hide the other-conference switcher and clear its options.
 */
function hideInterestingConferenceMenu(): void {
    const menu = document.getElementById('interesting-conference-menu');
    if (menu) {
        menu.classList.add('hidden');
    }

    const select = document.getElementById('interesting-conference-select') as HTMLSelectElement | null;
    if (select) {
        select.innerHTML = '<option value="">Select conference/year…</option>';
    }
}

/**
 * Collect conference/year combinations represented among the given papers.
 */
function collectInterestingConferenceCombos(papers: Paper[]): InterestingConferenceCombo[] {
    const combos = new Map<string, InterestingConferenceCombo>();

    papers.forEach(paper => {
        const conference = (paper.conference || '').trim();
        const year = Number(paper.year);
        if (!conference || !Number.isFinite(year) || year <= 0) {
            return;
        }

        const key = `${conference}${COMBO_VALUE_SEPARATOR}${year}`;
        const existing = combos.get(key);
        if (existing) {
            existing.count += 1;
        } else {
            combos.set(key, { conference, year, count: 1 });
        }
    });

    return Array.from(combos.values()).sort((a, b) => {
        const conferenceCompare = a.conference.localeCompare(b.conference);
        if (conferenceCompare !== 0) return conferenceCompare;
        return b.year - a.year;
    });
}

/**
 * Render the menu of other conference/year combinations that have rated papers.
 */
export function renderInterestingConferenceMenu(papers: Paper[]): void {
    const menu = document.getElementById('interesting-conference-menu');
    const select = document.getElementById('interesting-conference-select') as HTMLSelectElement | null;
    if (!menu || !select) return;

    const selectedYears = getSelectedYears();
    const selectedConference = getSelectedConference();
    const selectedYear = selectedYears.length > 0 ? selectedYears[0] : null;

    const otherCombos = collectInterestingConferenceCombos(papers).filter(combo => {
        return !(combo.conference === selectedConference && combo.year === selectedYear);
    });

    if (otherCombos.length === 0) {
        hideInterestingConferenceMenu();
        return;
    }

    select.innerHTML = '<option value="">Select conference/year…</option>';
    otherCombos.forEach(combo => {
        const option = document.createElement('option');
        option.value = `${encodeURIComponent(combo.conference)}${COMBO_VALUE_SEPARATOR}${combo.year}`;
        option.textContent = `${combo.conference} ${combo.year} (${combo.count})`;
        select.appendChild(option);
    });

    menu.classList.remove('hidden');
}

/**
 * Switch the header selectors to the given conference/year and refresh the view.
 */
export function switchInterestingConference(conference: string, year: number): void {
    const conferenceSelect = document.getElementById('conference-selector') as HTMLSelectElement | null;
    const yearSelect = document.getElementById('year-selector') as HTMLSelectElement | null;
    if (!conferenceSelect || !yearSelect) return;

    conferenceSelect.value = conference;
    updateYearsForConference();

    const yearValue = String(year);
    const hasYearOption = Array.from(yearSelect.options).some(option => option.value === yearValue);
    if (!hasYearOption) {
        const option = document.createElement('option');
        option.value = yearValue;
        option.textContent = yearValue;
        yearSelect.appendChild(option);
    }
    yearSelect.value = yearValue;

    const menuSelect = document.getElementById('interesting-conference-select') as HTMLSelectElement | null;
    if (menuSelect) {
        menuSelect.value = '';
    }

    if (window.handleConferenceChange) {
        window.handleConferenceChange();
    }
}

/**
 * Handle a selection from the other-conference switcher.
 */
export function switchInterestingConferenceFromMenu(value: string): void {
    if (!value) return;

    const separatorIndex = value.lastIndexOf(COMBO_VALUE_SEPARATOR);
    if (separatorIndex < 0) return;

    const conference = decodeURIComponent(value.slice(0, separatorIndex));
    const year = Number(value.slice(separatorIndex + 1));
    if (!conference || !Number.isFinite(year)) return;

    switchInterestingConference(conference, year);
}

/**
 * Load and display interesting papers.
 */
export async function loadInterestingPapers(): Promise<void> {
    const listDiv = document.getElementById('interesting-papers-list') as HTMLElement;
    const tabsContainer = document.getElementById('interesting-session-tabs') as HTMLElement;

    updateControlsVisibility();

    if (Object.keys(getAllPaperPriorities()).length === 0) {
        hideInterestingConferenceMenu();
        listDiv.innerHTML = renderEmptyState(
            'No papers rated yet',
            'Rate papers using the stars to add them here',
            'fa-star'
        );
        tabsContainer.classList.add('hidden');

        const countElement = document.getElementById('interesting-count');
        if (countElement) {
            countElement.textContent = '0';
        }
        return;
    }

    tabsContainer.classList.remove('hidden');

    listDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        const paperIds = getPaperIdsWithPriorities();
        const response = await fetch(`${API_BASE}/api/papers/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ paper_ids: paperIds })
        });

        const data = await parseJsonResponse(response);

        if (data.error) {
            listDiv.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-6">
                    <p class="text-red-700">${escapeHtml(data.error)}</p>
                </div>
            `;
            return;
        }

        displayInterestingPapers(data.papers as Paper[]);
    } catch (error) {
        console.error('Error loading interesting papers:', error);
        listDiv.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-6">
                <p class="text-red-700">${escapeHtml((error as Error).message || 'Could not load papers. Please try again.')}</p>
            </div>
        `;
    }
}

/**
 * Display interesting papers grouped by search term.
 */
export function displayInterestingPapers(papers: Paper[]): void {
    const listDiv = document.getElementById('interesting-papers-list') as HTMLElement;
    const tabsNav = document.getElementById('interesting-session-tabs-nav') as HTMLElement;

    renderInterestingConferenceMenu(papers);

    const selectedYears = getSelectedYears();
    const selectedConference = getSelectedConference();

    let filteredPapers = papers;
    if (selectedYears.length > 0) {
        filteredPapers = filteredPapers.filter(paper => selectedYears.includes(Number(paper.year)));
    }
    if (selectedConference) {
        filteredPapers = filteredPapers.filter(paper => paper.conference === selectedConference);
    }

    if (filteredPapers.length === 0) {
        listDiv.innerHTML = renderEmptyState(
            'No papers match the selected filters',
            'Try selecting a different conference or year',
            'fa-filter'
        );
        const tabsContainer = document.getElementById('interesting-session-tabs') as HTMLElement;
        tabsContainer.classList.add('hidden');

        const countElement = document.getElementById('interesting-count');
        if (countElement) {
            countElement.textContent = '0';
        }
        return;
    }

    const priorities = getAllPaperPriorities();
    filteredPapers.forEach(paper => {
        const paperData = priorities[paper.uid];
        paper.priority = paperData?.priority || 0;
        paper.searchTerm = paperData?.searchTerm || 'Unknown';
    });

    const sortOrder = getInterestingPapersSortOrder();
    filteredPapers.sort((a, b) => {
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;

        if (sortOrder === 'search-rating-poster') {
            const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
            if (searchTermCompare !== 0) return searchTermCompare;

            if (a.priority !== b.priority) return (b.priority ?? 0) - (a.priority ?? 0);

            return naturalSortPosterPosition(a.poster_position, b.poster_position);
        } else if (sortOrder === 'rating-poster-search') {
            if (a.priority !== b.priority) return (b.priority ?? 0) - (a.priority ?? 0);

            const posterCompare = naturalSortPosterPosition(a.poster_position, b.poster_position);
            if (posterCompare !== 0) return posterCompare;

            return (a.searchTerm || '').localeCompare(b.searchTerm || '');
        } else if (sortOrder === 'poster-search-rating') {
            const posterCompare = naturalSortPosterPosition(a.poster_position, b.poster_position);
            if (posterCompare !== 0) return posterCompare;

            const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
            if (searchTermCompare !== 0) return searchTermCompare;

            return (b.priority ?? 0) - (a.priority ?? 0);
        }
        return 0;
    });

    const groupedBySession: Record<string, Record<string, Paper[]>> = {};
    filteredPapers.forEach(paper => {
        const session = paper.session || 'No Session';
        if (!groupedBySession[session]) {
            groupedBySession[session] = {};
        }

        let groupKey: string;
        if (sortOrder === 'search-rating-poster') {
            groupKey = paper.searchTerm || 'Unknown';
        } else if (sortOrder === 'rating-poster-search') {
            groupKey = `${paper.priority} ${paper.priority === 1 ? 'star' : 'stars'}`;
        } else {
            groupKey = 'All Papers';
        }

        if (!groupedBySession[session][groupKey]) {
            groupedBySession[session][groupKey] = [];
        }
        groupedBySession[session][groupKey].push(paper);
    });

    const sessions = Object.keys(groupedBySession).sort();

    let currentSession = getCurrentInterestingSession();
    if (!currentSession || !sessions.includes(currentSession)) {
        currentSession = sessions[0] || null;
        setCurrentInterestingSession(currentSession);
    }

    let tabsHtml = '';
    sessions.forEach(session => {
        const isActive = session === currentSession;
        const activeClass = isActive ? 'border-b-2 border-purple-600 text-purple-600 dark:text-purple-400' : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:border-gray-300';
        tabsHtml += `
            <button
                onclick="switchInterestingSession(${escapeAttr(JSON.stringify(session))})"
                class="px-6 py-3 text-sm font-medium ${activeClass} border-b-2 border-transparent focus:outline-none transition-colors whitespace-nowrap">
                <i class="fas fa-calendar-alt mr-2"></i>${escapeHtml(session)}
            </button>
        `;
    });
    tabsNav.innerHTML = tabsHtml;

    let html = '';

    if (currentSession && groupedBySession[currentSession]) {
        const groups = groupedBySession[currentSession];

        let groupIcon = 'fa-search';
        if (sortOrder === 'rating-poster-search') {
            groupIcon = 'fa-star';
        } else if (sortOrder === 'poster-search-rating') {
            groupIcon = 'fa-list';
        }

        const sortedGroupKeys = Object.keys(groups);
        if (sortOrder === 'search-rating-poster') {
            sortedGroupKeys.sort();
        } else if (sortOrder === 'rating-poster-search') {
            sortedGroupKeys.sort((a, b) => {
                const starsA = parseInt(a.split(' ')[0], 10);
                const starsB = parseInt(b.split(' ')[0], 10);
                return starsB - starsA;
            });
        }

        for (const groupKey of sortedGroupKeys) {
            const groupPapers = groups[groupKey];

            if (sortOrder === 'poster-search-rating') {
                html += '<div class="space-y-4">';
                groupPapers.forEach(paper => {
                    html += formatPaperCard(paper, { compact: false, showSearchTerm: true });
                });
                html += '</div>';
            } else {
                const editButton = (sortOrder === 'search-rating-poster')
                    ? `<button onclick="editSearchTerm(${escapeAttr(JSON.stringify(currentSession))}, ${escapeAttr(JSON.stringify(groupKey))}, event)"
                              class="ml-2 text-sm text-purple-600 hover:text-purple-800 focus:outline-none"
                              title="Edit search term">
                           <i class="fas fa-edit"></i>
                       </button>`
                    : '';

                html += `
                    <div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg shadow-md p-5 mb-4">
                        <h3 class="text-lg font-bold text-gray-800 dark:text-gray-100 mb-4 border-b border-blue-200 dark:border-blue-700 pb-2 flex items-center justify-between">
                            <span>
                                <i class="fas ${groupIcon} text-blue-600 dark:text-blue-400 mr-2"></i>${escapeHtml(groupKey)}
                            </span>
                            ${editButton}
                        </h3>
                        <div class="space-y-4">
                `;

                groupPapers.forEach(paper => {
                    const showSearchTermBadge = sortOrder !== 'search-rating-poster';
                    html += formatPaperCard(paper, { compact: false, showSearchTerm: showSearchTermBadge });
                });

                html += `
                        </div>
                    </div>
                `;
            }
        }
    }

    listDiv.innerHTML = html;

    const countElement = document.getElementById('interesting-count');
    if (countElement) {
        countElement.textContent = String(filteredPapers.length);
    }
}

/**
 * Save interesting papers as markdown (zip export).
 */
export async function saveInterestingPapersAsMarkdown(event: Event): Promise<void> {
    if (Object.keys(getAllPaperPriorities()).length === 0) {
        alert('No papers rated yet. Rate some papers before saving.');
        return;
    }

    if (!window.readOnly && !donationPromptShownThisSession) {
        const exportDonateMessage =
            '💡 Before you export, would you like to donate your ratings?\n\n' +
            'Your data helps improve our service and is fully anonymized.\n\n' +
            'Click "OK" to donate now (recommended)\n' +
            'Click "Cancel" to skip and continue with export';

        if (confirm(exportDonateMessage)) {
            await donateInterestingPapersData();
        }

        donationPromptShownThisSession = true;
    }

    const button = event.target as HTMLButtonElement;
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Preparing download...';
    button.disabled = true;

    try {
        const paperIds = getPaperIdsWithPriorities();
        const searchInput = document.getElementById('search-input') as HTMLInputElement | null;
        const searchQuery = searchInput ? searchInput.value : '';

        const response = await fetch(`${API_BASE}/api/export/interesting-papers`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                paper_ids: paperIds,
                priorities: getAllPaperPriorities(),
                search_query: searchQuery,
                sort_order: getInterestingPapersSortOrder()
            })
        });

        if (!response.ok) {
            await parseJsonResponse(response);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `interesting-papers-${new Date().toISOString().split('T')[0]}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        button.innerHTML = '<i class="fas fa-check mr-2"></i>Downloaded!';
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);
    } catch (error) {
        console.error('Error saving markdown:', error);
        alert('Error creating export: ' + (error as Error).message);
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

/**
 * Save interesting papers data as JSON file.
 */
export function saveInterestingPapersAsJSON(): void {
    if (Object.keys(getAllPaperPriorities()).length === 0) {
        alert('No papers rated yet. Rate some papers before saving.');
        return;
    }

    try {
        const exportData = {
            version: '1.0',
            exportDate: new Date().toISOString(),
            sortOrder: getInterestingPapersSortOrder(),
            paperPriorities: getAllPaperPriorities(),
            paperCount: getPaperIdsWithPriorities().length
        };

        const jsonString = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `interesting-papers-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log('Successfully saved interesting papers as JSON');
    } catch (error) {
        console.error('Error saving JSON:', error);
        alert('Error saving JSON file: ' + (error as Error).message);
    }
}

/**
 * Trigger file input for loading JSON.
 */
export function loadInterestingPapersFromJSON(): void {
    const fileInput = document.getElementById('json-file-input') as HTMLInputElement | null;
    if (fileInput) {
        fileInput.click();
    }
}

/**
 * Handle JSON file load.
 */
export function handleJSONFileLoad(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.json')) {
        alert('Please select a JSON file.');
        input.value = '';
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        try {
            const importData = JSON.parse((e.target as FileReader).result as string);

            if (!importData.paperPriorities || typeof importData.paperPriorities !== 'object') {
                throw new Error('Invalid JSON format: missing or invalid paperPriorities');
            }

            for (const [paperId, rating] of Object.entries(importData.paperPriorities)) {
                if (typeof rating !== 'object' || rating === null || Array.isArray(rating)) {
                    throw new Error(
                        `Invalid rating format for paper '${paperId}': expected an object with ` +
                        "'priority' and optional 'searchTerm'."
                    );
                }
            }

            const existingCount = getPaperIdsWithPriorities().length;
            let shouldProceed = true;

            if (existingCount > 0) {
                const importCount = Object.keys(importData.paperPriorities).length;
                shouldProceed = confirm(
                    `You have ${existingCount} existing rated paper(s).\n` +
                    `This will load ${importCount} paper(s) from the file.\n\n` +
                    `Choose OK to merge (existing ratings will be preserved unless they conflict),\n` +
                    `or Cancel to abort.`
                );
            }

            if (!shouldProceed) {
                input.value = '';
                return;
            }

            let newCount = 0;
            let updatedCount = 0;

            for (const [paperId, data] of Object.entries(importData.paperPriorities)) {
                if (paperPriorities[paperId]) {
                    updatedCount++;
                } else {
                    paperPriorities[paperId] = data as PaperPriority;
                    newCount++;
                }
            }

            if (importData.sortOrder && !localStorage.getItem('interestingPapersSortOrder')) {
                setInterestingPapersSortOrder(importData.sortOrder);
                const sortSelect = document.getElementById('sort-order') as HTMLSelectElement | null;
                if (sortSelect) {
                    sortSelect.value = importData.sortOrder;
                }
            }

            savePriorities();

            if (window.updateInterestingPapersCount) {
                window.updateInterestingPapersCount();
            }

            updateControlsVisibility();

            const currentTab = document.querySelector('.tab-btn.border-purple-600')?.id?.replace('tab-', '');
            if (currentTab === 'interesting') {
                loadInterestingPapers();
            }

            const afterCount = getPaperIdsWithPriorities().length;
            let message = `Successfully loaded ${newCount} new rated paper(s).`;
            if (updatedCount > 0) {
                message += `\n${updatedCount} paper(s) were already rated (kept existing ratings).`;
            }
            message += `\nTotal rated papers: ${afterCount}`;

            if (importData.exportDate) {
                const exportDate = new Date(importData.exportDate).toLocaleString();
                message += `\n\nFile exported on: ${exportDate}`;
            }

            alert(message);
            console.log('Successfully loaded interesting papers from JSON', {
                newCount,
                updatedCount,
                totalCount: afterCount
            });

        } catch (error) {
            console.error('Error loading JSON:', error);
            alert('Error loading JSON file: ' + (error as Error).message + '\n\nPlease ensure the file is a valid interesting papers export.');
        }

        input.value = '';
    };

    reader.onerror = function () {
        alert('Error reading file. Please try again.');
        input.value = '';
    };

    reader.readAsText(file);
}

/**
 * Edit search term for a group of papers.
 */
export function editSearchTerm(session: string, oldSearchTerm: string, event: Event): void {
    event.stopPropagation();

    const newSearchTerm = prompt('Edit search term:', oldSearchTerm);

    if (newSearchTerm === null || newSearchTerm.trim() === '') {
        return;
    }

    const trimmedNewSearchTerm = newSearchTerm.trim();

    if (trimmedNewSearchTerm === oldSearchTerm) {
        return;
    }

    const updatedCount = updateSearchTermForMultiplePapers(oldSearchTerm, trimmedNewSearchTerm);

    loadInterestingPapers();

    if (updatedCount > 0) {
        console.log(`Updated search term for ${updatedCount} paper(s)`);
    }
}

/**
 * Edit search term for a single paper.
 */
export function editPaperSearchTerm(paperId: string, event: Event): void {
    event.stopPropagation();

    const priorities = getAllPaperPriorities();
    const paperData = priorities[paperId];
    if (!paperData) {
        console.error('Paper not found in priorities');
        return;
    }

    const oldSearchTerm = paperData.searchTerm || 'Unknown';
    const newSearchTerm = prompt('Edit search term for this paper:', oldSearchTerm);

    if (newSearchTerm === null || newSearchTerm.trim() === '') {
        return;
    }

    const trimmedNewSearchTerm = newSearchTerm.trim();

    if (trimmedNewSearchTerm === oldSearchTerm) {
        return;
    }

    updatePaperSearchTerm(paperId, trimmedNewSearchTerm);
    loadInterestingPapers();
}

/**
 * Switch to a different session in the interesting papers tab.
 */
export function switchInterestingSession(session: string): void {
    setCurrentInterestingSession(session);
    loadInterestingPapers();
}

/**
 * Change the sort order for interesting papers.
 */
export function changeInterestingPapersSortOrder(sortOrder: string): void {
    setInterestingPapersSortOrder(sortOrder);
    loadInterestingPapers();
}

/**
 * Donate interesting papers data for validation purposes.
 */
export async function donateInterestingPapersData(): Promise<void> {
    if (window.readOnly) {
        alert('Data donation is unavailable in read-only mode.');
        return;
    }

    if (Object.keys(getAllPaperPriorities()).length === 0) {
        alert('No papers rated yet. Rate some papers before donating data.');
        return;
    }

    if (hasDataBeenDonated()) {
        alert('You have already donated this data. Thank you for your contribution!');
        return;
    }

    if (!confirm(DONATION_CONFIRMATION_MESSAGE)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/donate-data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                paperPriorities: getAllPaperPriorities()
            })
        });

        const result = await parseJsonResponse(response);

        markDataAsDonated();
        updateControlsVisibility();

        alert(`✅ ${result.message}\n\nThank you for helping improve our service!`);
        console.log('Successfully donated data:', result);
    } catch (error) {
        console.error('Error donating data:', error);
        alert('❌ Error donating data: ' + (error as Error).message + '\n\nPlease try again later.');
    }
}
