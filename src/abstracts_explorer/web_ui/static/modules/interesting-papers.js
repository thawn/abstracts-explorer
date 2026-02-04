/**
 * Interesting Papers Module
 * 
 * Handles the display, management, and export of rated/interesting papers.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml } from './utils/dom-utils.js';
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
    loadPriorities as loadPrioritiesFromState,
    paperPriorities
} from './state.js';
import { formatPaperCard } from './paper-card.js';

/**
 * Track whether donation prompt was shown in this session to avoid being intrusive
 */
let donationPromptShownThisSession = false;

/**
 * Standard donation confirmation message
 */
const DONATION_CONFIRMATION_MESSAGE = 
    'Would you like to donate your paper ratings to help improve our service?\n\n' +
    '✓ Your data will be fully anonymized\n' +
    '✓ No personal information will be collected\n' +
    '✓ Data will only be used to improve paper recommendations\n' +
    '✓ You can donate as many times as you like\n\n' +
    'Thank you for contributing to research!';

/**
 * Load and display interesting papers
 * @async
 */
export async function loadInterestingPapers() {
    const listDiv = document.getElementById('interesting-papers-list');
    const tabsContainer = document.getElementById('interesting-session-tabs');
    const tabsNav = document.getElementById('interesting-session-tabs-nav');

    // If no rated papers, show empty state and hide tabs
    if (Object.keys(getAllPaperPriorities()).length === 0) {
        listDiv.innerHTML = renderEmptyState(
            'No papers rated yet',
            'Rate papers using the stars to add them here',
            'fa-star'
        );
        tabsContainer.classList.add('hidden');

        // Update count to show 0
        const countElement = document.getElementById('interesting-count');
        if (countElement) {
            countElement.textContent = '0';
        }
        return;
    }

    // Show tabs container
    tabsContainer.classList.remove('hidden');

    // Show loading
    listDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        // Fetch details for all rated papers
        const paperIds = getPaperIdsWithPriorities();
        const response = await fetch(`${API_BASE}/api/papers/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ paper_ids: paperIds })
        });

        const data = await response.json();

        if (data.error) {
            listDiv.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-6">
                    <p class="text-red-700">${escapeHtml(data.error)}</p>
                </div>
            `;
            return;
        }

        displayInterestingPapers(data.papers);
    } catch (error) {
        console.error('Error loading interesting papers:', error);
        listDiv.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-6">
                <p class="text-red-700">Error loading papers. Please try again.</p>
            </div>
        `;
    }
}

/**
 * Display interesting papers grouped by search term
 * @param {Array} papers - Array of paper objects
 */
export function displayInterestingPapers(papers) {
    const listDiv = document.getElementById('interesting-papers-list');
    const tabsNav = document.getElementById('interesting-session-tabs-nav');

    // Get selected year and conference from header
    const yearSelect = document.getElementById('year-selector');
    const conferenceSelect = document.getElementById('conference-selector');
    const selectedYear = yearSelect ? yearSelect.value : '';
    const selectedConference = conferenceSelect ? conferenceSelect.value : '';

    // Filter papers by year and conference
    let filteredPapers = papers;
    if (selectedYear) {
        filteredPapers = filteredPapers.filter(paper => String(paper.year) === String(selectedYear));
    }
    if (selectedConference) {
        filteredPapers = filteredPapers.filter(paper => paper.conference === selectedConference);
    }

    // If no papers match the filter, show empty state
    if (filteredPapers.length === 0) {
        listDiv.innerHTML = renderEmptyState(
            'No papers match the selected filters',
            'Try selecting a different conference or year',
            'fa-filter'
        );
        const tabsContainer = document.getElementById('interesting-session-tabs');
        tabsContainer.classList.add('hidden');

        // Update count to show 0 filtered papers
        const countElement = document.getElementById('interesting-count');
        if (countElement) {
            countElement.textContent = '0';
        }
        return;
    }

    // Add priority and search term to each paper
    const priorities = getAllPaperPriorities();
    filteredPapers.forEach(paper => {
        const paperData = priorities[paper.uid];
        paper.priority = paperData?.priority || 0;
        paper.searchTerm = paperData?.searchTerm || 'Unknown';
    });

    // Sort papers based on the selected sort order
    const sortOrder = getInterestingPapersSortOrder();
    filteredPapers.sort((a, b) => {
        // First by session (always)
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;

        // Then apply the selected sort order
        if (sortOrder === 'search-rating-poster') {
            const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
            if (searchTermCompare !== 0) return searchTermCompare;
            
            if (a.priority !== b.priority) return b.priority - a.priority;
            
            return naturalSortPosterPosition(a.poster_position, b.poster_position);
        } else if (sortOrder === 'rating-poster-search') {
            if (a.priority !== b.priority) return b.priority - a.priority;
            
            const posterCompare = naturalSortPosterPosition(a.poster_position, b.poster_position);
            if (posterCompare !== 0) return posterCompare;
            
            return (a.searchTerm || '').localeCompare(b.searchTerm || '');
        } else if (sortOrder === 'poster-search-rating') {
            const posterCompare = naturalSortPosterPosition(a.poster_position, b.poster_position);
            if (posterCompare !== 0) return posterCompare;
            
            const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
            if (searchTermCompare !== 0) return searchTermCompare;
            
            return b.priority - a.priority;
        }
    });

    // Group by session, then by grouping key based on sort order
    const groupedBySession = {};
    filteredPapers.forEach(paper => {
        const session = paper.session || 'No Session';
        if (!groupedBySession[session]) {
            groupedBySession[session] = {};
        }
        
        // Determine the grouping key based on sort order
        let groupKey;
        if (sortOrder === 'search-rating-poster') {
            groupKey = paper.searchTerm || 'Unknown';
        } else if (sortOrder === 'rating-poster-search') {
            groupKey = `${paper.priority} ${paper.priority === 1 ? 'star' : 'stars'}`;
        } else if (sortOrder === 'poster-search-rating') {
            groupKey = 'All Papers';
        }
        
        if (!groupedBySession[session][groupKey]) {
            groupedBySession[session][groupKey] = [];
        }
        groupedBySession[session][groupKey].push(paper);
    });

    // Get all sessions
    const sessions = Object.keys(groupedBySession).sort();

    // Set default session if not set or if the current session no longer exists
    let currentSession = getCurrentInterestingSession();
    if (!currentSession || !sessions.includes(currentSession)) {
        currentSession = sessions[0] || null;
        setCurrentInterestingSession(currentSession);
    }

    // Generate session tabs
    let tabsHtml = '';
    sessions.forEach(session => {
        const isActive = session === currentSession;
        const activeClass = isActive ? 'border-b-2 border-purple-600 text-purple-600' : 'text-gray-600 hover:text-gray-800 hover:border-gray-300';
        tabsHtml += `
            <button 
                onclick="switchInterestingSession('${escapeHtml(session).replace(/'/g, "\\'")}')"
                class="px-6 py-3 text-sm font-medium ${activeClass} border-b-2 border-transparent focus:outline-none transition-colors whitespace-nowrap">
                <i class="fas fa-calendar-alt mr-2"></i>${escapeHtml(session)}
            </button>
        `;
    });
    tabsNav.innerHTML = tabsHtml;

    // Generate HTML for the selected session only
    let html = '';

    if (currentSession && groupedBySession[currentSession]) {
        const groups = groupedBySession[currentSession];

        // Determine icon based on sort order
        let groupIcon = 'fa-search';
        if (sortOrder === 'rating-poster-search') {
            groupIcon = 'fa-star';
        } else if (sortOrder === 'poster-search-rating') {
            groupIcon = 'fa-list';
        }

        // Sort group keys appropriately
        let sortedGroupKeys = Object.keys(groups);
        if (sortOrder === 'search-rating-poster') {
            sortedGroupKeys.sort();
        } else if (sortOrder === 'rating-poster-search') {
            sortedGroupKeys.sort((a, b) => {
                const starsA = parseInt(a.split(' ')[0]);
                const starsB = parseInt(b.split(' ')[0]);
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
                    ? `<button onclick="editSearchTerm('${escapeHtml(currentSession).replace(/'/g, "\\'")}', '${escapeHtml(groupKey).replace(/'/g, "\\'")}', event)" 
                              class="ml-2 text-sm text-purple-600 hover:text-purple-800 focus:outline-none" 
                              title="Edit search term">
                           <i class="fas fa-edit"></i>
                       </button>`
                    : '';

                html += `
                    <div class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-md p-5 mb-4">
                        <h3 class="text-lg font-bold text-gray-800 mb-4 border-b border-blue-200 pb-2 flex items-center justify-between">
                            <span>
                                <i class="fas ${groupIcon} text-blue-600 mr-2"></i>${escapeHtml(groupKey)}
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

    // Update the count in the tab heading to show filtered papers
    const countElement = document.getElementById('interesting-count');
    if (countElement) {
        countElement.textContent = filteredPapers.length;
    }
}

/**
 * Save interesting papers as markdown (zip export)
 * @param {Event} event - Click event
 * @async
 */
export async function saveInterestingPapersAsMarkdown(event) {
    if (Object.keys(getAllPaperPriorities()).length === 0) {
        alert('No papers rated yet. Rate some papers before saving.');
        return;
    }

    // Show data donation prompt before export (only once per session to avoid being intrusive)
    if (!donationPromptShownThisSession) {
        const donateMessage = 
            '💡 Before you export, would you like to donate your ratings to help improve our service?\n\n' +
            '✓ Fully anonymized - no personal data collected\n' +
            '✓ Used only to improve recommendations\n' +
            '✓ Takes just a moment\n\n' +
            'Click "OK" to donate now (recommended)\n' +
            'Click "Cancel" to skip and continue with export';
        
        if (confirm(donateMessage)) {
            // User wants to donate - call donation function
            await donateInterestingPapersData();
        }
        
        // Mark that we've shown the prompt this session
        donationPromptShownThisSession = true;
    }

    const button = event.target;
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Preparing download...';
    button.disabled = true;

    try {
        const paperIds = getPaperIdsWithPriorities();
        const searchInput = document.getElementById('search-input');
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
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate export');
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
        alert('Error creating export: ' + error.message);
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

/**
 * Save interesting papers data as JSON file
 */
export function saveInterestingPapersAsJSON() {
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
        alert('Error saving JSON file: ' + error.message);
    }
}

/**
 * Trigger file input for loading JSON
 */
export function loadInterestingPapersFromJSON() {
    const fileInput = document.getElementById('json-file-input');
    if (fileInput) {
        fileInput.click();
    }
}

/**
 * Handle JSON file load
 * @param {Event} event - File input change event
 */
export function handleJSONFileLoad(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.json')) {
        alert('Please select a JSON file.');
        event.target.value = '';
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        try {
            const importData = JSON.parse(e.target.result);

            if (!importData.paperPriorities || typeof importData.paperPriorities !== 'object') {
                throw new Error('Invalid JSON format: missing or invalid paperPriorities');
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
                event.target.value = '';
                return;
            }

            let newCount = 0;
            let updatedCount = 0;

            for (const [paperId, data] of Object.entries(importData.paperPriorities)) {
                if (paperPriorities[paperId]) {
                    updatedCount++;
                } else {
                    paperPriorities[paperId] = data;
                    newCount++;
                }
            }

            if (importData.sortOrder && !localStorage.getItem('interestingPapersSortOrder')) {
                setInterestingPapersSortOrder(importData.sortOrder);
                const sortSelect = document.getElementById('sort-order');
                if (sortSelect) {
                    sortSelect.value = importData.sortOrder;
                }
            }

            savePriorities();

            // Trigger update of interesting papers count
            if (window.updateInterestingPapersCount) {
                window.updateInterestingPapersCount();
            }

            // Reload interesting papers if on that tab
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
            alert('Error loading JSON file: ' + error.message + '\n\nPlease ensure the file is a valid interesting papers export.');
        }

        event.target.value = '';
    };

    reader.onerror = function () {
        alert('Error reading file. Please try again.');
        event.target.value = '';
    };

    reader.readAsText(file);
}

/**
 * Edit search term for a group of papers
 * @param {string} session - Session name
 * @param {string} oldSearchTerm - Old search term
 * @param {Event} event - Click event
 */
export function editSearchTerm(session, oldSearchTerm, event) {
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
 * Edit search term for a single paper
 * @param {string} paperId - Paper ID
 * @param {Event} event - Click event
 */
export function editPaperSearchTerm(paperId, event) {
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
 * Switch to a different session in the interesting papers tab
 * @param {string} session - Session name
 */
export function switchInterestingSession(session) {
    setCurrentInterestingSession(session);
    loadInterestingPapers();
}

/**
 * Change the sort order for interesting papers
 * @param {string} sortOrder - New sort order
 */
export function changeInterestingPapersSortOrder(sortOrder) {
    setInterestingPapersSortOrder(sortOrder);
    loadInterestingPapers();
}

/**
 * Generate markdown content for interesting papers (legacy function for compatibility)
 * @param {Array} papers - Array of paper objects
 * @returns {string} Markdown content
 */
export function generateInterestingPapersMarkdown(papers) {
    const priorities = getAllPaperPriorities();
    
    papers.forEach(paper => {
        paper.priority = priorities[paper.uid]?.priority || 0;
    });

    papers.sort((a, b) => {
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;
        if (a.priority !== b.priority) return b.priority - a.priority;
        return naturalSortPosterPosition(a.poster_position, b.poster_position);
    });

    const searchInput = document.getElementById('search-input');
    const searchQuery = searchInput ? searchInput.value : '';

    let markdown = `# Interesting Papers from NeurIPS 2025\n\n`;
    markdown += `Generated: ${new Date().toLocaleString()}\n\n`;

    if (searchQuery) {
        markdown += `## Search Context\n\n`;
        markdown += `**Search Query:** ${searchQuery}\n\n`;
    }

    markdown += `**Total Papers:** ${papers.length}\n\n`;
    markdown += `---\n\n`;

    const groupedBySession = {};
    papers.forEach(paper => {
        const session = paper.session || 'No Session';
        if (!groupedBySession[session]) {
            groupedBySession[session] = [];
        }
        groupedBySession[session].push(paper);
    });

    for (const [session, sessionPapers] of Object.entries(groupedBySession)) {
        markdown += `## ${session}\n\n`;

        sessionPapers.forEach(paper => {
            const stars = '⭐'.repeat(paper.priority);
            markdown += `### ${paper.title || 'Untitled'}\n\n`;
            markdown += `**Rating:** ${stars} (${paper.priority}/5)\n\n`;

            if (paper.authors && paper.authors.length > 0) {
                markdown += `**Authors:** ${paper.authors.join(', ')}\n\n`;
            }

            if (paper.poster_position) {
                markdown += `**Poster:** ${paper.poster_position}\n\n`;
            }

            if (paper.paper_url) {
                markdown += `**Paper URL:** ${paper.paper_url}\n\n`;
            }

            if (paper.url) {
                markdown += `**Source URL:** ${paper.url}\n\n`;
            }

            if (paper.abstract) {
                markdown += `**Abstract:**\n\n${paper.abstract}\n\n`;
            }

            markdown += `---\n\n`;
        });
    }

    return markdown;
}

/**
 * Donate interesting papers data for validation purposes
 * @async
 */
export async function donateInterestingPapersData() {
    if (Object.keys(getAllPaperPriorities()).length === 0) {
        alert('No papers rated yet. Rate some papers before donating data.');
        return;
    }

    // Show confirmation dialog with information about data anonymization
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

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to donate data');
        }

        alert(`✅ ${result.message}\n\nThank you for helping improve our service!`);
        console.log('Successfully donated data:', result);
    } catch (error) {
        console.error('Error donating data:', error);
        alert('❌ Error donating data: ' + error.message + '\n\nPlease try again later.');
    }
}
