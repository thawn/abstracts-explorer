/**
 * Paper Card Module
 * 
 * Handles formatting and display of paper cards, paper details modal,
 * and star ratings.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { showError } from './utils/ui-utils.js';
import { renderMarkdownWithLatex, renderInlineMarkdownWithLatex } from './utils/markdown-utils.js';
import {
    getPaperPriority,
    setPaperPriority as setPaperPriorityInState,
    getPaperIdsWithPriorities,
    getAllPaperPriorities,
    getCurrentTab
} from './state.js';

/**
 * Build URL badges HTML for a paper
 * @param {Object} paper - Paper object with optional url, paper_pdf_url, poster_image_url
 * @param {boolean} compact - Whether to use compact styling
 * @param {boolean} stopPropagation - Whether to include onclick="event.stopPropagation()"
 * @returns {string} HTML string (empty string if no URLs available)
 */
export function buildUrlBadges(paper, compact = false, stopPropagation = true) {
    const stop = stopPropagation ? ' onclick="event.stopPropagation()"' : '';
    const size = compact ? 'px-2 py-0.5 text-xs' : 'px-2 py-1 text-xs';
    let badges = '';
    if (paper.url) {
        badges += `<a href="${escapeHtml(paper.url)}" target="_blank"${stop} class="${size} bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-full hover:bg-purple-200 dark:hover:bg-purple-900 inline-flex items-center gap-1"><i class="fas fa-external-link-alt"></i>Paper Page</a>`;
    }
    if (paper.paper_pdf_url) {
        badges += `<a href="${escapeHtml(paper.paper_pdf_url)}" target="_blank"${stop} class="${size} bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-full hover:bg-blue-200 dark:hover:bg-blue-900 inline-flex items-center gap-1"><i class="fas fa-file-pdf"></i>PDF</a>`;
    }
    if (paper.poster_image_url) {
        badges += `<a href="${escapeHtml(paper.poster_image_url)}" target="_blank"${stop} class="${size} bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full hover:bg-green-200 dark:hover:bg-green-900 inline-flex items-center gap-1"><i class="fas fa-image"></i>Poster</a>`;
    }
    return badges ? `<div class="flex flex-wrap gap-1 ${compact ? 'mb-2' : 'mb-3'}">${badges}</div>` : '';
}

/**
 * Format a single paper card
 * @param {Object} paper - Paper object
 * @param {Object} options - Display options
 * @returns {string} HTML string
 */
export function formatPaperCard(paper, options = {}) {
    const {
        compact = false,
        showNumber = null,
        abstractLength = compact ? 200 : 300,
        idPrefix = '',
        showSearchTerm = false
    } = options;

    const title = paper.title || 'Untitled';

    // Validate authors is an array
    if (paper.authors && !Array.isArray(paper.authors)) {
        throw new TypeError(`Expected authors to be an array, got ${typeof paper.authors}`);
    }

    const authors = (paper.authors && paper.authors.length > 0)
        ? paper.authors.join(', ')
        : 'Unknown';

    // Build abstract with collapsible details if needed
    let abstractHtml = '';
    if (paper.abstract) {
        if (paper.abstract.length > abstractLength) {
            const preview = paper.abstract.substring(0, abstractLength);
            abstractHtml = `
                <details class="abstract-details text-gray-700 dark:text-gray-300 ${compact ? 'text-xs' : 'text-sm'} leading-relaxed ${compact ? 'mt-2' : ''} markdown-content" onclick="event.stopPropagation()">
                    <summary class="cursor-pointer">
                        <span class="abstract-preview">
                            ${renderMarkdownWithLatex(preview)}... <span class="text-purple-600 dark:text-purple-400 font-medium hover:text-purple-800 dark:hover:text-purple-300">Show more</span>
                        </span>
                        <span class="abstract-full">
                            ${renderMarkdownWithLatex(paper.abstract)} <span class="text-purple-600 dark:text-purple-400 font-medium hover:text-purple-800 dark:hover:text-purple-300">Show less</span>
                        </span>
                    </summary>
                </details>
            `;
        } else {
            abstractHtml = `<div class="text-gray-700 dark:text-gray-300 ${compact ? 'text-xs' : 'text-sm'} leading-relaxed ${compact ? 'mt-2' : ''} markdown-content">${renderMarkdownWithLatex(paper.abstract)}</div>`;
        }
    } else if (!compact) {
        abstractHtml = `<p class="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">No abstract available</p>`;
    }

    // Build metadata badges
    let metadata = '';
    if (paper.conference) {
        metadata += `<span class="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 text-xs rounded-full mr-${compact ? '1' : '2'}"><i class="fas fa-university mr-1"></i>${escapeHtml(paper.conference)}</span>`;
    }
    if (paper.session) {
        metadata += `<span class="px-2 py-1 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 text-xs rounded-full mr-${compact ? '1' : '2'}"><i class="fas fa-calendar-alt mr-1"></i>${escapeHtml(paper.session)}</span>`;
    }
    if (paper.poster_position) {
        metadata += `<span class="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 text-xs rounded-full mr-${compact ? '1' : '2'}"><i class="fas fa-map-pin mr-1"></i>Poster ${escapeHtml(paper.poster_position)}</span>`;
    }
    if (showSearchTerm && paper.searchTerm) {
        metadata += `<span class="px-2 py-1 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 text-xs rounded-full mr-${compact ? '1' : '2'} inline-flex items-center">
            <i class="fas fa-search mr-1"></i>${escapeHtml(paper.searchTerm)}
            <button onclick="editPaperSearchTerm(${paper.uid}, event)" class="ml-1 text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-200 focus:outline-none" title="Edit search term">
                <i class="fas fa-edit text-xs"></i>
            </button>
        </span>`;
    }
    if (paper.distance !== undefined) {
        metadata += `<span class="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-xs rounded-full" title="Similarity score: how closely this paper matches your search query (-1 = no match, 1 = perfect match)"><i class="fas fa-chart-line mr-1"></i>${(1 - paper.distance).toFixed(compact ? 2 : 3)}</span>`;
    }

    const cardId = idPrefix ? `id="${idPrefix}"` : '';
    const cardClasses = compact
        ? 'paper-card bg-white dark:bg-gray-800 rounded-lg shadow-sm p-3 hover:shadow-md cursor-pointer border border-gray-200 dark:border-gray-700'
        : 'paper-card bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg cursor-pointer';

    // Get current priority for this paper
    const currentPriority = getPaperPriority(paper.uid);

    // Generate star rating HTML
    let starsHtml = '<div class="flex-shrink-0 ml-2 flex items-center gap-0.5" onclick="event.stopPropagation()" title="Rate this paper">';
    for (let i = 1; i <= 5; i++) {
        const isSelected = i <= currentPriority;
        const starClass = isSelected
            ? 'fas fa-star text-yellow-400 hover:text-yellow-500'
            : 'far fa-star text-gray-300 dark:text-gray-600 hover:text-yellow-400';
        starsHtml += `<i class="${starClass} cursor-pointer text-${compact ? 'sm' : 'base'} paper-star" data-paper-id="${paper.uid}" onclick="setPaperPriority('${paper.uid}', ${i})"></i>`;
    }
    starsHtml += '</div>';

    return `
        <div ${cardId} class="${cardClasses}" onclick="showPaperDetails('${paper.uid}')">
            ${showNumber !== null ? `
                <div class="flex items-start justify-between mb-1">
                    <span class="text-xs font-semibold text-purple-600">#${showNumber}</span>
                </div>
            ` : ''}
            <div class="flex items-start justify-between ${compact ? 'mb-1' : 'mb-2'}">
                <h${compact ? '4' : '3'} class="${compact ? 'text-sm' : 'text-lg'} font-semibold text-gray-800 dark:text-gray-100 flex-1 ${compact ? 'leading-tight pr-2' : 'pr-2'}">${renderInlineMarkdownWithLatex(title)}</h${compact ? '4' : '3'}>
                ${starsHtml}
            </div>
            <p class="${compact ? 'text-xs' : 'text-sm'} text-gray-600 dark:text-gray-400 ${compact ? 'mb-1 truncate' : 'mb-2'}">
                <i class="fas fa-users mr-1"></i>${escapeHtml(authors)}
            </p>
            ${buildUrlBadges(paper, compact)}
            ${metadata ? `<div class="${compact ? 'mb-2' : 'mb-3'}">${metadata}</div>` : ''}
            ${abstractHtml}
            ${!compact && paper.distance !== undefined && !metadata.includes('chart-line') ? `
                <div class="mt-3 pt-3 border-t dark:border-gray-700">
                    <span class="text-xs text-gray-500 dark:text-gray-400">
                        <i class="fas fa-chart-line mr-1"></i>
                        Relevance: ${(1 - paper.distance).toFixed(3)}
                    </span>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Show paper details in modal
 * @param {string} paperId - Paper ID
 * @async
 */
export async function showPaperDetails(paperId) {
    try {
        const response = await fetch(`${API_BASE}/api/paper/${paperId}`);
        const paper = await response.json();

        if (paper.error) {
            showError(paper.error);
            return;
        }

        // Create modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50';
        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };

        // Validate authors is an array
        if (paper.authors && !Array.isArray(paper.authors)) {
            throw new TypeError(`Expected authors to be an array, got ${typeof paper.authors}`);
        }

        const authors = (paper.authors && paper.authors.length > 0)
            ? paper.authors.join(', ')
            : 'Unknown';
        const title = paper.title || 'Untitled';

        modal.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto p-8">
                <div class="flex items-start justify-between mb-4">
                    <h2 class="text-2xl font-bold text-gray-800 dark:text-gray-100 flex-1">${renderInlineMarkdownWithLatex(title)}</h2>
                    <button onclick="this.closest('.fixed').remove()" class="ml-4 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                
                <div class="mb-4 flex flex-wrap gap-2">
                    ${paper.conference ? `
                        <span class="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 rounded-full text-sm">
                            <i class="fas fa-university mr-1"></i>${escapeHtml(paper.conference)}
                        </span>
                    ` : ''}
                    ${paper.session ? `
                        <span class="px-3 py-1 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full text-sm">
                            <i class="fas fa-calendar-alt mr-1"></i>${escapeHtml(paper.session)}
                        </span>
                    ` : ''}
                    ${paper.poster_position ? `
                        <span class="px-3 py-1 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 rounded-full text-sm">
                            <i class="fas fa-map-pin mr-1"></i>Poster ${escapeHtml(paper.poster_position)}
                        </span>
                    ` : ''}
                    <span class="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full text-sm">
                        <i class="fas fa-fingerprint mr-1"></i>ID: ${paper.uid}
                    </span>
                </div>
                
                <div class="mb-6">
                    <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        <i class="fas fa-users mr-2"></i>Authors
                    </h3>
                    <p class="text-gray-700 dark:text-gray-300 mb-3">${escapeHtml(authors)}</p>
                    ${buildUrlBadges(paper, false, false)}
                </div>
                
                <div class="mb-6">
                    <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        <i class="fas fa-file-alt mr-2"></i>Abstract
                    </h3>
                    <div class="text-gray-700 dark:text-gray-300 leading-relaxed markdown-content">${paper.abstract ? renderMarkdownWithLatex(paper.abstract) : 'No abstract available'}</div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    } catch (error) {
        console.error('Error loading paper details:', error);
        showError('Error loading paper details');
    }
}

/**
 * Update star rating display for a paper
 * @param {string} paperId - Paper ID
 */
export function updateStarDisplay(paperId) {
    const priority = getPaperPriority(paperId);

    // Find all star elements for this paper anywhere on the page using the paper-star class
    const allStars = document.querySelectorAll(`.paper-star[data-paper-id="${paperId}"]`);
    
    allStars.forEach((star) => {
        // Determine which star number this is by looking at its onclick attribute
        const onclickMatch = star.getAttribute('onclick').match(/setPaperPriority\('[^']+',\s*(\d+)\)/);
        if (!onclickMatch) return;
        
        const starNumber = parseInt(onclickMatch[1], 10);
        
        if (starNumber <= priority) {
            // Set to filled star
            star.classList.remove('far', 'text-gray-300', 'dark:text-gray-600');
            star.classList.add('fas', 'text-yellow-400');
            // Update hover state
            star.classList.remove('hover:text-yellow-400');
            star.classList.add('hover:text-yellow-500');
        } else {
            // Set to empty star
            star.classList.remove('fas', 'text-yellow-400', 'hover:text-yellow-500');
            star.classList.add('far', 'text-gray-300', 'dark:text-gray-600', 'hover:text-yellow-400');
        }
    });
}

/**
 * Update interesting papers count in tab
 * @async
 */
export async function updateInterestingPapersCount() {
    const countElement = document.getElementById('interesting-count');
    if (!countElement) {
        return;
    }

    const paperIds = getPaperIdsWithPriorities();
    if (paperIds.length === 0) {
        countElement.textContent = '0';
        return;
    }

    // Get selected year and conference from header
    const selectedYears = getSelectedYears();
    const selectedConference = getSelectedConference();

    // If no filters, show total count
    if (selectedYears.length === 0 && !selectedConference) {
        countElement.textContent = paperIds.length;
        return;
    }

    // Fetch papers to filter by year/conference
    try {
        const response = await fetch(`${API_BASE}/api/papers/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ paper_ids: paperIds })
        });

        const data = await response.json();

        if (data.error || !data.papers) {
            countElement.textContent = paperIds.length;
            return;
        }

        let filteredPapers = data.papers;
        if (selectedYears.length > 0) {
            filteredPapers = filteredPapers.filter(paper => selectedYears.includes(Number(paper.year)));
        }
        if (selectedConference) {
            filteredPapers = filteredPapers.filter(paper => paper.conference === selectedConference);
        }

        countElement.textContent = filteredPapers.length;
    } catch (error) {
        console.error('Error updating interesting papers count:', error);
        countElement.textContent = paperIds.length;
    }
}

/**
 * Set paper priority (wrapper function called from HTML)
 * @param {string} paperId - Paper ID
 * @param {number} priority - Priority (1-5)
 */
export function setPaperPriority(paperId, priority) {
    // Stop event propagation to prevent opening paper details
    event.stopPropagation();

    setPaperPriorityInState(paperId, priority);

    // Update the stars display
    updateStarDisplay(paperId);

    // Update interesting papers count
    updateInterestingPapersCount();

    // Refresh interesting papers tab if it's currently visible
    const currentTab = getCurrentTab();
    if (currentTab === 'interesting') {
        if (window.loadInterestingPapers) {
            window.loadInterestingPapers();
        }
    }
}
