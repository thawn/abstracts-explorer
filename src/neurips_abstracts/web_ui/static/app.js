// API base URL
const API_BASE = '';

// State
let currentTab = 'search';
let chatHistory = [];
let paperPriorities = {}; // Store paper priorities: { paperId: { priority: number, searchTerm: string } }
let currentSearchTerm = ''; // Track the current search term
let currentInterestingSession = null; // Track the current selected session in interesting papers tab
let interestingPapersSortOrder = 'search-rating-poster'; // Sort order: 'search-rating-poster', 'rating-poster-search', or 'poster-search-rating'

// Utility: Natural sort comparison for poster positions
// Handles strings like "Board 123" or "123" correctly (99 < 100 < 1000)
function naturalSortPosterPosition(a, b) {
    const aPos = a || '';
    const bPos = b || '';
    
    // Extract numbers from the strings
    const aMatch = aPos.match(/\d+/);
    const bMatch = bPos.match(/\d+/);
    
    if (aMatch && bMatch) {
        const aNum = parseInt(aMatch[0], 10);
        const bNum = parseInt(bMatch[0], 10);
        
        if (aNum !== bNum) {
            return aNum - bNum;
        }
    }
    
    // Fallback to string comparison if no numbers or numbers are equal
    return aPos.localeCompare(bPos);
}

// Initialize app
document.addEventListener('DOMContentLoaded', function () {
    loadStats();
    loadFilterOptions();
    loadPriorities();
    updateDownloadButton();

    // Close modal when clicking outside of it
    document.getElementById('settings-modal').addEventListener('click', function (event) {
        if (event.target === this) {
            closeSettings();
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            const modal = document.getElementById('settings-modal');
            if (!modal.classList.contains('hidden')) {
                closeSettings();
            }
        }
    });
});

// Open search settings modal
function openSearchSettings() {
    // Sync modal filters with search filters
    syncFiltersToModal('search');

    // Update modal title
    document.getElementById('modal-title').textContent = 'Search Settings';

    // Show search-specific settings, hide chat-specific
    document.getElementById('search-settings-section').classList.remove('hidden');
    document.getElementById('chat-settings-section').classList.add('hidden');

    // Open modal
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Store context
    modal.dataset.context = 'search';
}

// Open chat settings modal
function openChatSettings() {
    // Sync modal filters with chat filters
    syncFiltersToModal('chat');

    // Update modal title
    document.getElementById('modal-title').textContent = 'Chat Settings';

    // Show chat-specific settings, hide search-specific
    document.getElementById('search-settings-section').classList.add('hidden');
    document.getElementById('chat-settings-section').classList.remove('hidden');

    // Open modal
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Store context
    modal.dataset.context = 'chat';
}

// Close settings modal
function closeSettings() {
    const modal = document.getElementById('settings-modal');
    const context = modal.dataset.context;

    // Sync modal filters back to the appropriate context
    if (context) {
        syncFiltersFromModal(context);
    }

    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

// Sync filters from search/chat to modal
function syncFiltersToModal(context) {
    const prefix = context === 'search' ? '' : 'chat-';

    // Sync sessions
    const sessionFilter = document.getElementById(prefix + 'session-filter');
    const modalSessionFilter = document.getElementById('modal-session-filter');
    if (sessionFilter && modalSessionFilter) {
        // Copy options if modal is empty
        if (modalSessionFilter.options.length === 0) {
            modalSessionFilter.innerHTML = sessionFilter.innerHTML;
        }
        // Copy selections
        Array.from(modalSessionFilter.options).forEach((opt, idx) => {
            opt.selected = sessionFilter.options[idx]?.selected || false;
        });
    }
}

// Sync filters from modal back to search/chat
function syncFiltersFromModal(context) {
    const prefix = context === 'search' ? '' : 'chat-';

    // Sync sessions
    const sessionFilter = document.getElementById(prefix + 'session-filter');
    const modalSessionFilter = document.getElementById('modal-session-filter');
    if (sessionFilter && modalSessionFilter) {
        Array.from(sessionFilter.options).forEach((opt, idx) => {
            opt.selected = modalSessionFilter.options[idx]?.selected || false;
        });
    }
}

// Load priorities from localStorage
function loadPriorities() {
    const stored = localStorage.getItem('paperPriorities');
    if (stored) {
        try {
            paperPriorities = JSON.parse(stored);
        } catch (e) {
            console.error('Error loading priorities:', e);
            paperPriorities = {};
        }
    }
    
    // Load sort order preference
    const storedSortOrder = localStorage.getItem('interestingPapersSortOrder');
    if (storedSortOrder) {
        interestingPapersSortOrder = storedSortOrder;
    }
    
    // Update the count in the tab
    updateInterestingPapersCount();
}

// Save priorities to localStorage
function savePriorities() {
    try {
        localStorage.setItem('paperPriorities', JSON.stringify(paperPriorities));
    } catch (e) {
        console.error('Error saving priorities:', e);
    }
}

// Set paper priority
function setPaperPriority(paperId, priority) {
    // Stop event propagation to prevent opening paper details
    event.stopPropagation();

    const currentPriority = paperPriorities[paperId]?.priority || 0;
    const existingSearchTerm = paperPriorities[paperId]?.searchTerm;

    // If clicking the same star, remove the rating
    if (currentPriority === priority) {
        delete paperPriorities[paperId];
    } else if (priority === 0) {
        // Remove priority if set to 0
        delete paperPriorities[paperId];
    } else {
        paperPriorities[paperId] = {
            priority: priority,
            // Preserve the existing search term when updating, only use currentSearchTerm for new ratings
            searchTerm: existingSearchTerm || currentSearchTerm || 'Unknown'
        };
    }
    savePriorities();

    // Update the stars display
    updateStarDisplay(paperId);

    // Update interesting papers count
    updateInterestingPapersCount();

    // Refresh interesting papers tab if it's currently visible
    if (currentTab === 'interesting') {
        loadInterestingPapers();
    }
}

// Update star rating display
function updateStarDisplay(paperId) {
    const priority = paperPriorities[paperId]?.priority || 0;

    // Find all star elements for this paper
    const paperCard = document.querySelector(`[onclick*="showPaperDetails('${paperId}')"]`);
    if (paperCard) {
        const stars = paperCard.querySelectorAll('i[class*="fa-star"]');
        stars.forEach((star, index) => {
            const starNumber = index + 1;
            if (starNumber <= priority) {
                // Filled star
                star.className = star.className.replace('far fa-star text-gray-300', 'fas fa-star text-yellow-400');
                star.className = star.className.replace('hover:text-yellow-400', 'hover:text-yellow-500');
            } else {
                // Empty star
                star.className = star.className.replace('fas fa-star text-yellow-400', 'far fa-star text-gray-300');
                star.className = star.className.replace('hover:text-yellow-500', 'hover:text-yellow-400');
            }
        });
    }
}

// Update interesting papers count in tab
async function updateInterestingPapersCount() {
    const countElement = document.getElementById('interesting-count');
    if (!countElement) {
        return;
    }

    // If no rated papers, set count to 0
    const paperIds = Object.keys(paperPriorities);
    if (paperIds.length === 0) {
        countElement.textContent = '0';
        return;
    }

    // Get selected year and conference from header
    const yearSelect = document.getElementById('year-selector');
    const conferenceSelect = document.getElementById('conference-selector');
    const selectedYear = yearSelect ? yearSelect.value : '';
    const selectedConference = conferenceSelect ? conferenceSelect.value : '';

    // If no filters are selected, just show total count
    if (!selectedYear && !selectedConference) {
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
            // On error, show total count
            countElement.textContent = paperIds.length;
            return;
        }

        // Filter papers by year and conference
        let filteredPapers = data.papers;
        if (selectedYear) {
            filteredPapers = filteredPapers.filter(paper => String(paper.year) === String(selectedYear));
        }
        if (selectedConference) {
            filteredPapers = filteredPapers.filter(paper => paper.conference === selectedConference);
        }

        countElement.textContent = filteredPapers.length;
    } catch (error) {
        console.error('Error updating interesting papers count:', error);
        // On error, show total count
        countElement.textContent = paperIds.length;
    }
}


// Load and display interesting papers
async function loadInterestingPapers() {
    const listDiv = document.getElementById('interesting-papers-list');
    const tabsContainer = document.getElementById('interesting-session-tabs');
    const tabsNav = document.getElementById('interesting-session-tabs-nav');

    // If no rated papers, show empty state and hide tabs
    if (Object.keys(paperPriorities).length === 0) {
        listDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-star text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">No papers rated yet</p>
                <p class="text-sm">Rate papers using the stars to add them here</p>
            </div>
        `;
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
        const paperIds = Object.keys(paperPriorities);
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

// Display interesting papers grouped by search term
function displayInterestingPapers(papers) {
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
        listDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-filter text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">No papers match the selected filters</p>
                <p class="text-sm">Try selecting a different conference or year</p>
            </div>
        `;
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
    filteredPapers.forEach(paper => {
        const paperData = paperPriorities[paper.uid];
        paper.priority = paperData?.priority || 0;
        paper.searchTerm = paperData?.searchTerm || 'Unknown';
    });

    // Sort papers based on the selected sort order
    filteredPapers.sort((a, b) => {
        // First by session (always)
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;

        // Then apply the selected sort order
        if (interestingPapersSortOrder === 'search-rating-poster') {
            // Search term, then priority, then poster position
            const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
            if (searchTermCompare !== 0) return searchTermCompare;
            
            if (a.priority !== b.priority) return b.priority - a.priority;
            
            return naturalSortPosterPosition(a.poster_position, b.poster_position);
        } else if (interestingPapersSortOrder === 'rating-poster-search') {
            // Priority, then poster position, then search term
            if (a.priority !== b.priority) return b.priority - a.priority;
            
            const posterCompare = naturalSortPosterPosition(a.poster_position, b.poster_position);
            if (posterCompare !== 0) return posterCompare;
            
            return (a.searchTerm || '').localeCompare(b.searchTerm || '');
        } else if (interestingPapersSortOrder === 'poster-search-rating') {
            // Poster position, then search term, then priority
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
        if (interestingPapersSortOrder === 'search-rating-poster') {
            groupKey = paper.searchTerm || 'Unknown';
        } else if (interestingPapersSortOrder === 'rating-poster-search') {
            groupKey = `${paper.priority} ${paper.priority === 1 ? 'star' : 'stars'}`;
        } else if (interestingPapersSortOrder === 'poster-search-rating') {
            groupKey = 'All Papers'; // Don't group by poster, just show all
        }
        
        if (!groupedBySession[session][groupKey]) {
            groupedBySession[session][groupKey] = [];
        }
        groupedBySession[session][groupKey].push(paper);
    });

    // Get all sessions
    const sessions = Object.keys(groupedBySession).sort();

    // Set default session if not set or if the current session no longer exists
    if (!currentInterestingSession || !sessions.includes(currentInterestingSession)) {
        currentInterestingSession = sessions[0] || null;
    }

    // Generate session tabs
    let tabsHtml = '';
    sessions.forEach(session => {
        const isActive = session === currentInterestingSession;
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

    if (currentInterestingSession && groupedBySession[currentInterestingSession]) {
        const groups = groupedBySession[currentInterestingSession];

        // Determine icon based on sort order
        let groupIcon = 'fa-search'; // default for search-rating-poster
        if (interestingPapersSortOrder === 'rating-poster-search') {
            groupIcon = 'fa-star';
        } else if (interestingPapersSortOrder === 'poster-search-rating') {
            groupIcon = 'fa-list'; // Don't show grouping header for poster sort
        }

        // Sort group keys appropriately
        let sortedGroupKeys = Object.keys(groups);
        if (interestingPapersSortOrder === 'search-rating-poster') {
            // Sort alphabetically by search term
            sortedGroupKeys.sort();
        } else if (interestingPapersSortOrder === 'rating-poster-search') {
            // Sort by star count (descending - highest rating first)
            sortedGroupKeys.sort((a, b) => {
                const starsA = parseInt(a.split(' ')[0]);
                const starsB = parseInt(b.split(' ')[0]);
                return starsB - starsA;
            });
        }
        // For poster-search-rating, no sorting needed since there's only one group

        for (const groupKey of sortedGroupKeys) {
            const groupPapers = groups[groupKey];
            // For poster-search-rating, don't show the group header wrapper
            if (interestingPapersSortOrder === 'poster-search-rating') {
                html += '<div class="space-y-4">';
                groupPapers.forEach(paper => {
                    html += formatPaperCard(paper, { compact: false, showSearchTerm: true });
                });
                html += '</div>';
            } else {
                // Add edit button for search term if we're sorting by search terms
                const editButton = (interestingPapersSortOrder === 'search-rating-poster')
                    ? `<button onclick="editSearchTerm('${escapeHtml(currentInterestingSession).replace(/'/g, "\\'")}', '${escapeHtml(groupKey).replace(/'/g, "\\'")}', event)" 
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
                    // Show search term badge only when not sorting by search term (to avoid redundancy)
                    const showSearchTermBadge = interestingPapersSortOrder !== 'search-rating-poster';
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

// Switch to a different session in the interesting papers tab
function switchInterestingSession(session) {
    currentInterestingSession = session;
    loadInterestingPapers();
}

// Edit search term for a group of papers
function editSearchTerm(session, oldSearchTerm, event) {
    event.stopPropagation();

    // Prompt user for new search term
    const newSearchTerm = prompt('Edit search term:', oldSearchTerm);

    // If user cancelled or entered empty string, do nothing
    if (newSearchTerm === null || newSearchTerm.trim() === '') {
        return;
    }

    const trimmedNewSearchTerm = newSearchTerm.trim();

    // If search term hasn't changed, do nothing
    if (trimmedNewSearchTerm === oldSearchTerm) {
        return;
    }

    // Update all papers in this session with this search term
    let updatedCount = 0;
    for (const paperId in paperPriorities) {
        const paperData = paperPriorities[paperId];
        if (paperData.searchTerm === oldSearchTerm) {
            // We need to verify this paper belongs to the correct session
            // For now, update all papers with this search term
            paperData.searchTerm = trimmedNewSearchTerm;
            updatedCount++;
        }
    }

    // Save the updated priorities
    savePriorities();

    // Reload the interesting papers view to show the changes
    loadInterestingPapers();

    // Show a brief confirmation message
    if (updatedCount > 0) {
        console.log(`Updated search term for ${updatedCount} paper(s)`);
    }
}

// Edit search term for a single paper
function editPaperSearchTerm(paperId, event) {
    event.stopPropagation();

    // Get current search term
    const paperData = paperPriorities[paperId];
    if (!paperData) {
        console.error('Paper not found in priorities');
        return;
    }

    const oldSearchTerm = paperData.searchTerm || 'Unknown';

    // Prompt user for new search term
    const newSearchTerm = prompt('Edit search term for this paper:', oldSearchTerm);

    // If user cancelled or entered empty string, do nothing
    if (newSearchTerm === null || newSearchTerm.trim() === '') {
        return;
    }

    const trimmedNewSearchTerm = newSearchTerm.trim();

    // If search term hasn't changed, do nothing
    if (trimmedNewSearchTerm === oldSearchTerm) {
        return;
    }

    // Update this paper's search term
    paperData.searchTerm = trimmedNewSearchTerm;

    // Save the updated priorities
    savePriorities();

    // Reload the interesting papers view to show the changes
    loadInterestingPapers();
}

// Change the sort order for interesting papers
function changeInterestingPapersSortOrder(sortOrder) {
    interestingPapersSortOrder = sortOrder;
    localStorage.setItem('interestingPapersSortOrder', sortOrder);
    loadInterestingPapers();
}

// Save interesting papers as markdown
async function saveInterestingPapersAsMarkdown(event) {
    // Get all rated papers
    if (Object.keys(paperPriorities).length === 0) {
        alert('No papers rated yet. Rate some papers before saving.');
        return;
    }

    // Show progress message
    const button = event.target;
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Preparing download...';
    button.disabled = true;

    try {
        // Fetch details for all rated papers
        const paperIds = Object.keys(paperPriorities);

        // Get current search context
        const searchInput = document.getElementById('search-input');
        const searchQuery = searchInput ? searchInput.value : '';

        // Request the backend to generate markdown
        const response = await fetch(`${API_BASE}/api/export/interesting-papers`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                paper_ids: paperIds,
                priorities: paperPriorities,
                search_query: searchQuery,
                sort_order: interestingPapersSortOrder
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate export');
        }

        // Get the zip file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `interesting-papers-${new Date().toISOString().split('T')[0]}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show success message
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

// Save interesting papers data as JSON file
function saveInterestingPapersAsJSON() {
    if (Object.keys(paperPriorities).length === 0) {
        alert('No papers rated yet. Rate some papers before saving.');
        return;
    }

    try {
        // Create export data with metadata
        const exportData = {
            version: '1.0',
            exportDate: new Date().toISOString(),
            sortOrder: interestingPapersSortOrder,
            paperPriorities: paperPriorities,
            paperCount: Object.keys(paperPriorities).length
        };

        // Convert to JSON string
        const jsonString = JSON.stringify(exportData, null, 2);

        // Create blob and download
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

// Trigger file input for loading JSON
function loadInterestingPapersFromJSON() {
    const fileInput = document.getElementById('json-file-input');
    if (fileInput) {
        fileInput.click();
    }
}

// Handle JSON file load
function handleJSONFileLoad(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    // Validate file type
    if (!file.name.endsWith('.json')) {
        alert('Please select a JSON file.');
        event.target.value = '';
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        try {
            const importData = JSON.parse(e.target.result);

            // Validate data structure
            if (!importData.paperPriorities || typeof importData.paperPriorities !== 'object') {
                throw new Error('Invalid JSON format: missing or invalid paperPriorities');
            }

            // Ask for confirmation if there are existing ratings
            const existingCount = Object.keys(paperPriorities).length;
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

            // Merge the imported data with existing data
            // Existing ratings take precedence (we don't overwrite)
            const beforeCount = Object.keys(paperPriorities).length;
            let newCount = 0;
            let updatedCount = 0;

            for (const [paperId, data] of Object.entries(importData.paperPriorities)) {
                if (paperPriorities[paperId]) {
                    // Paper already rated - keep existing
                    updatedCount++;
                } else {
                    // New rating from import
                    paperPriorities[paperId] = data;
                    newCount++;
                }
            }

            // Optionally restore sort order if not set
            if (importData.sortOrder && !localStorage.getItem('interestingPapersSortOrder')) {
                interestingPapersSortOrder = importData.sortOrder;
                localStorage.setItem('interestingPapersSortOrder', interestingPapersSortOrder);

                // Update the dropdown
                const sortSelect = document.getElementById('sort-order');
                if (sortSelect) {
                    sortSelect.value = interestingPapersSortOrder;
                }
            }

            // Save to localStorage
            savePriorities();

            // Update UI
            updateInterestingPapersCount();

            // Reload interesting papers if on that tab
            if (currentTab === 'interesting') {
                loadInterestingPapers();
            }

            // Show success message
            const afterCount = Object.keys(paperPriorities).length;
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

        // Reset the file input
        event.target.value = '';
    };

    reader.onerror = function () {
        alert('Error reading file. Please try again.');
        event.target.value = '';
    };

    reader.readAsText(file);
}

// Generate markdown content for interesting papers
function generateInterestingPapersMarkdown(papers) {
    // Add priority to each paper
    papers.forEach(paper => {
        paper.priority = paperPriorities[paper.uid]?.priority || 0;
    });

    // Sort by session, priority (descending), and poster position
    papers.sort((a, b) => {
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;
        if (a.priority !== b.priority) return b.priority - a.priority;
        return naturalSortPosterPosition(a.poster_position, b.poster_position);
    });

    // Get current search context (if any)
    const searchInput = document.getElementById('search-input');
    const searchQuery = searchInput ? searchInput.value : '';

    // Build markdown
    let markdown = `# Interesting Papers from NeurIPS 2025\n\n`;
    markdown += `Generated: ${new Date().toLocaleString()}\n\n`;

    if (searchQuery) {
        markdown += `## Search Context\n\n`;
        markdown += `**Search Query:** ${searchQuery}\n\n`;
    }

    markdown += `**Total Papers:** ${papers.length}\n\n`;
    markdown += `---\n\n`;

    // Group by session
    const groupedBySession = {};
    papers.forEach(paper => {
        const session = paper.session || 'No Session';
        if (!groupedBySession[session]) {
            groupedBySession[session] = [];
        }
        groupedBySession[session].push(paper);
    });

    // Write each session
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

// Tab switching
function switchTab(tab) {
    currentTab = tab;

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
        // Update sort dropdown to match current sort order
        const sortDropdown = document.getElementById('sort-order');
        if (sortDropdown) {
            sortDropdown.value = interestingPapersSortOrder;
        }
        loadInterestingPapers();
    }
}

// Load statistics
async function loadStats() {
    try {
        // Get selected year and conference from header
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedYear = yearSelect ? yearSelect.value : '';
        const selectedConference = conferenceSelect ? conferenceSelect.value : '';

        // Build query params for stats
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

        // Build display text based on filters
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

// Load filter options
async function loadFilterOptions() {
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

        // Load session, topic, eventtype filters from database (filtered by year/conference)
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

            // Populate year selector in header (only if empty)
            if (yearSelect && yearSelect.options.length === 1) { // Only "All Years" option
                if (availableData.years && availableData.years.length > 0) {
                    availableData.years.forEach(year => {
                        const option = document.createElement('option');
                        option.value = year;
                        option.textContent = year;
                        yearSelect.appendChild(option);
                    });
                }
            }

            // Populate conference selector in header (only if empty)
            if (conferenceSelect && conferenceSelect.options.length === 1) { // Only "All Conferences" option
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
        sessionSelect.innerHTML = ''; // Clear existing options
        filtersData.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true; // Select all by default
            sessionSelect.appendChild(option);
        });

        // Clear and repopulate chat session filter
        const chatSessionSelect = document.getElementById('chat-session-filter');
        chatSessionSelect.innerHTML = ''; // Clear existing options
        filtersData.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true; // Select all by default
            chatSessionSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading filter options:', error);
    }
}

// Select all options in a filter
function selectAllFilter(filterId) {
    const select = document.getElementById(filterId);
    Array.from(select.options).forEach(option => {
        option.selected = true;
    });
}

// Deselect all options in a filter
function deselectAllFilter(filterId) {
    const select = document.getElementById(filterId);
    Array.from(select.options).forEach(option => {
        option.selected = false;
    });
}

// Search papers
async function searchPapers() {
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
    currentSearchTerm = query;

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

// Format a single paper card
function formatPaperCard(paper, options = {}) {
    const {
        compact = false,
        showNumber = null,
        abstractLength = compact ? 200 : 300,
        idPrefix = '',
        showSearchTerm = false
    } = options;

    const title = paper.title || 'Untitled';

    // Validate authors is an array (fail-early design)
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
            const remaining = paper.abstract.substring(abstractLength);
            // Render the full abstract as markdown with LaTeX, then split for preview
            const fullRendered = renderMarkdownWithLatex(paper.abstract);
            abstractHtml = `
                <details class="text-gray-700 ${compact ? 'text-xs' : 'text-sm'} leading-relaxed ${compact ? 'mt-2' : ''} markdown-content" onclick="event.stopPropagation()">
                    <summary class="cursor-pointer hover:text-purple-600">
                        ${renderMarkdownWithLatex(preview)}... <span class="text-purple-600 font-medium">Show more</span>
                    </summary>
                    <div class="mt-2">${renderMarkdownWithLatex(remaining)}</div>
                </details>
            `;
        } else {
            abstractHtml = `<div class="text-gray-700 ${compact ? 'text-xs' : 'text-sm'} leading-relaxed ${compact ? 'mt-2' : ''} markdown-content">${renderMarkdownWithLatex(paper.abstract)}</div>`;
        }
    } else if (!compact) {
        abstractHtml = `<p class="text-gray-700 text-sm leading-relaxed">No abstract available</p>`;
    }

    // Build metadata badges
    let metadata = '';
    if (paper.session) {
        metadata += `<span class="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full mr-${compact ? '1' : '2'}"><i class="fas fa-calendar-alt mr-1"></i>${escapeHtml(paper.session)}</span>`;
    }
    if (paper.poster_position) {
        metadata += `<span class="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded-full mr-${compact ? '1' : '2'}"><i class="fas fa-map-pin mr-1"></i>Poster ${escapeHtml(paper.poster_position)}</span>`;
    }
    if (showSearchTerm && paper.searchTerm) {
        metadata += `<span class="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full mr-${compact ? '1' : '2'} inline-flex items-center">
            <i class="fas fa-search mr-1"></i>${escapeHtml(paper.searchTerm)}
            <button onclick="editPaperSearchTerm(${paper.uid}, event)" class="ml-1 text-purple-600 hover:text-purple-800 focus:outline-none" title="Edit search term">
                <i class="fas fa-edit text-xs"></i>
            </button>
        </span>`;
    }
    if (paper.distance !== undefined) {
        metadata += `<span class="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"><i class="fas fa-chart-line mr-1"></i>${(1 - paper.distance).toFixed(compact ? 2 : 3)}</span>`;
    }

    const cardId = idPrefix ? `id="${idPrefix}"` : '';
    const cardClasses = compact
        ? 'paper-card bg-white rounded-lg shadow-sm p-3 hover:shadow-md cursor-pointer border border-gray-200'
        : 'paper-card bg-white rounded-lg shadow-md p-6 hover:shadow-lg cursor-pointer';

    // Get current priority for this paper
    const currentPriority = paperPriorities[paper.uid]?.priority || 0;

    // Generate star rating HTML
    // NOTE: paper.uid must be quoted in onclick handlers because it's a string UID
    // Without quotes: onclick="setPaperPriority(test_uid_123, 1)" - invalid JavaScript
    // With quotes: onclick="setPaperPriority('test_uid_123', 1)" - valid JavaScript
    let starsHtml = '<div class="flex-shrink-0 ml-2 flex items-center gap-0.5" onclick="event.stopPropagation()" title="Rate this paper">';
    for (let i = 1; i <= 5; i++) {
        const isSelected = i <= currentPriority;
        const starClass = isSelected
            ? 'fas fa-star text-yellow-400 hover:text-yellow-500'
            : 'far fa-star text-gray-300 hover:text-yellow-400';
        starsHtml += `<i class="${starClass} cursor-pointer text-${compact ? 'sm' : 'base'}" onclick="setPaperPriority('${paper.uid}', ${i})"></i>`;
    }
    starsHtml += '</div>';

    return `
        <div ${cardId} class="${cardClasses}" onclick="showPaperDetails('${paper.uid}')">
            ${showNumber !== null ? `
                <div class="flex items-start justify-between mb-1">
                    <span class="text-xs font-semibold text-purple-600">#${showNumber}</span>
                    ${paper.paper_url ? `
                        <a href="${escapeHtml(paper.paper_url)}" target="_blank" class="text-purple-600 hover:text-purple-800 text-xs" onclick="event.stopPropagation()">
                            <i class="fas fa-external-link-alt"></i>
                        </a>
                    ` : ''}
                </div>
            ` : ''}
            <div class="flex items-start justify-between ${compact ? 'mb-1' : 'mb-2'}">
                <h${compact ? '4' : '3'} class="${compact ? 'text-sm' : 'text-lg'} font-semibold text-gray-800 flex-1 ${compact ? 'leading-tight pr-2' : 'pr-2'}">${escapeHtml(title)}</h${compact ? '4' : '3'}>
                ${starsHtml}
            </div>
            <p class="${compact ? 'text-xs' : 'text-sm'} text-gray-600 ${compact ? 'mb-2 truncate' : 'mb-3'}">
                <i class="fas fa-users mr-1"></i>${escapeHtml(authors)}
            </p>
            ${metadata ? `<div class="${compact ? 'mb-2' : 'mb-3'}">${metadata}</div>` : ''}
            ${abstractHtml}
            ${!compact && paper.paper_url ? `
                <div class="mt-3">
                    <a href="${escapeHtml(paper.paper_url)}" target="_blank" class="text-purple-600 hover:text-purple-800 text-sm" onclick="event.stopPropagation()">
                        <i class="fas fa-external-link-alt mr-1"></i>View Paper Details
                    </a>
                </div>
            ` : ''}
            ${!compact && paper.distance !== undefined && !metadata.includes('chart-line') ? `
                <div class="mt-3 pt-3 border-t">
                    <span class="text-xs text-gray-500">
                        <i class="fas fa-chart-line mr-1"></i>
                        Relevance: ${(1 - paper.distance).toFixed(3)}
                    </span>
                </div>
            ` : ''}
        </div>
    `;
}

// Display search results
function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');

    if (!data.papers || data.papers.length === 0) {
        resultsDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-inbox text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">No papers found</p>
                <p class="text-sm">Try different keywords or search terms</p>
            </div>
        `;
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

// Show paper details (modal or expanded view)
async function showPaperDetails(paperId) {
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

        // Validate authors is an array (fail-early design)
        if (paper.authors && !Array.isArray(paper.authors)) {
            throw new TypeError(`Expected authors to be an array, got ${typeof paper.authors}`);
        }

        const authors = (paper.authors && paper.authors.length > 0)
            ? paper.authors.join(', ')
            : 'Unknown';
        const title = paper.title || 'Untitled';

        modal.innerHTML = `
            <div class="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto p-8">
                <div class="flex items-start justify-between mb-4">
                    <h2 class="text-2xl font-bold text-gray-800 flex-1">${escapeHtml(title)}</h2>
                    <button onclick="this.closest('.fixed').remove()" class="ml-4 text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                
                <div class="mb-4 flex flex-wrap gap-2">
                    ${paper.session ? `
                        <span class="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                            <i class="fas fa-calendar-alt mr-1"></i>${escapeHtml(paper.session)}
                        </span>
                    ` : ''}
                    ${paper.poster_position ? `
                        <span class="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm">
                            <i class="fas fa-map-pin mr-1"></i>Poster ${escapeHtml(paper.poster_position)}
                        </span>
                    ` : ''}
                    <span class="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                        <i class="fas fa-fingerprint mr-1"></i>ID: ${paper.uid}
                    </span>
                </div>
                
                <div class="mb-6">
                    <h3 class="text-sm font-semibold text-gray-700 mb-2">
                        <i class="fas fa-users mr-2"></i>Authors
                    </h3>
                    <p class="text-gray-700">${escapeHtml(authors)}</p>
                </div>
                
                <div class="mb-6">
                    <h3 class="text-sm font-semibold text-gray-700 mb-2">
                        <i class="fas fa-file-alt mr-2"></i>Abstract
                    </h3>
                    <div class="text-gray-700 leading-relaxed markdown-content">${paper.abstract ? renderMarkdownWithLatex(paper.abstract) : 'No abstract available'}</div>
                </div>
                
                <div class="flex gap-3">
                    ${paper.url ? `
                        <a href="${escapeHtml(paper.url)}" target="_blank" class="inline-block px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                            <i class="fas fa-external-link-alt mr-2"></i>View Paper Details
                        </a>
                    ` : ''}
                    ${paper.pdf_url ? `
                        <a href="${escapeHtml(paper.pdf_url)}" target="_blank" class="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                            <i class="fas fa-file-pdf mr-2"></i>View PDF
                        </a>
                    ` : ''}
                    ${paper.paper_url ? `
                        <a href="${escapeHtml(paper.paper_url)}" target="_blank" class="inline-block px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">
                            <i class="fas fa-link mr-2"></i>Paper Link
                        </a>
                    ` : ''}
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    } catch (error) {
        console.error('Error loading paper details:', error);
        showError('Error loading paper details');
    }
}

// Send chat message
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Store the current search term for rating papers from chat
    currentSearchTerm = message;

    // Add user message
    addChatMessage(message, 'user');
    input.value = '';

    // Show loading
    const loadingId = addChatMessage('Thinking...', 'assistant', true);

    try {
        const nPapers = parseInt(document.getElementById('n-papers').value);

        // Get multiple selected values from chat multi-select dropdowns
        const chatSessionSelect = document.getElementById('chat-session-filter');

        const sessions = Array.from(chatSessionSelect.selectedOptions).map(opt => opt.value);

        // Get year and conference from header selectors
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedYear = yearSelect.value;
        const selectedConference = conferenceSelect.value;

        const requestBody = {
            message,
            n_papers: nPapers
        };

        // Add filters only if NOT all options are selected (all selected = no filter)
        if (sessions.length > 0 && sessions.length < chatSessionSelect.options.length) {
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

        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        // Remove loading message
        document.getElementById(loadingId).remove();

        if (data.error) {
            addChatMessage(`Error: ${data.error}`, 'assistant');
            return;
        }

        // Extract response text and papers from the response object
        // The API returns: { response: { response: "text", papers: [...], metadata: {...} } }
        let responseText, papers, metadata;
        if (typeof data.response === 'object' && data.response !== null) {
            responseText = data.response.response || JSON.stringify(data.response);
            papers = data.response.papers || [];
            metadata = data.response.metadata || {};
        } else if (typeof data.response === 'string') {
            responseText = data.response;
            papers = [];
            metadata = {};
        } else {
            responseText = JSON.stringify(data.response);
            papers = [];
            metadata = {};
        }

        // Update currentSearchTerm to use the rewritten query if available
        if (metadata.rewritten_query) {
            currentSearchTerm = metadata.rewritten_query;
        }

        addChatMessage(responseText, 'assistant');

        // Display relevant papers with metadata (including rewritten query)
        displayChatPapers(papers, metadata);
    } catch (error) {
        console.error('Chat error:', error);
        document.getElementById(loadingId).remove();
        addChatMessage('Sorry, an error occurred. Please try again.', 'assistant');
    }
}

// Add chat message
function addChatMessage(text, role, isLoading = false) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageId = `msg-${Date.now()}`;

    const isUser = role === 'user';
    const bgColor = isUser ? 'bg-purple-600 text-white' : 'bg-white text-gray-700';
    const iconBg = isUser ? 'bg-gray-600' : 'bg-purple-600';
    const icon = isUser ? 'fa-user' : 'fa-robot';
    const justifyClass = isUser ? 'justify-end' : 'justify-start';

    // Render markdown for assistant messages, escape HTML for user messages
    const contentHtml = isUser
        ? `<p class="whitespace-pre-wrap">${escapeHtml(text)}</p>`
        : `<div class="markdown-content">${marked.parse(text)}</div>`;

    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = 'chat-message';
    messageDiv.innerHTML = `
        <div class="flex items-start gap-3 ${justifyClass}">
            ${!isUser ? `
                <div class="flex-shrink-0 w-8 h-8 ${iconBg} rounded-full flex items-center justify-center text-white">
                    <i class="fas ${icon} text-sm"></i>
                </div>
            ` : ''}
            <div class="${bgColor} rounded-lg p-4 shadow-sm max-w-2xl">
                ${contentHtml}
                ${isLoading ? '<div class="spinner mt-2" style="width: 20px; height: 20px; border-width: 2px;"></div>' : ''}
            </div>
            ${isUser ? `
                <div class="flex-shrink-0 w-8 h-8 ${iconBg} rounded-full flex items-center justify-center text-white">
                    <i class="fas ${icon} text-sm"></i>
                </div>
            ` : ''}
        </div>
    `;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return messageId;
}

// Display chat papers in the side panel
function displayChatPapers(papers, metadata = {}) {
    const papersDiv = document.getElementById('chat-papers');

    if (!papers || papers.length === 0) {
        papersDiv.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fas fa-inbox text-4xl mb-3 opacity-20"></i>
                <p class="text-sm">No papers found for this query</p>
            </div>
        `;
        return;
    }

    let html = '';

    // Display rewritten query if available
    if (metadata.rewritten_query) {
        const wasRetrieved = metadata.retrieved_new_papers !== false;
        const cacheIcon = wasRetrieved ? 'fa-sync-alt' : 'fa-check-circle';
        const cacheColor = wasRetrieved ? 'text-blue-600' : 'text-green-600';
        const cacheText = wasRetrieved ? 'Retrieved new papers' : 'Using cached papers';

        html += `
            <div class="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4 mb-4 shadow-sm">
                <div class="flex items-start gap-2 mb-2">
                    <i class="fas fa-magic text-purple-600 mt-1"></i>
                    <div class="flex-1">
                        <h3 class="text-sm font-semibold text-gray-700 mb-1">Optimized Search Query</h3>
                        <p class="text-sm text-gray-800 font-medium italic">"${escapeHtml(metadata.rewritten_query)}"</p>
                    </div>
                </div>
                <div class="flex items-center gap-2 text-xs text-gray-600 mt-2 pt-2 border-t border-purple-200">
                    <i class="fas ${cacheIcon} ${cacheColor}"></i>
                    <span>${cacheText}</span>
                    <span class="ml-auto">${papers.length} paper${papers.length !== 1 ? 's' : ''} found</span>
                </div>
            </div>
        `;
    }

    // Display papers
    papers.forEach((paper, index) => {
        html += formatPaperCard(paper, {
            compact: true,
            showNumber: index + 1,
            idPrefix: `paper-${index + 1}`
        });
    });

    papersDiv.innerHTML = html;
}

// Reset chat
async function resetChat() {
    try {
        await fetch(`${API_BASE}/api/chat/reset`, {
            method: 'POST'
        });

        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = '';
        addChatMessage('Conversation reset. How can I help you explore NeurIPS abstracts?', 'assistant');

        // Clear papers panel
        const papersDiv = document.getElementById('chat-papers');
        papersDiv.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fas fa-inbox text-4xl mb-3 opacity-20"></i>
                <p class="text-sm">Ask a question to see relevant papers</p>
            </div>
        `;
    } catch (error) {
        console.error('Error resetting chat:', error);
    }
}

// Show error message
function showError(message) {
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = `
        <div class="bg-red-50 border border-red-200 rounded-lg p-6">
            <div class="flex items-center">
                <i class="fas fa-exclamation-circle text-red-500 text-2xl mr-3"></i>
                <div>
                    <h3 class="text-red-800 font-semibold">Error</h3>
                    <p class="text-red-700 text-sm mt-1">${escapeHtml(message)}</p>
                </div>
            </div>
        </div>
    `;
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Configure marked with KaTeX extension (run once on load)
if (typeof markedKatex !== 'undefined') {
    marked.use(markedKatex({
        throwOnError: false,
        nonStandard: true,
        output: 'html'
    }));
}

// Utility: Render markdown with LaTeX support
function renderMarkdownWithLatex(text) {
    if (!text) return '';

    try {
        return marked.parse(text);
    } catch (e) {
        console.warn('Markdown parsing error:', e);
        // Fallback to escaped HTML if markdown parsing fails
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

// Handle year selector change
function handleYearChange() {
    // Reload stats with the new year selection
    loadStats();

    // Reload filter options with the new year selection
    loadFilterOptions();

    // Update download button text
    updateDownloadButton();

    // Refresh interesting papers if the tab is visible, otherwise just update the count
    if (currentTab === 'interesting') {
        loadInterestingPapers();
    } else {
        updateInterestingPapersCount();
    }

    // Clear search results when year changes
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = `
        <div class="text-center text-gray-500 py-12">
            <i class="fas fa-search text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">Enter a search query to find abstracts</p>
            <p class="text-sm">Try "transformer architecture" or "reinforcement learning"</p>
        </div>
    `;
}

// Handle conference selector change
function handleConferenceChange() {
    // Update years dropdown based on selected conference
    updateYearsForConference();

    // Reload stats with the new conference selection
    loadStats();

    // Reload filter options with the new conference selection
    loadFilterOptions();

    // Update download button text
    updateDownloadButton();

    // Refresh interesting papers if the tab is visible, otherwise just update the count
    if (currentTab === 'interesting') {
        loadInterestingPapers();
    } else {
        updateInterestingPapersCount();
    }

    // Clear search results when conference changes
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = `
        <div class="text-center text-gray-500 py-12">
            <i class="fas fa-search text-6xl mb-4 opacity-20"></i>
            <p class="text-lg">Enter a search query to find abstracts</p>
            <p class="text-sm">Try "transformer architecture" or "reinforcement learning"</p>
        </div>
    `;
}

// Update years dropdown based on selected conference
function updateYearsForConference() {
    const conferenceSelect = document.getElementById('conference-selector');
    const yearSelect = document.getElementById('year-selector');

    if (!conferenceSelect || !yearSelect || !window.conferenceYearsMap) {
        return;
    }

    const selectedConference = conferenceSelect.value;
    const currentYear = yearSelect.value; // Remember current selection

    // Clear years dropdown (except "All Years" option)
    yearSelect.innerHTML = '<option value="">All Years</option>';

    if (selectedConference) {
        // Conference selected: show only years for that conference
        const yearsForConference = window.conferenceYearsMap[selectedConference] || [];
        yearsForConference.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });
    } else {
        // "All Conferences" selected: show all years
        if (window.allYears && window.allYears.length > 0) {
            window.allYears.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearSelect.appendChild(option);
            });
        }
    }

    // Try to restore previous year selection if it's still available
    const availableYears = Array.from(yearSelect.options).map(opt => opt.value);
    if (currentYear && availableYears.includes(currentYear)) {
        yearSelect.value = currentYear;
    } else {
        yearSelect.value = ''; // Reset to "All Years" if previous selection not available
    }
}

// Update download button text based on whether data exists for selected conference/year
async function updateDownloadButton() {
    const conferenceSelect = document.getElementById('conference-selector');
    const yearSelect = document.getElementById('year-selector');
    const downloadBtn = document.getElementById('download-btn');
    const downloadBtnText = document.getElementById('download-btn-text');

    if (!conferenceSelect || !yearSelect || !downloadBtn || !downloadBtnText) {
        return;
    }

    const selectedConference = conferenceSelect.value;
    const selectedYear = yearSelect.value;

    // If no conference or year selected, disable button
    if (!selectedConference || !selectedYear) {
        downloadBtn.disabled = true;
        downloadBtn.classList.add('opacity-50', 'cursor-not-allowed');
        downloadBtn.title = 'Select a conference and year to download';
        downloadBtnText.textContent = 'Download';
        return;
    }

    // Enable button
    downloadBtn.disabled = false;
    downloadBtn.classList.remove('opacity-50', 'cursor-not-allowed');

    // Check if we have data for this conference/year
    try {
        const response = await fetch(`${API_BASE}/api/stats?year=${selectedYear}&conference=${selectedConference}`);
        const data = await response.json();

        if (data.total_papers > 0) {
            downloadBtnText.textContent = 'Update';
            downloadBtn.title = `Update abstracts for ${selectedConference} ${selectedYear} (${data.total_papers} papers currently)`;
        } else {
            downloadBtnText.textContent = 'Download';
            downloadBtn.title = `Download abstracts for ${selectedConference} ${selectedYear}`;
        }
    } catch (error) {
        console.error('Error checking paper count:', error);
        downloadBtnText.textContent = 'Download';
        downloadBtn.title = `Download/update abstracts for ${selectedConference} ${selectedYear}`;
    }
}

// Handle download/update button click
async function handleDownload() {
    const conferenceSelect = document.getElementById('conference-selector');
    const yearSelect = document.getElementById('year-selector');
    const downloadBtn = document.getElementById('download-btn');
    const downloadBtnText = document.getElementById('download-btn-text');
    const downloadIcon = document.getElementById('download-icon');
    const progressBg = document.getElementById('download-progress-bg');

    const selectedConference = conferenceSelect ? conferenceSelect.value : '';
    const selectedYear = yearSelect ? yearSelect.value : '';

    // Validate selections
    if (!selectedConference) {
        alert('Please select a conference first');
        return;
    }

    if (!selectedYear) {
        alert('Please select a year first');
        return;
    }

    // Disable button during download
    downloadBtn.disabled = true;
    downloadBtn.classList.add('opacity-50', 'cursor-not-allowed');
    const originalIcon = downloadIcon.className;
    downloadIcon.className = 'fas fa-spinner fa-spin';
    downloadBtnText.textContent = 'Starting...';

    // Show progress bar background
    progressBg.classList.remove('hidden');
    progressBg.style.width = '0%';

    try {
        const response = await fetch(`${API_BASE}/api/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                conference: selectedConference,
                year: parseInt(selectedYear)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Read the SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6);
                    if (dataStr.trim()) {
                        try {
                            const data = JSON.parse(dataStr);

                            if (data.error) {
                                alert(`Error: ${data.error}`);
                                return;
                            }

                            // Update button progress based on stage
                            if (data.stage === 'download') {
                                downloadBtnText.textContent = `Downloading (${data.progress}%)`;
                                progressBg.style.width = `${Math.round(data.progress / 3)}%`;
                            } else if (data.stage === 'database') {
                                downloadBtnText.textContent = `Loading DB (${data.progress}%)`;
                                const progress = 33 + Math.round(data.progress / 3);
                                progressBg.style.width = `${progress}%`;
                            } else if (data.stage === 'embeddings') {
                                downloadBtnText.textContent = `Embedding (${data.progress}%)`;
                                const progress = 66 + Math.round(data.progress / 3);
                                progressBg.style.width = `${progress}%`;
                            } else if (data.stage === 'complete' && data.success) {
                                // Complete
                                progressBg.style.width = '100%';
                                downloadIcon.className = 'fas fa-check';
                                downloadBtnText.textContent = 'Complete!';

                                // Show success message
                                const action = data.action === 'updating' ? 'Updated' : 'Downloaded';
                                const message = `${action} ${data.downloaded} papers\n` +
                                    `Loaded ${data.updated} papers into database\n` +
                                    `Created ${data.embedded} embeddings for new/changed content`;

                                setTimeout(() => {
                                    alert(message);

                                    // Refresh the page data
                                    loadStats();
                                    loadFilterOptions();

                                    // Refresh interesting papers if on that tab
                                    if (currentTab === 'interesting') {
                                        loadInterestingPapers();
                                    } else {
                                        updateInterestingPapersCount();
                                    }

                                    // Reset button after a delay
                                    setTimeout(() => {
                                        progressBg.classList.add('hidden');
                                        progressBg.style.width = '0%';
                                        updateDownloadButton();
                                    }, 2000);
                                }, 500);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e, dataStr);
                        }
                    }
                }
            }
        }

    } catch (error) {
        console.error('Error downloading abstracts:', error);
        alert(`Error downloading abstracts: ${error.message}`);
        progressBg.classList.add('hidden');
        progressBg.style.width = '0%';
    } finally {
        // Re-enable button
        downloadBtn.disabled = false;
        downloadBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        downloadIcon.className = originalIcon;
        // Button text will be updated by updateDownloadButton() after data refresh
    }
}
