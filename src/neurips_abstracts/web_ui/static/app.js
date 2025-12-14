// API base URL
const API_BASE = '';

// State
let currentTab = 'search';
let chatHistory = [];
let paperPriorities = {}; // Store paper priorities: { paperId: { priority: number, searchTerm: string } }
let currentSearchTerm = ''; // Track the current search term

// Initialize app
document.addEventListener('DOMContentLoaded', function () {
    loadStats();
    loadFilterOptions();
    loadPriorities();
});

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

    // If clicking the same star, remove the rating
    if (currentPriority === priority) {
        delete paperPriorities[paperId];
    } else if (priority === 0) {
        // Remove priority if set to 0
        delete paperPriorities[paperId];
    } else {
        paperPriorities[paperId] = {
            priority: priority,
            searchTerm: currentSearchTerm || 'Unknown'
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
    const paperCard = document.querySelector(`[onclick*="showPaperDetails(${paperId})"]`);
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
function updateInterestingPapersCount() {
    const count = Object.keys(paperPriorities).length;
    const countElement = document.getElementById('interesting-count');
    if (countElement) {
        countElement.textContent = count;
    }
}

// Load and display interesting papers
async function loadInterestingPapers() {
    const listDiv = document.getElementById('interesting-papers-list');

    // If no rated papers, show empty state
    if (Object.keys(paperPriorities).length === 0) {
        listDiv.innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-star text-6xl mb-4 opacity-20"></i>
                <p class="text-lg">No papers rated yet</p>
                <p class="text-sm">Rate papers using the stars to add them here</p>
            </div>
        `;
        return;
    }

    // Show loading
    listDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        // Fetch details for all rated papers
        const paperIds = Object.keys(paperPriorities).map(id => parseInt(id));
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

    // Add priority and search term to each paper
    papers.forEach(paper => {
        const paperData = paperPriorities[paper.id];
        paper.priority = paperData?.priority || 0;
        paper.searchTerm = paperData?.searchTerm || 'Unknown';
    });

    // Sort by session, search term, priority (descending), and poster position
    papers.sort((a, b) => {
        // First by session
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;

        // Then by search term
        const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
        if (searchTermCompare !== 0) return searchTermCompare;

        // Then by priority (descending - higher priority first)
        if (a.priority !== b.priority) return b.priority - a.priority;

        // Finally by poster position
        const aPoster = a.poster_position || '';
        const bPoster = b.poster_position || '';
        return aPoster.localeCompare(bPoster);
    });

    // Group by session, then by search term
    const groupedBySession = {};
    papers.forEach(paper => {
        const session = paper.session || 'No Session';
        if (!groupedBySession[session]) {
            groupedBySession[session] = {};
        }
        const searchTerm = paper.searchTerm || 'Unknown';
        if (!groupedBySession[session][searchTerm]) {
            groupedBySession[session][searchTerm] = [];
        }
        groupedBySession[session][searchTerm].push(paper);
    });

    // Generate HTML
    let html = '';

    for (const [session, searchTerms] of Object.entries(groupedBySession)) {
        html += `
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h2 class="text-2xl font-bold text-gray-900 mb-6 border-b-2 border-green-300 pb-3">
                    <i class="fas fa-calendar-alt text-green-600 mr-2"></i>${escapeHtml(session)}
                </h2>
        `;

        for (const [searchTerm, termPapers] of Object.entries(searchTerms)) {
            html += `
                <div class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-md p-5 mb-4">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 border-b border-blue-200 pb-2">
                        <i class="fas fa-search text-blue-600 mr-2"></i>${escapeHtml(searchTerm)}
                    </h3>
                    <div class="space-y-4">
            `;

            termPapers.forEach(paper => {
                html += formatPaperCard(paper, { compact: false });
            });

            html += `
                    </div>
                </div>
            `;
        }

        html += `
            </div>
        `;
    }

    listDiv.innerHTML = html;
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
        const paperIds = Object.keys(paperPriorities).map(id => parseInt(id));

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
                search_query: searchQuery
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate export');
        }

        // Get the file (either .md or .zip depending on download_assets)
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const fileExtension = downloadAssets ? 'zip' : 'md';
        a.download = `interesting-papers-${new Date().toISOString().split('T')[0]}.${fileExtension}`;
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

// Generate markdown content for interesting papers
function generateInterestingPapersMarkdown(papers) {
    // Add priority to each paper
    papers.forEach(paper => {
        paper.priority = paperPriorities[paper.id]?.priority || 0;
    });

    // Sort by session, priority (descending), and poster position
    papers.sort((a, b) => {
        const sessionCompare = (a.session || '').localeCompare(b.session || '');
        if (sessionCompare !== 0) return sessionCompare;
        if (a.priority !== b.priority) return b.priority - a.priority;
        const aPoster = a.poster_position || '';
        const bPoster = b.poster_position || '';
        return aPoster.localeCompare(bPoster);
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
            markdown += `### ${paper.name || 'Untitled'}\n\n`;
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
        loadInterestingPapers();
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('stats').innerHTML = `
                <div class="text-sm text-red-200">${data.error}</div>
            `;
            return;
        }

        document.getElementById('stats').innerHTML = `
            <div class="text-sm font-semibold">${data.total_papers.toLocaleString()} Abstracts</div>
            <div class="text-xs opacity-90">Neurips 2025</div>
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
        const response = await fetch(`${API_BASE}/api/filters`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading filters:', data.error);
            return;
        }

        // Populate search session filter
        const sessionSelect = document.getElementById('session-filter');
        data.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true; // Select all by default
            sessionSelect.appendChild(option);
        });

        // Populate search topic filter
        const topicSelect = document.getElementById('topic-filter');
        data.topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic;
            option.textContent = topic;
            option.selected = true; // Select all by default
            topicSelect.appendChild(option);
        });

        // Populate search eventtype filter (single select)
        const eventtypeSelect = document.getElementById('eventtype-filter');
        data.eventtypes.forEach(eventtype => {
            const option = document.createElement('option');
            option.value = eventtype;
            option.textContent = eventtype;
            eventtypeSelect.appendChild(option);
        });

        // Populate chat session filter
        const chatSessionSelect = document.getElementById('chat-session-filter');
        data.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            option.selected = true; // Select all by default
            chatSessionSelect.appendChild(option);
        });

        // Populate chat topic filter
        const chatTopicSelect = document.getElementById('chat-topic-filter');
        data.topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic;
            option.textContent = topic;
            option.selected = true; // Select all by default
            chatTopicSelect.appendChild(option);
        });

        // Populate chat eventtype filter (single select)
        const chatEventtypeSelect = document.getElementById('chat-eventtype-filter');
        data.eventtypes.forEach(eventtype => {
            const option = document.createElement('option');
            option.value = eventtype;
            option.textContent = eventtype;
            chatEventtypeSelect.appendChild(option);
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
    const topicSelect = document.getElementById('topic-filter');
    const eventtypeSelect = document.getElementById('eventtype-filter');

    const sessions = Array.from(sessionSelect.selectedOptions).map(opt => opt.value);
    const topics = Array.from(topicSelect.selectedOptions).map(opt => opt.value);
    const eventtype = eventtypeSelect.value; // Single select

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
        if (topics.length > 0 && topics.length < topicSelect.options.length) {
            requestBody.topics = topics;
        }
        if (eventtype) requestBody.eventtypes = [eventtype]; // Single value as array

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
        idPrefix = ''
    } = options;

    const title = paper.name || paper.title || 'Untitled';

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
    if (paper.distance !== undefined) {
        metadata += `<span class="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"><i class="fas fa-chart-line mr-1"></i>${(1 - paper.distance).toFixed(compact ? 2 : 3)}</span>`;
    }

    const cardId = idPrefix ? `id="${idPrefix}"` : '';
    const cardClasses = compact
        ? 'paper-card bg-white rounded-lg shadow-sm p-3 hover:shadow-md cursor-pointer border border-gray-200'
        : 'paper-card bg-white rounded-lg shadow-md p-6 hover:shadow-lg cursor-pointer';

    // Get current priority for this paper
    const currentPriority = paperPriorities[paper.id]?.priority || 0;

    // Generate star rating HTML
    let starsHtml = '<div class="flex-shrink-0 ml-2 flex items-center gap-0.5" onclick="event.stopPropagation()" title="Rate this paper">';
    for (let i = 1; i <= 5; i++) {
        const isSelected = i <= currentPriority;
        const starClass = isSelected
            ? 'fas fa-star text-yellow-400 hover:text-yellow-500'
            : 'far fa-star text-gray-300 hover:text-yellow-400';
        starsHtml += `<i class="${starClass} cursor-pointer text-${compact ? 'sm' : 'base'}" onclick="setPaperPriority(${paper.id}, ${i})"></i>`;
    }
    starsHtml += '</div>';

    return `
        <div ${cardId} class="${cardClasses}" onclick="showPaperDetails(${paper.id})">
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
        const title = paper.name || paper.title || 'Untitled';

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
                        <i class="fas fa-fingerprint mr-1"></i>ID: ${paper.id}
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
        const chatTopicSelect = document.getElementById('chat-topic-filter');
        const chatEventtypeSelect = document.getElementById('chat-eventtype-filter');

        const sessions = Array.from(chatSessionSelect.selectedOptions).map(opt => opt.value);
        const topics = Array.from(chatTopicSelect.selectedOptions).map(opt => opt.value);
        const eventtype = chatEventtypeSelect.value; // Single select

        const requestBody = {
            message,
            n_papers: nPapers
        };

        // Add filters only if NOT all options are selected (all selected = no filter)
        if (sessions.length > 0 && sessions.length < chatSessionSelect.options.length) {
            requestBody.sessions = sessions;
        }
        if (topics.length > 0 && topics.length < chatTopicSelect.options.length) {
            requestBody.topics = topics;
        }
        if (eventtype) requestBody.eventtypes = [eventtype]; // Single value as array

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

