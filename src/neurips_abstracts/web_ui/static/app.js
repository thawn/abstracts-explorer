// API base URL
const API_BASE = '';

// State
let currentTab = 'search';
let chatHistory = [];

// Initialize app
document.addEventListener('DOMContentLoaded', function () {
    loadStats();
});

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
            <div class="text-sm font-semibold">${data.total_papers.toLocaleString()} Papers</div>
            <div class="text-xs opacity-90">${data.min_year} - ${data.max_year}</div>
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('stats').innerHTML = `
            <div class="text-sm text-red-200">Error loading stats</div>
        `;
    }
}

// Search papers
async function searchPapers() {
    const query = document.getElementById('search-input').value.trim();
    const useEmbeddings = document.getElementById('use-embeddings').checked;
    const limit = parseInt(document.getElementById('limit-select').value);

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    // Show loading
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = `
        <div class="flex justify-center items-center py-12">
            <div class="spinner"></div>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                use_embeddings: useEmbeddings,
                limit
            })
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

    // Display papers
    data.papers.forEach(paper => {
        const title = paper.name || paper.title || 'Untitled';
        const authors = paper.authors ? paper.authors.join(', ') : 'Unknown';

        // Build abstract with collapsible details if needed
        let abstractHtml = '';
        if (paper.abstract) {
            if (paper.abstract.length > 300) {
                const preview = paper.abstract.substring(0, 300);
                abstractHtml = `
                    <details class="text-gray-700 text-sm leading-relaxed" onclick="event.stopPropagation()">
                        <summary class="cursor-pointer hover:text-purple-600">
                            ${escapeHtml(preview)}... <span class="text-purple-600 font-medium">Show more</span>
                        </summary>
                        <p class="mt-2">${escapeHtml(paper.abstract)}</p>
                    </details>
                `;
            } else {
                abstractHtml = `<p class="text-gray-700 text-sm leading-relaxed">${escapeHtml(paper.abstract)}</p>`;
            }
        } else {
            abstractHtml = `<p class="text-gray-700 text-sm leading-relaxed">No abstract available</p>`;
        }

        // Build metadata badges
        let metadata = '';
        if (paper.session) {
            metadata += `<span class="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full mr-2"><i class="fas fa-calendar-alt mr-1"></i>${escapeHtml(paper.session)}</span>`;
        }
        if (paper.poster_position) {
            metadata += `<span class="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded-full mr-2"><i class="fas fa-map-pin mr-1"></i>Poster ${escapeHtml(paper.poster_position)}</span>`;
        }

        html += `
            <div class="paper-card bg-white rounded-lg shadow-md p-6 hover:shadow-lg cursor-pointer" onclick="showPaperDetails(${paper.id})">
                <div class="flex items-start justify-between mb-2">
                    <h3 class="text-lg font-semibold text-gray-800 flex-1">${escapeHtml(title)}</h3>
                </div>
                <p class="text-sm text-gray-600 mb-3">
                    <i class="fas fa-users mr-1"></i>${escapeHtml(authors)}
                </p>
                ${metadata ? `<div class="mb-3">${metadata}</div>` : ''}
                ${abstractHtml}
                ${paper.paper_url ? `
                    <div class="mt-3">
                        <a href="${escapeHtml(paper.paper_url)}" target="_blank" class="text-purple-600 hover:text-purple-800 text-sm" onclick="event.stopPropagation()">
                            <i class="fas fa-external-link-alt mr-1"></i>View Paper Details
                        </a>
                    </div>
                ` : ''}
                ${paper.distance !== undefined ? `
                    <div class="mt-3 pt-3 border-t">
                        <span class="text-xs text-gray-500">
                            <i class="fas fa-chart-line mr-1"></i>
                            Relevance: ${(1 - paper.distance).toFixed(3)}
                        </span>
                    </div>
                ` : ''}
            </div>
        `;
    });

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
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50';
        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };

        const authors = paper.authors ? paper.authors.join(', ') : 'Unknown';
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
                    <p class="text-gray-700 leading-relaxed">${escapeHtml(paper.abstract || 'No abstract available')}</p>
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

    // Add user message
    addChatMessage(message, 'user');
    input.value = '';

    // Show loading
    const loadingId = addChatMessage('Thinking...', 'assistant', true);

    try {
        const nPapers = parseInt(document.getElementById('n-papers').value);

        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                n_papers: nPapers
            })
        });

        const data = await response.json();

        // Remove loading message
        document.getElementById(loadingId).remove();

        if (data.error) {
            addChatMessage(`Error: ${data.error}`, 'assistant');
            return;
        }

        // Extract response text from the response object
        const responseText = typeof data.response === 'string'
            ? data.response
            : data.response.response || JSON.stringify(data.response);

        addChatMessage(responseText, 'assistant');
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
                <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
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

// Reset chat
async function resetChat() {
    try {
        await fetch(`${API_BASE}/api/chat/reset`, {
            method: 'POST'
        });

        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = `
            <div class="chat-message">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white">
                        <i class="fas fa-robot text-sm"></i>
                    </div>
                    <div class="bg-white rounded-lg p-4 shadow-sm max-w-2xl">
                        <p class="text-gray-700">Conversation reset. How can I help you explore NeurIPS papers?</p>
                    </div>
                </div>
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
