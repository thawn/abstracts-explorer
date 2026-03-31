/**
 * Chat Module
 * 
 * Handles chat functionality including message sending, display,
 * and conversation management.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { renderEmptyState } from './utils/ui-utils.js';
import { setCurrentSearchTerm } from './state.js';
import { formatPaperCard } from './paper-card.js';

/**
 * Send chat message
 * @async
 */
export async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Store the current search term for rating papers from chat
    setCurrentSearchTerm(message);

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
        const selectedYears = getSelectedYears();
        const selectedConference = getSelectedConference();

        const requestBody = {
            message,
            n_papers: nPapers
        };

        // Add filters only if NOT all options are selected (all selected = no filter)
        if (sessions.length > 0 && sessions.length < chatSessionSelect.options.length) {
            requestBody.sessions = sessions;
        }

        // Add year filter if selected
        if (selectedYears.length > 0) {
            requestBody.years = selectedYears;
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
        let responseText, papers, metadata, visualizations;
        if (typeof data.response === 'object' && data.response !== null) {
            responseText = data.response.response || JSON.stringify(data.response);
            papers = data.response.papers || [];
            metadata = data.response.metadata || {};
            visualizations = data.response.visualizations || [];
        } else if (typeof data.response === 'string') {
            responseText = data.response;
            papers = [];
            metadata = {};
            visualizations = [];
        } else {
            responseText = JSON.stringify(data.response);
            papers = [];
            metadata = {};
            visualizations = [];
        }

        // Update currentSearchTerm to use the rewritten query if available
        if (metadata.rewritten_query) {
            setCurrentSearchTerm(metadata.rewritten_query);
        }

        addChatMessage(responseText, 'assistant');

        // Display visualizations from MCP tool results
        if (visualizations.length > 0) {
            renderChatVisualizations(visualizations);
        }

        // Display relevant papers with metadata (including rewritten query)
        displayChatPapers(papers, metadata);
    } catch (error) {
        console.error('Chat error:', error);
        document.getElementById(loadingId).remove();
        addChatMessage('Sorry, an error occurred. Please try again.', 'assistant');
    }
}

/**
 * Display chat papers in the side panel
 * @param {Array} papers - Array of paper objects
 * @param {Object} metadata - Metadata about the search
 */
export function displayChatPapers(papers, metadata = {}) {
    const papersDiv = document.getElementById('chat-papers');

    if (!papers || papers.length === 0) {
        papersDiv.innerHTML = renderEmptyState(
            'No papers found for this query',
            '',
            'fa-inbox'
        );
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

/**
 * Add chat message to conversation
 * @param {string} text - Message text
 * @param {string} role - 'user' or 'assistant'
 * @param {boolean} isLoading - Whether to show loading spinner
 * @returns {string} Message ID
 */
export function addChatMessage(text, role, isLoading = false) {
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

/**
 * Reset chat conversation
 * @async
 */
export async function resetChat() {
    try {
        await fetch(`${API_BASE}/api/chat/reset`, {
            method: 'POST'
        });

        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = '';
        addChatMessage('Conversation reset. How can I help you explore NeurIPS abstracts?', 'assistant');

        // Clear papers panel
        const papersDiv = document.getElementById('chat-papers');
        papersDiv.innerHTML = renderEmptyState(
            'Ask a question to see relevant papers',
            '',
            'fa-inbox'
        );
    } catch (error) {
        console.error('Error resetting chat:', error);
    }
}

/**
 * Render visualizations returned by MCP tools in the chat area.
 * @param {Array} visualizations - Array of visualization descriptor objects
 */
export function renderChatVisualizations(visualizations) {
    const messagesDiv = document.getElementById('chat-messages');

    for (let i = 0; i < visualizations.length; i++) {
        const viz = visualizations[i];
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-message';

        const plotId = `chat-plot-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 7)}`;

        wrapper.innerHTML = `
            <div class="flex items-start gap-3 justify-start">
                <div class="flex-shrink-0 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white">
                    <i class="fas fa-chart-bar text-sm"></i>
                </div>
                <div class="bg-white rounded-lg p-4 shadow-sm w-full max-w-2xl">
                    <div id="${plotId}" style="width:100%;height:400px;"></div>
                </div>
            </div>
        `;

        messagesDiv.appendChild(wrapper);

        if (viz.type === 'topic_evolution') {
            _renderTopicEvolutionChart(plotId, viz);
        } else if (viz.type === 'cluster_visualization') {
            _renderClusterVisualizationChart(plotId, viz);
        }
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Render a topic evolution bar chart using Plotly.
 * @param {string} plotId - DOM element id for the plot container
 * @param {Object} viz - Visualization data with year_counts, topic, conference
 */
function _renderTopicEvolutionChart(plotId, viz) {
    const yearCounts = viz.year_counts || {};
    const years = Object.keys(yearCounts).sort();
    const counts = years.map(y => yearCounts[y]);

    const trace = {
        x: years,
        y: counts,
        type: 'bar',
        marker: { color: '#7c3aed' }
    };

    const layout = {
        title: `Topic Evolution: ${viz.topic || ''}` + (viz.conference ? ` (${viz.conference})` : ''),
        xaxis: { title: 'Year', type: 'category' },
        yaxis: { title: 'Number of Papers', dtick: 1 },
        margin: { t: 40, b: 50, l: 50, r: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    /* global Plotly */
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(plotId, [trace], layout, { responsive: true, displayModeBar: false });
    }
}

/**
 * Render a cluster visualization scatter plot using Plotly.
 * @param {string} plotId - DOM element id for the plot container
 * @param {Object} viz - Visualization data with points and statistics
 */
function _renderClusterVisualizationChart(plotId, viz) {
    const points = viz.points || [];
    if (points.length === 0) return;

    // Group points by cluster
    const clusters = {};
    for (const p of points) {
        let cid = 0;
        if (p.cluster !== undefined) {
            cid = p.cluster;
        } else if (p.cluster_id !== undefined) {
            cid = p.cluster_id;
        }
        if (!clusters[cid]) clusters[cid] = { x: [], y: [], text: [] };
        clusters[cid].x.push(p.x);
        clusters[cid].y.push(p.y);
        clusters[cid].text.push(p.title || '');
    }

    const traces = Object.keys(clusters).map(cid => ({
        x: clusters[cid].x,
        y: clusters[cid].y,
        text: clusters[cid].text,
        mode: 'markers',
        type: 'scatter',
        name: `Cluster ${cid}`,
        marker: { size: 4, opacity: 0.7 },
        hovertemplate: `%{text}<extra>Cluster ${cid}</extra>`
    }));

    const stats = viz.statistics || {};
    const title = `Cluster Visualization (${stats.total_papers || points.length} papers, ${stats.n_clusters || Object.keys(clusters).length} clusters)`;

    const layout = {
        title: title,
        xaxis: { title: '', zeroline: false, showticklabels: false },
        yaxis: { title: '', zeroline: false, showticklabels: false },
        margin: { t: 40, b: 20, l: 20, r: 20 },
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        hovermode: 'closest'
    };

    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(plotId, traces, layout, { responsive: true, displayModeBar: false });
    }
}
