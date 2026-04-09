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
 * Get the current chat donation consent status from localStorage.
 * @returns {boolean|null} true if accepted, false if declined, null if not yet asked
 */
function getChatDonationConsent() {
    const stored = localStorage.getItem('chatDonationConsent');
    if (stored === 'true') return true;
    if (stored === 'false') return false;
    return null;
}

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
 * Display chat papers in the side panel and mobile modal
 * @param {Array} papers - Array of paper objects
 * @param {Object} metadata - Metadata about the search
 */
export function displayChatPapers(papers, metadata = {}) {
    const papersDiv = document.getElementById('chat-papers');
    const modalContent = document.getElementById('papers-modal-content');
    const mobileBtnWrapper = document.getElementById('mobile-papers-btn-wrapper');
    const mobileCount = document.getElementById('mobile-papers-count');

    if (!papers || papers.length === 0) {
        const emptyHtml = renderEmptyState(
            'No papers found for this query',
            '',
            'fa-inbox'
        );
        if (papersDiv) papersDiv.innerHTML = emptyHtml;
        if (modalContent) modalContent.innerHTML = emptyHtml;
        if (mobileBtnWrapper) mobileBtnWrapper.classList.add('hidden');
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

    if (papersDiv) papersDiv.innerHTML = html;
    if (modalContent) modalContent.innerHTML = html;

    // Show the mobile "View Papers" button with paper count
    if (mobileBtnWrapper) {
        mobileBtnWrapper.classList.remove('hidden');
        if (mobileCount) mobileCount.textContent = papers.length;
    }
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
    messageDiv.dataset.role = role;
    messageDiv.innerHTML = `
        <div class="flex items-start gap-3 ${justifyClass}">
            ${!isUser ? `
                <div class="flex-shrink-0 w-8 h-8 ${iconBg} rounded-full flex items-center justify-center text-white">
                    <i class="fas ${icon} text-sm"></i>
                </div>
            ` : ''}
            <div class="${bgColor} rounded-lg p-4 shadow-sm max-w-2xl">
                <div data-chat-content>${contentHtml}</div>
                ${isLoading ? '<div class="spinner mt-2" style="width: 20px; height: 20px; border-width: 2px;"></div>' : ''}
                ${!isUser && !isLoading ? `
                <div class="chat-feedback-buttons flex items-center gap-2 mt-3 pt-2 border-t border-gray-100">
                    <span class="text-xs text-gray-400 mr-1">Helpful?</span>
                    <button class="chat-feedback-btn text-gray-300 hover:text-green-500 transition-colors p-1"
                        data-rating="up" data-msg-id="${messageId}" title="Thumbs up">
                        <i class="fas fa-thumbs-up text-sm"></i>
                    </button>
                    <button class="chat-feedback-btn text-gray-300 hover:text-red-500 transition-colors p-1"
                        data-rating="down" data-msg-id="${messageId}" title="Thumbs down">
                        <i class="fas fa-thumbs-down text-sm"></i>
                    </button>
                </div>
                ` : ''}
            </div>
            ${isUser ? `
                <div class="flex-shrink-0 w-8 h-8 ${iconBg} rounded-full flex items-center justify-center text-white">
                    <i class="fas ${icon} text-sm"></i>
                </div>
            ` : ''}
        </div>
    `;

    // Attach click handlers for feedback buttons
    if (!isUser && !isLoading) {
        const feedbackBtns = messageDiv.querySelectorAll('.chat-feedback-btn');
        feedbackBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                handleChatFeedback(btn.dataset.msgId, btn.dataset.rating);
            });
        });
    }

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

        // Clear papers panel and modal
        const emptyHtml = renderEmptyState(
            'Ask a question to see relevant papers',
            '',
            'fa-inbox'
        );
        const papersDiv = document.getElementById('chat-papers');
        if (papersDiv) papersDiv.innerHTML = emptyHtml;
        const modalContent = document.getElementById('papers-modal-content');
        if (modalContent) modalContent.innerHTML = emptyHtml;

        // Hide mobile papers button
        const mobileBtnWrapper = document.getElementById('mobile-papers-btn-wrapper');
        if (mobileBtnWrapper) mobileBtnWrapper.classList.add('hidden');
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
                    <i class="fas fa-chart-line text-sm"></i>
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
 * Color palette for multi-conference line plots.
 * @type {string[]}
 */
const CONFERENCE_COLORS = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#6366f1', '#0891b2', '#be185d'];

/**
 * Render a topic evolution line chart using Plotly.
 * Supports multiple topics and multiple conferences as separate traces.
 * @param {string} plotId - DOM element id for the plot container
 * @param {Object} viz - Visualization data with topics array (each with conference_data, topic, conferences)
 */
function _renderTopicEvolutionChart(plotId, viz) {
    // Normalize: support both new `topics` array and legacy single-topic format
    const topics = viz.topics || [{
        topic: viz.topic || '',
        conferences: viz.conferences || [],
        conference_data: viz.conference_data || {}
    }];

    const traces = [];
    let colorIdx = 0;
    const allConferences = new Set();
    const allTopicNames = [];

    for (const entry of topics) {
        const conferenceData = entry.conference_data || {};
        const conferences = Object.keys(conferenceData);
        const topicName = entry.topic || '';
        allTopicNames.push(topicName);
        conferences.forEach(c => allConferences.add(c));

        for (const conf of conferences) {
            const cdata = conferenceData[conf] || {};
            const yearRelative = cdata.year_relative || {};
            const years = Object.keys(yearRelative).sort();
            const values = years.map(y => yearRelative[y]);

            // Build trace name: include topic and/or conference as needed
            let name;
            if (topics.length > 1 && allConferences.size > 1) {
                name = `${topicName} (${conf})`;
            } else if (topics.length > 1) {
                name = topicName;
            } else {
                name = conf;
            }

            traces.push({
                x: years.map(Number),
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: name,
                line: { color: CONFERENCE_COLORS[colorIdx % CONFERENCE_COLORS.length], width: 2 },
                marker: { size: 6 }
            });
            colorIdx++;
        }
    }

    if (traces.length === 0) return;

    // Build title
    const uniqueConfs = [...allConferences];
    const confLabel = uniqueConfs.length === 1
        ? ` (${uniqueConfs[0]})`
        : uniqueConfs.length > 1
            ? ` (${uniqueConfs.join(', ')})`
            : '';
    const topicLabel = allTopicNames.length === 1
        ? allTopicNames[0]
        : allTopicNames.join(', ');

    const layout = {
        title: { text: `Topic Evolution: ${topicLabel}${confLabel}` },
        xaxis: { title: { text: 'Year' }, type: 'linear', automargin: true, dtick: 1 },
        yaxis: { title: { text: 'Percentage of Papers (%)' }, automargin: true },
        margin: { t: 50, b: 60, l: 80, r: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: traces.length > 1
    };

    /* global Plotly */
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(plotId, traces, layout, { responsive: true, displayModeBar: false });
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

/**
 * Collect the current chat transcript from the DOM.
 * @returns {Array<{role: string, text: string}>} Array of message objects
 */
function collectChatTranscript() {
    const messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return [];

    const messages = [];
    const messageDivs = messagesDiv.querySelectorAll('.chat-message[data-role]');

    for (const msgDiv of messageDivs) {
        const role = msgDiv.dataset.role;
        const contentEl = msgDiv.querySelector('[data-chat-content]');

        if (contentEl) {
            messages.push({
                role: role,
                text: contentEl.textContent.trim()
            });
        }
    }

    return messages;
}

/**
 * Handle chat feedback (thumbs up/down) button click.
 * Shows a consent popup on first use, then sends the transcript to the server.
 * @param {string} messageId - The ID of the message being rated
 * @param {string} rating - 'up' or 'down'
 * @async
 */
export async function handleChatFeedback(messageId, rating) {
    // Check consent status from localStorage
    let consent = getChatDonationConsent();

    // If consent has not been asked yet, show the consent popup
    if (consent === null) {
        const accepted = confirm(
            'Thank you for your feedback! 🎉\n\n' +
            'To help us improve, clicking this button will upload your current chat conversation.\n\n' +
            '✓ Your data will be fully anonymized\n' +
            '✓ No personal information will be collected\n' +
            '✓ Data will only be used to improve chat quality\n\n' +
            'Do you agree to share your chat conversation?'
        );

        consent = accepted;
        localStorage.setItem('chatDonationConsent', String(accepted));

        if (!accepted) {
            return;
        }
    }

    // If user previously declined, do nothing
    if (consent === false) {
        return;
    }

    // Collect transcript and send to backend
    const transcript = collectChatTranscript();
    if (transcript.length === 0) {
        return;
    }

    // Disable the feedback buttons for this message to prevent duplicate submissions
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
        if (feedbackDiv) {
            const buttons = feedbackDiv.querySelectorAll('.chat-feedback-btn');
            buttons.forEach(btn => { btn.disabled = true; });

            // Highlight the selected rating button
            const selectedBtn = feedbackDiv.querySelector(`[data-rating="${rating}"]`);
            if (selectedBtn) {
                selectedBtn.classList.remove('text-gray-300');
                selectedBtn.classList.add(rating === 'up' ? 'text-green-500' : 'text-red-500');
            }
        }
    }

    try {
        const response = await fetch(`${API_BASE}/api/donate-chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ rating, transcript })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to send feedback');
        }

        // Show a brief "thank you" in place of the buttons
        if (messageDiv) {
            const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
            if (feedbackDiv) {
                feedbackDiv.innerHTML = '<span class="text-xs text-gray-400">Thanks for your feedback!</span>';
            }
        }
    } catch (error) {
        console.error('Error sending chat feedback:', error);
        // Re-enable buttons on error
        if (messageDiv) {
            const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
            if (feedbackDiv) {
                const buttons = feedbackDiv.querySelectorAll('.chat-feedback-btn');
                buttons.forEach(btn => { btn.disabled = false; });
            }
        }
    }
}

/**
 * Open the relevant papers modal (used on narrow screens).
 */
export function openPapersModal() {
    const modal = document.getElementById('papers-modal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }
}

/**
 * Close the relevant papers modal.
 */
export function closePapersModal() {
    const modal = document.getElementById('papers-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}
