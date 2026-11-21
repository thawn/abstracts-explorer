/**
 * Chat Module
 *
 * Handles chat functionality including message sending, display,
 * and conversation management.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml, getSelectedConference, getSelectedYears } from './utils/dom-utils.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { renderEmptyState } from './utils/ui-utils.js';
import { setCurrentSearchTerm, getAllPaperPriorities } from './state.js';
import { formatPaperCard } from './paper-card.js';
import { renderMarkdownWithLatex } from './utils/markdown-utils.js';
import type { Paper, Visualization, ChatMetadata } from './types.js';
import { ensurePlotly, getPlotly } from './plotly-loader.js';

// Plotly is loaded lazily (and tests inject a global mock). The proxy defers
// resolution to call time so the existing `Plotly.xxx` call sites keep working.
const Plotly: any = new Proxy({}, {
    get(_target, prop) {
        const impl = getPlotly();
        const value = impl[prop];
        return typeof value === 'function' ? value.bind(impl) : value;
    }
});

/** Whether the user has sent at least one chat message in this session. */
let _hasUserSentMessage = false;

/** Whether the feedback highlight animation has already been triggered. */
let _feedbackHighlightShown = false;

/** Monotonically increasing counter used to generate unique message IDs. */
let _messageIdCounter = 0;

function getChatDonationConsent(): boolean | null {
    const stored = localStorage.getItem('chatDonationConsent');
    if (stored === 'true') return true;
    if (stored === 'false') return false;
    return null;
}

/**
 * Build HTML for the MCP tools hint bubble.
 */
export function buildMcpToolsHintHtml(): string {
    return `
        <div id="mcp-tools-hint" class="mcp-tools-hint bg-gradient-to-br from-purple-50 dark:from-purple-900/30 to-blue-50 dark:to-blue-900/30 border border-purple-200 dark:border-purple-700 rounded-lg p-4 shadow-sm max-w-2xl mx-auto">
            <div class="flex items-start gap-2 mb-3">
                <i class="fas fa-lightbulb text-purple-500 mt-0.5"></i>
                <p class="text-sm font-semibold text-gray-700 dark:text-gray-300">I can use specialized tools to help answer your questions. Try asking:</p>
            </div>
            <ul class="space-y-2 text-sm text-gray-600 dark:text-gray-400 ml-6">
                <li class="flex items-start gap-2">
                    <i class="fas fa-chart-pie text-purple-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"What are the main topics?"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-chart-line text-blue-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"How has the topic of large language models evolved over the years?"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-hexagon-nodes text-yellow-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Show me a visual overview of how topics are clustered."</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-search text-green-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Find papers about reinforcement learning at ICLR"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-fire text-orange-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"How relevant is uncertainty quantification at NeurIPS?"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-info-circle text-indigo-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Who are the authors of the paper titled 'Large Language Diffusion Models'?"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-star text-yellow-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Give me a summary about the papers I marked as interesting for ICLR'26!"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-wand-magic-sparkles text-pink-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Based on my interesting papers, suggest other papers that may interest me."</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-users text-teal-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Who should I meet based on my interesting papers?"</em></span>
                </li>
                <li class="flex items-start gap-2">
                    <i class="fas fa-id-card text-cyan-400 mt-0.5 flex-shrink-0"></i>
                    <span><em>"Suggest search terms based on the ORCID record 0000-0002-2315-9247."</em></span>
                </li>
            </ul>
        </div>`;
}

/**
 * Remove the MCP tools hint bubble from the chat area (if present).
 */
export function removeMcpToolsHint(): void {
    const hint = document.getElementById('mcp-tools-hint');
    if (hint) {
        hint.classList.add('mcp-tools-hint-hide');
        let removed = false;
        const remove = () => {
            if (!removed) {
                removed = true;
                hint.remove();
            }
        };
        hint.addEventListener('animationend', remove, { once: true });
        setTimeout(remove, 500);
    }
}

/**
 * Insert the MCP tools hint bubble into the chat messages area.
 */
export function initMcpToolsHint(): void {
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv && !document.getElementById('mcp-tools-hint')) {
        messagesDiv.insertAdjacentHTML('beforeend', buildMcpToolsHintHtml());
    }
}

/**
 * Send chat message.
 */
export async function sendChatMessage(): Promise<void> {
    const input = document.getElementById('chat-input') as HTMLInputElement;
    const message = input.value.trim();

    if (!message) return;

    if (!_hasUserSentMessage) {
        _hasUserSentMessage = true;
        removeMcpToolsHint();
    }

    setCurrentSearchTerm(message);

    addChatMessage(message, 'user');
    input.value = '';

    const loadingId = addChatMessage('Thinking...', 'assistant', true);

    try {
        const nPapers = parseInt((document.getElementById('n-papers') as HTMLSelectElement).value, 10);

        const chatSessionSelect = document.getElementById('chat-session-filter') as HTMLSelectElement;
        const sessions = Array.from(chatSessionSelect.selectedOptions).map(opt => opt.value);

        const selectedYears = getSelectedYears();
        const selectedConference = getSelectedConference();

        const requestBody: Record<string, any> = {
            message,
            n_papers: nPapers
        };

        const paperPriorities = getAllPaperPriorities();
        if (paperPriorities && Object.keys(paperPriorities).length > 0) {
            requestBody.paper_priorities = paperPriorities;
        }

        if (sessions.length > 0 && sessions.length < chatSessionSelect.options.length) {
            requestBody.sessions = sessions;
        }

        if (selectedYears.length > 0) {
            requestBody.years = selectedYears;
        }

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

        const data = await parseJsonResponse(response);

        document.getElementById(loadingId)?.remove();

        if (data.error) {
            addChatMessage(data.error, 'assistant');
            return;
        }

        let responseText: string;
        let papers: Paper[];
        let metadata: ChatMetadata;
        let visualizations: Visualization[];
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

        if (metadata.rewritten_query) {
            setCurrentSearchTerm(metadata.rewritten_query);
        }

        addChatMessage(responseText, 'assistant');

        if (visualizations.length > 0) {
            await ensurePlotly();
            renderChatVisualizations(visualizations);
        }

        displayChatPapers(papers, metadata);
    } catch (error) {
        console.error('Chat error:', error);
        document.getElementById(loadingId)?.remove();
        const message = error && (error as Error).message
            ? `Sorry, I couldn't complete that request. ${(error as Error).message}`
            : "Sorry, I couldn't complete that request. Please try again.";
        addChatMessage(message, 'assistant');
    }
}

/**
 * Display chat papers in the side panel and mobile modal.
 */
export function displayChatPapers(papers: Paper[], metadata: ChatMetadata = {}): void {
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

    if (metadata.rewritten_query) {
        const wasRetrieved = metadata.retrieved_new_papers !== false;
        const cacheIcon = wasRetrieved ? 'fa-sync-alt' : 'fa-check-circle';
        const cacheColor = wasRetrieved ? 'text-blue-600' : 'text-green-600';
        const cacheText = wasRetrieved ? 'Retrieved new papers' : 'Using cached papers';

        html += `
            <div class="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-200 dark:border-purple-700 rounded-lg p-4 mb-4 shadow-sm">
                <div class="flex items-start gap-2 mb-2">
                    <i class="fas fa-magic text-purple-600 mt-1"></i>
                    <div class="flex-1">
                        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">Optimized Search Query</h3>
                        <p class="text-sm text-gray-800 dark:text-gray-200 font-medium italic">"${escapeHtml(metadata.rewritten_query)}"</p>
                    </div>
                </div>
                <div class="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 mt-2 pt-2 border-t border-purple-200 dark:border-purple-700">
                    <i class="fas ${cacheIcon} ${cacheColor}"></i>
                    <span>${cacheText}</span>
                    <span class="ml-auto">${papers.length} paper${papers.length !== 1 ? 's' : ''} found</span>
                </div>
            </div>
        `;
    }

    papers.forEach((paper, index) => {
        const citationNumber = typeof paper.citation_number === 'number' ? paper.citation_number : index + 1;
        html += formatPaperCard(paper, {
            compact: true,
            showNumber: citationNumber,
            idPrefix: `paper-${citationNumber}`
        });
    });

    if (papersDiv) papersDiv.innerHTML = html;
    if (modalContent) modalContent.innerHTML = html;

    if (mobileBtnWrapper) {
        mobileBtnWrapper.classList.remove('hidden');
        if (mobileCount) mobileCount.textContent = String(papers.length);
    }
}

/**
 * Add chat message to conversation.
 *
 * @returns Message ID
 */
export function addChatMessage(text: string, role: string, isLoading = false): string {
    const messagesDiv = document.getElementById('chat-messages') as HTMLElement;
    const messageId = `msg-${++_messageIdCounter}`;

    const isUser = role === 'user';
    const bgColor = isUser ? 'bg-purple-600 text-white' : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200';
    const iconBg = isUser ? 'bg-gray-600' : 'bg-purple-600';
    const icon = isUser ? 'fa-user' : 'fa-robot';
    const justifyClass = isUser ? 'justify-end' : 'justify-start';

    const contentHtml = isUser
        ? `<p class="whitespace-pre-wrap">${escapeHtml(text)}</p>`
        : `<div class="markdown-content">${renderMarkdownWithLatex(text)}</div>`;

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
                <div class="chat-feedback-buttons flex items-center gap-2 mt-3 pt-2 border-t border-gray-100 dark:border-gray-600">
                    <span class="text-s text-gray-500 dark:text-gray-400 mr-1">Helpful?</span>
                    <button class="chat-feedback-btn text-gray-400 hover:text-green-600 transition-colors p-1"
                        data-rating="up" data-msg-id="${messageId}" title="Thumbs up">
                        <i class="fas fa-thumbs-up text-s"></i>
                    </button>
                    <button class="chat-feedback-btn text-gray-400 hover:text-red-600 transition-colors p-1"
                        data-rating="down" data-msg-id="${messageId}" title="Thumbs down">
                        <i class="fas fa-thumbs-down text-s"></i>
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

    if (!isUser && !isLoading) {
        const feedbackBtns = messageDiv.querySelectorAll('.chat-feedback-btn');
        feedbackBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const el = btn as HTMLElement;
                handleChatFeedback(el.dataset.msgId as string, el.dataset.rating as string);
            });
        });
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (!isUser && !isLoading && !_feedbackHighlightShown) {
        _feedbackHighlightShown = true;
        const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
        if (feedbackDiv) {
            feedbackDiv.classList.add('feedback-highlight');
            const buttons = feedbackDiv.querySelectorAll('.chat-feedback-btn');
            buttons.forEach((btn, index) => {
                (btn as HTMLElement).style.animationDelay = `${index * 0.15 + 0.5}s`;
                btn.classList.add('feedback-bounce');
            });
        }
    }

    return messageId;
}

/**
 * Reset chat conversation.
 */
export async function resetChat(): Promise<void> {
    try {
        const response = await fetch(`${API_BASE}/api/chat/reset`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error(`Reset failed (HTTP ${response.status})`);
        }

        const messagesDiv = document.getElementById('chat-messages') as HTMLElement;
        messagesDiv.innerHTML = '';
        const conf = getSelectedConference() || 'conference';
        addChatMessage(`Conversation reset. How can I help you explore ${conf} abstracts?`, 'assistant');

        _hasUserSentMessage = false;
        initMcpToolsHint();

        const emptyHtml = renderEmptyState(
            'Ask a question to see relevant papers',
            '',
            'fa-inbox'
        );
        const papersDiv = document.getElementById('chat-papers');
        if (papersDiv) papersDiv.innerHTML = emptyHtml;
        const modalContent = document.getElementById('papers-modal-content');
        if (modalContent) modalContent.innerHTML = emptyHtml;

        const mobileBtnWrapper = document.getElementById('mobile-papers-btn-wrapper');
        if (mobileBtnWrapper) mobileBtnWrapper.classList.add('hidden');
    } catch (error) {
        console.error('Error resetting chat:', error);
    }
}

/**
 * Render visualizations returned by MCP tools in the chat area.
 */
export function renderChatVisualizations(visualizations: Visualization[]): void {
    const messagesDiv = document.getElementById('chat-messages') as HTMLElement;

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
                <div class="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm w-full max-w-2xl">
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
 */
const CONFERENCE_COLORS = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#6366f1', '#0891b2', '#be185d'];

/**
 * Render a topic evolution line chart using Plotly.
 */
function _renderTopicEvolutionChart(plotId: string, viz: Visualization): void {
    const topics = viz.topics || [];
    const topicConferenceData = viz.conference_data || {};

    const traces: any[] = [];
    let colorIdx = 0;
    const allConferences = new Set<string>();

    for (const topicName of topics) {
        const conferenceData = topicConferenceData[topicName] || {};
        const conferences = Object.keys(conferenceData);
        conferences.forEach(c => allConferences.add(c));

        for (const conf of conferences) {
            const cdata = conferenceData[conf] || {};
            const yearRelative = cdata.year_relative || {};
            const years = Object.keys(yearRelative).sort();
            const values = years.map(y => yearRelative[y]);

            let name: string;
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

    const uniqueConfs = [...allConferences];
    const confLabel = uniqueConfs.length === 1
        ? ` (${uniqueConfs[0]})`
        : uniqueConfs.length > 1
            ? ` (${uniqueConfs.join(', ')})`
            : '';
    const topicLabel = topics.length === 1
        ? topics[0]
        : topics.join(', ');

    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const fontColor = isDark ? '#e5e7eb' : '#374151';

    const layout = {
        title: { text: `Topic Evolution: ${topicLabel}${confLabel}` },
        xaxis: { title: { text: 'Year' }, type: 'linear', automargin: true, dtick: 1, showgrid: false, zeroline: false },
        yaxis: { title: { text: 'Percentage of Papers (%)' }, automargin: true, showgrid: false, zeroline: false },
        margin: { t: 50, b: 60, l: 80, r: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: fontColor },
        showlegend: traces.length > 1
    };

    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(plotId, traces, layout, { responsive: true, displayModeBar: false });
    }
}

/**
 * Render a cluster visualization scatter plot using Plotly.
 */
function _renderClusterVisualizationChart(plotId: string, viz: Visualization): void {
    const points = viz.points || [];
    if (points.length === 0) return;

    const clusters: Record<string, { x: number[]; y: number[]; text: string[] }> = {};
    for (const p of points) {
        let cid: number | string = 0;
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

    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const fontColor = isDark ? '#e5e7eb' : '#374151';

    const layout = {
        title: title,
        xaxis: { title: '', zeroline: false, showticklabels: false, showgrid: false },
        yaxis: { title: '', zeroline: false, showticklabels: false, showgrid: false },
        margin: { t: 40, b: 20, l: 20, r: 20 },
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: fontColor },
        hovermode: 'closest'
    };

    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(plotId, traces, layout, { responsive: true, displayModeBar: false });
    }
}

/**
 * Collect the current chat transcript from the DOM.
 */
function collectChatTranscript(): Array<{ role: string; text: string }> {
    const messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return [];

    const messages: Array<{ role: string; text: string }> = [];
    const messageDivs = messagesDiv.querySelectorAll('.chat-message[data-role]');

    for (const msgDiv of messageDivs) {
        const role = (msgDiv as HTMLElement).dataset.role;
        const contentEl = msgDiv.querySelector('[data-chat-content]');

        if (contentEl) {
            messages.push({
                role: role as string,
                text: contentEl.textContent!.trim()
            });
        }
    }

    return messages;
}

/**
 * Handle chat feedback (thumbs up/down) button click.
 */
export async function handleChatFeedback(messageId: string, rating: string): Promise<void> {
    let consent = getChatDonationConsent();

    if (consent === null) {
        const accepted = confirm(
            'Thank you for your feedback! 🎉❤️\n\n' +
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

    if (consent === false) {
        return;
    }

    const transcript = collectChatTranscript();
    if (transcript.length === 0) {
        return;
    }

    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
        if (feedbackDiv) {
            const buttons = feedbackDiv.querySelectorAll('.chat-feedback-btn');
            buttons.forEach(btn => { (btn as HTMLButtonElement).disabled = true; });

            const selectedBtn = feedbackDiv.querySelector(`[data-rating="${rating}"]`);
            if (selectedBtn) {
                selectedBtn.classList.remove('text-gray-400');
                selectedBtn.classList.add(rating === 'up' ? 'text-green-600' : 'text-red-600');
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

        await parseJsonResponse(response);

        if (messageDiv) {
            const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
            if (feedbackDiv) {
                feedbackDiv.innerHTML = '<span class="text-s text-gray-500">Thanks for your feedback!  🎉❤️</span>';
            }
        }
    } catch (error) {
        console.error('Error sending chat feedback:', error);
        if (messageDiv) {
            const feedbackDiv = messageDiv.querySelector('.chat-feedback-buttons');
            if (feedbackDiv) {
                const buttons = feedbackDiv.querySelectorAll('.chat-feedback-btn');
                buttons.forEach(btn => { (btn as HTMLButtonElement).disabled = false; });
                const errorSpan = document.createElement('span');
                errorSpan.className = 'text-s text-red-500';
                errorSpan.textContent = (error as Error).message || 'Could not send feedback. Please try again.';
                feedbackDiv.appendChild(errorSpan);
            }
        }
    }
}

/**
 * Open the relevant papers modal (used on narrow screens).
 */
export function openPapersModal(): void {
    const modal = document.getElementById('papers-modal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }
}

/**
 * Close the relevant papers modal.
 */
export function closePapersModal(): void {
    const modal = document.getElementById('papers-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

/**
 * Reset internal module state. Intended for use in tests only.
 */
export function _resetChatState(): void {
    _hasUserSentMessage = false;
    _feedbackHighlightShown = false;
    _messageIdCounter = 0;
}

/**
 * Re-apply font colour to every active Plotly chart inside the chat message area.
 */
function _refreshChatPlotColors(): void {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const fontColor = isDark ? '#e5e7eb' : '#374151';
    const messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv || typeof Plotly === 'undefined') return;
    messagesDiv.querySelectorAll('.js-plotly-plot').forEach(function (el) {
        Plotly.relayout(el, { 'font.color': fontColor });
    });
}

if (typeof window !== 'undefined' && window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', _refreshChatPlotColors);
}
