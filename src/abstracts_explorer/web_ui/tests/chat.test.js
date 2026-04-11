/**
 * Tests for chat module
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.fetch = jest.fn();
global.marked = { parse: jest.fn((text) => text), use: jest.fn() };
global.Plotly = { newPlot: jest.fn() };

import { sendChatMessage, displayChatPapers, addChatMessage, resetChat, renderChatVisualizations, openPapersModal, closePapersModal, handleChatFeedback, buildMcpToolsHintHtml, removeMcpToolsHint, initMcpToolsHint, _resetChatState } from '../static/modules/chat.js';
import * as State from '../static/modules/state.js';

describe('Chat Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorage.clear();
        _resetChatState();
        document.body.innerHTML = `
            <input id="chat-input" value="test message" />
            <select id="n-papers"><option value="3" selected>3</option></select>
            <select id="chat-session-filter" multiple>
                <option value="Session 1" selected>Session 1</option>
            </select>
            <select id="year-selector"><option value="2025">2025</option></select>
            <select id="conference-selector"><option value="">All</option></select>
            <div id="chat-messages"></div>
            <div id="chat-papers"></div>
            <div id="papers-modal" class="hidden"></div>
            <div id="papers-modal-content"></div>
            <div id="mobile-papers-btn-wrapper" class="hidden"></div>
            <span id="mobile-papers-count"></span>
        `;
    });

    describe('sendChatMessage', () => {
        it('should not send empty message', async () => {
            document.getElementById('chat-input').value = '  ';
            
            await sendChatMessage();
            
            expect(global.fetch).not.toHaveBeenCalled();
        });

        it('should add user message to chat', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: {
                        response: 'AI response',
                        papers: [],
                        metadata: {}
                    }
                })
            });

            await sendChatMessage();

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('test message');
        });

        it('should clear input after sending', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: {
                        response: 'AI response',
                        papers: []
                    }
                })
            });

            const input = document.getElementById('chat-input');
            await sendChatMessage();

            expect(input.value).toBe('');
        });

        it('should send correct request', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: { response: 'OK', papers: [] }
                })
            });

            await sendChatMessage();

            expect(global.fetch).toHaveBeenCalledWith(
                '/api/chat',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
            );

            const body = JSON.parse(global.fetch.mock.calls[0][1].body);
            expect(body.message).toBe('test message');
            expect(body.n_papers).toBe(3);
        });

        it('should handle error response', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    error: 'Chat error'
                })
            });

            await sendChatMessage();

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('Chat error');
        });

        it('should set current search term', async () => {
            State.setCurrentSearchTerm('');
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: { response: 'OK', papers: [] }
                })
            });

            await sendChatMessage();

            expect(State.getCurrentSearchTerm()).toBe('test message');
        });
    });

    describe('displayChatPapers', () => {
        it('should show empty state when no papers', () => {
            displayChatPapers([]);

            const papers = document.getElementById('chat-papers');
            expect(papers.innerHTML).toContain('No papers found');
        });

        it('should display papers with metadata', () => {
            const testPapers = [
                {
                    uid: 'p1',
                    title: 'Test Paper',
                    authors: ['Author 1'],
                    abstract: 'Test abstract'
                }
            ];
            const metadata = {
                rewritten_query: 'optimized query',
                retrieved_new_papers: true
            };

            displayChatPapers(testPapers, metadata);

            const papers = document.getElementById('chat-papers');
            expect(papers.innerHTML).toContain('optimized query');
            expect(papers.innerHTML).toContain('Test Paper');
        });

        it('should show cache status', () => {
            const testPapers = [{ uid: 'p1', title: 'Paper', authors: [] }];
            const metadata = {
                rewritten_query: 'query',
                retrieved_new_papers: false
            };

            displayChatPapers(testPapers, metadata);

            const papers = document.getElementById('chat-papers');
            expect(papers.innerHTML).toContain('Using cached papers');
        });
    });

    describe('addChatMessage', () => {
        it('should add user message', () => {
            const messageId = addChatMessage('User text', 'user');

            expect(messageId).toMatch(/^msg-\d+$/);
            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('User text');
        });

        it('should add assistant message', () => {
            addChatMessage('Assistant text', 'assistant');

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('Assistant text');
        });

        it('should show loading indicator', () => {
            addChatMessage('Loading...', 'assistant', true);

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('spinner');
        });

        it('should escape HTML in user messages', () => {
            addChatMessage('<script>alert("xss")</script>', 'user');

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).not.toContain('<script>');
            expect(messages.innerHTML).toContain('&lt;script&gt;');
        });
    });

    describe('resetChat', () => {
        it('should call reset endpoint', async () => {
            global.fetch.mockResolvedValueOnce({ ok: true });

            await resetChat();

            expect(global.fetch).toHaveBeenCalledWith(
                '/api/chat/reset',
                expect.objectContaining({ method: 'POST' })
            );
        });

        it('should clear chat messages', async () => {
            document.getElementById('chat-messages').innerHTML = '<div>Old message</div>';
            global.fetch.mockResolvedValueOnce({ ok: true });

            await resetChat();

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).not.toContain('Old message');
        });

        it('should add reset confirmation message', async () => {
            global.fetch.mockResolvedValueOnce({ ok: true });

            await resetChat();

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('Conversation reset');
        });
    });

    describe('renderChatVisualizations', () => {
        beforeEach(() => {
            global.Plotly = { newPlot: jest.fn() };
        });

        it('should render topic evolution chart', () => {
            const visualizations = [{
                type: 'topic_evolution',
                topics: ['transformers'],
                conference_data: {
                    'transformers': {
                        'NeurIPS': {
                            year_relative: { '2022': 2.5, '2023': 4.0, '2024': 5.0 },
                            year_counts: { '2022': 5, '2023': 10, '2024': 15 }
                        }
                    }
                }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const [plotId, traces, layout] = global.Plotly.newPlot.mock.calls[0];
            expect(plotId).toMatch(/^chat-plot-/);
            expect(traces).toHaveLength(1);
            expect(traces[0].x).toEqual([2022, 2023, 2024]);
            expect(traces[0].y).toEqual([2.5, 4.0, 5.0]);
            expect(traces[0].type).toBe('scatter');
            expect(traces[0].mode).toBe('lines+markers');
            expect(traces[0].name).toBe('NeurIPS');
            expect(layout.title.text).toContain('transformers');
            expect(layout.title.text).toContain('NeurIPS');
            expect(layout.xaxis.title.text).toContain('Year');
            expect(layout.xaxis.type).toBe('linear');
            expect(layout.yaxis.title.text).toContain('Percentage');
        });

        it('should render multiple topics in the same chart', () => {
            const visualizations = [{
                type: 'topic_evolution',
                topics: ['transformers', 'reinforcement learning'],
                conference_data: {
                    'transformers': {
                        'NeurIPS': {
                            year_relative: { '2022': 2.5, '2023': 4.0 },
                            year_counts: { '2022': 5, '2023': 10 }
                        }
                    },
                    'reinforcement learning': {
                        'NeurIPS': {
                            year_relative: { '2022': 6.0, '2023': 3.2 },
                            year_counts: { '2022': 12, '2023': 8 }
                        }
                    }
                }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const [plotId, traces, layout] = global.Plotly.newPlot.mock.calls[0];
            expect(traces).toHaveLength(2);
            expect(traces[0].name).toBe('transformers');
            expect(traces[0].x).toEqual([2022, 2023]);
            expect(traces[0].y).toEqual([2.5, 4.0]);
            expect(traces[1].name).toBe('reinforcement learning');
            expect(traces[1].x).toEqual([2022, 2023]);
            expect(traces[1].y).toEqual([6.0, 3.2]);
            expect(layout.title.text).toContain('transformers');
            expect(layout.title.text).toContain('reinforcement learning');
            expect(layout.showlegend).toBe(true);
        });

        it('should render multiple topics across multiple conferences', () => {
            const visualizations = [{
                type: 'topic_evolution',
                topics: ['transformers', 'reinforcement learning'],
                conference_data: {
                    'transformers': {
                        'NeurIPS': {
                            year_relative: { '2022': 2.5, '2023': 4.0 },
                        },
                        'ICLR': {
                            year_relative: { '2022': 1.8, '2023': 3.5 },
                        }
                    },
                    'reinforcement learning': {
                        'NeurIPS': {
                            year_relative: { '2022': 6.0, '2023': 3.2 },
                        },
                        'ICLR': {
                            year_relative: { '2022': 5.0, '2023': 2.8 },
                        }
                    }
                }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const [plotId, traces, layout] = global.Plotly.newPlot.mock.calls[0];
            // 2 topics × 2 conferences = 4 traces
            expect(traces).toHaveLength(4);
            expect(traces[0].name).toBe('transformers (NeurIPS)');
            expect(traces[0].x).toEqual([2022, 2023]);
            expect(traces[0].y).toEqual([2.5, 4.0]);
            expect(traces[1].name).toBe('transformers (ICLR)');
            expect(traces[1].x).toEqual([2022, 2023]);
            expect(traces[1].y).toEqual([1.8, 3.5]);
            expect(traces[2].name).toBe('reinforcement learning (NeurIPS)');
            expect(traces[3].name).toBe('reinforcement learning (ICLR)');
            expect(layout.title.text).toContain('transformers');
            expect(layout.title.text).toContain('reinforcement learning');
            expect(layout.title.text).toContain('NeurIPS');
            expect(layout.title.text).toContain('ICLR');
            expect(layout.showlegend).toBe(true);
        });

        it('should render cluster visualization chart', () => {
            const visualizations = [{
                type: 'cluster_visualization',
                points: [
                    { x: 1.0, y: 2.0, cluster: 0, title: 'Paper A' },
                    { x: 3.0, y: 4.0, cluster: 1, title: 'Paper B' },
                    { x: 1.5, y: 2.5, cluster: 0, title: 'Paper C' }
                ],
                statistics: { n_clusters: 2, total_papers: 3 }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const [plotId, traces, layout] = global.Plotly.newPlot.mock.calls[0];
            expect(plotId).toMatch(/^chat-plot-/);
            // Two clusters should produce two traces
            expect(traces).toHaveLength(2);
            expect(layout.title).toContain('3 papers');
            expect(layout.title).toContain('2 clusters');
        });

        it('should render multiple visualizations', () => {
            const visualizations = [
                {
                    type: 'topic_evolution',
                    topics: ['rl'],
                    conference_data: {
                        'rl': {
                            'ICML': { year_relative: { '2023': 3.0 }, year_counts: { '2023': 3 } }
                        }
                    }
                },
                { type: 'cluster_visualization', points: [{ x: 0, y: 0, cluster: 0 }], statistics: {} }
            ];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(2);
        });

        it('should add chart message elements to chat', () => {
            const visualizations = [{
                type: 'topic_evolution',
                topics: ['gnn'],
                conference_data: {
                    'gnn': {
                        'NeurIPS': { year_relative: { '2023': 2.0 }, year_counts: { '2023': 2 } }
                    }
                }
            }];

            renderChatVisualizations(visualizations);

            const messages = document.getElementById('chat-messages');
            expect(messages.querySelectorAll('.chat-message').length).toBe(1);
            expect(messages.innerHTML).toContain('fa-chart-line');
        });

        it('should skip cluster chart when points array is empty', () => {
            const visualizations = [{
                type: 'cluster_visualization',
                points: [],
                statistics: {}
            }];

            renderChatVisualizations(visualizations);

            // Plotly should not be called for empty points
            expect(global.Plotly.newPlot).not.toHaveBeenCalled();
        });

        it('should handle cluster_id key in points', () => {
            const visualizations = [{
                type: 'cluster_visualization',
                points: [
                    { x: 1.0, y: 2.0, cluster_id: 5, title: 'Paper X' }
                ],
                statistics: { n_clusters: 1, total_papers: 1 }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const traces = global.Plotly.newPlot.mock.calls[0][1];
            expect(traces).toHaveLength(1);
            expect(traces[0].name).toBe('Cluster 5');
        });
    });

    describe('sendChatMessage with visualizations', () => {
        it('should render visualizations from response', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: {
                        response: 'Topic evolution analysis',
                        papers: [],
                        metadata: {},
                        visualizations: [{
                            type: 'topic_evolution',
                            topics: ['attention'],
                            conference_data: {
                                'attention': {
                                    'NeurIPS': { year_relative: { '2023': 5.0 }, year_counts: { '2023': 5 } }
                                }
                            }
                        }]
                    }
                })
            });

            await sendChatMessage();

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
        });

        it('should not call Plotly when no visualizations', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: {
                        response: 'Just text',
                        papers: [],
                        metadata: {},
                        visualizations: []
                    }
                })
            });

            await sendChatMessage();

            expect(global.Plotly.newPlot).not.toHaveBeenCalled();
        });
    });

    describe('displayChatPapers - mobile modal sync', () => {
        it('should populate papers-modal-content alongside chat-papers', () => {
            const papers = [{ uid: 'p1', title: 'Modal Paper', authors: [] }];
            displayChatPapers(papers);

            const modalContent = document.getElementById('papers-modal-content');
            expect(modalContent.innerHTML).toContain('Modal Paper');
        });

        it('should show mobile papers button when papers are loaded', () => {
            const papers = [{ uid: 'p1', title: 'Paper', authors: [] }];
            displayChatPapers(papers);

            const btn = document.getElementById('mobile-papers-btn-wrapper');
            expect(btn.classList.contains('hidden')).toBe(false);
        });

        it('should update mobile papers count', () => {
            const papers = [
                { uid: 'p1', title: 'Paper 1', authors: [] },
                { uid: 'p2', title: 'Paper 2', authors: [] }
            ];
            displayChatPapers(papers);

            const count = document.getElementById('mobile-papers-count');
            expect(count.textContent).toBe('2');
        });

        it('should hide mobile papers button when no papers', () => {
            // First show it
            const papers = [{ uid: 'p1', title: 'Paper', authors: [] }];
            displayChatPapers(papers);
            expect(document.getElementById('mobile-papers-btn-wrapper').classList.contains('hidden')).toBe(false);

            // Then clear it
            displayChatPapers([]);
            expect(document.getElementById('mobile-papers-btn-wrapper').classList.contains('hidden')).toBe(true);
        });
    });

    describe('openPapersModal / closePapersModal', () => {
        it('should show the papers modal', () => {
            openPapersModal();
            const modal = document.getElementById('papers-modal');
            expect(modal.classList.contains('hidden')).toBe(false);
            expect(modal.classList.contains('flex')).toBe(true);
        });

        it('should hide the papers modal', () => {
            openPapersModal();
            closePapersModal();
            const modal = document.getElementById('papers-modal');
            expect(modal.classList.contains('hidden')).toBe(true);
            expect(modal.classList.contains('flex')).toBe(false);
        });
    });

    describe('resetChat - mobile modal', () => {
        it('should hide mobile papers button on reset', async () => {
            // Show the button first
            const papers = [{ uid: 'p1', title: 'Paper', authors: [] }];
            displayChatPapers(papers);
            expect(document.getElementById('mobile-papers-btn-wrapper').classList.contains('hidden')).toBe(false);

            global.fetch.mockResolvedValueOnce({ ok: true });
            await resetChat();

            expect(document.getElementById('mobile-papers-btn-wrapper').classList.contains('hidden')).toBe(true);
        });

        it('should clear papers-modal-content on reset', async () => {
            const papers = [{ uid: 'p1', title: 'Paper', authors: [] }];
            displayChatPapers(papers);
            expect(document.getElementById('papers-modal-content').innerHTML).toContain('Paper');

            global.fetch.mockResolvedValueOnce({ ok: true });
            await resetChat();

            expect(document.getElementById('papers-modal-content').innerHTML).not.toContain('Paper');
        });
    });

    describe('addChatMessage - feedback buttons', () => {
        it('should show feedback buttons on assistant messages', () => {
            addChatMessage('Test response', 'assistant');

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).toContain('chat-feedback-buttons');
            expect(messages.innerHTML).toContain('fa-thumbs-up');
            expect(messages.innerHTML).toContain('fa-thumbs-down');
            expect(messages.innerHTML).toContain('Helpful?');
        });

        it('should not show feedback buttons on user messages', () => {
            addChatMessage('User question', 'user');

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).not.toContain('chat-feedback-buttons');
            expect(messages.innerHTML).not.toContain('fa-thumbs-up');
        });

        it('should not show feedback buttons on loading messages', () => {
            addChatMessage('Thinking...', 'assistant', true);

            const messages = document.getElementById('chat-messages');
            expect(messages.innerHTML).not.toContain('chat-feedback-buttons');
        });

        it('should have data attributes on feedback buttons', () => {
            const messageId = addChatMessage('Test response', 'assistant');

            const msgDiv = document.getElementById(messageId);
            const upBtn = msgDiv.querySelector('[data-rating="up"]');
            const downBtn = msgDiv.querySelector('[data-rating="down"]');

            expect(upBtn).not.toBeNull();
            expect(downBtn).not.toBeNull();
            expect(upBtn.dataset.msgId).toBe(messageId);
            expect(downBtn.dataset.msgId).toBe(messageId);
        });
    });

    describe('handleChatFeedback', () => {
        beforeEach(() => {
            localStorage.clear();
            global.confirm = jest.fn();
        });

        it('should show consent popup on first click', async () => {
            global.confirm.mockReturnValue(false);

            const messageId = addChatMessage('Test response', 'assistant');
            await handleChatFeedback(messageId, 'up');

            expect(global.confirm).toHaveBeenCalledTimes(1);
            expect(global.confirm.mock.calls[0][0]).toContain('anonymized');
        });

        it('should not send data if user declines consent', async () => {
            global.confirm.mockReturnValue(false);

            const messageId = addChatMessage('Test response', 'assistant');
            await handleChatFeedback(messageId, 'up');

            expect(global.fetch).not.toHaveBeenCalledWith(
                '/api/donate-chat',
                expect.anything()
            );
        });

        it('should send transcript to backend if user accepts consent', async () => {
            global.confirm.mockReturnValue(true);
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ success: true, message: 'Thank you!' })
            });

            addChatMessage('User question', 'user');
            const assistantId = addChatMessage('AI response', 'assistant');

            await handleChatFeedback(assistantId, 'up');

            // Should have called fetch with /api/donate-chat
            const chatCalls = global.fetch.mock.calls.filter(c => c[0] === '/api/donate-chat');
            expect(chatCalls.length).toBe(1);

            const body = JSON.parse(chatCalls[0][1].body);
            expect(body.rating).toBe('up');
            expect(body.transcript).toBeInstanceOf(Array);
            expect(body.transcript.length).toBeGreaterThan(0);
        });

        it('should replace buttons with thank you message after success', async () => {
            global.confirm.mockReturnValue(true);
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ success: true, message: 'Thank you!' })
            });

            const messageId = addChatMessage('Test response', 'assistant');
            await handleChatFeedback(messageId, 'up');

            const msgDiv = document.getElementById(messageId);
            const feedbackDiv = msgDiv.querySelector('.chat-feedback-buttons');
            expect(feedbackDiv.innerHTML).toContain('Thanks for your feedback');
        });

        it('should store consent in localStorage', async () => {
            global.confirm.mockReturnValue(true);
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ success: true })
            });

            const messageId = addChatMessage('Test', 'assistant');
            await handleChatFeedback(messageId, 'up');

            expect(localStorage.getItem('chatDonationConsent')).toBe('true');
        });

        it('should not show consent popup on subsequent clicks after acceptance', async () => {
            // Simulate previously accepted consent
            localStorage.setItem('chatDonationConsent', 'true');

            // Need to re-import to pick up localStorage change
            // Since module already loaded, we set consent directly via another feedback call
            global.confirm.mockReturnValue(true);
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ success: true })
            });

            // First call to establish consent
            const messageId1 = addChatMessage('Test 1', 'assistant');
            await handleChatFeedback(messageId1, 'up');

            // Reset confirm mock
            global.confirm.mockClear();
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ success: true })
            });

            // Second call should not show confirm
            const messageId2 = addChatMessage('Test 2', 'assistant');
            await handleChatFeedback(messageId2, 'down');

            expect(global.confirm).not.toHaveBeenCalled();
        });
    });

    describe('MCP tools hint bubble', () => {
        it('buildMcpToolsHintHtml should return HTML with mcp-tools-hint id', () => {
            const html = buildMcpToolsHintHtml();
            expect(html).toContain('id="mcp-tools-hint"');
            expect(html).toContain('specialized tools');
        });

        it('should contain example queries for each MCP tool', () => {
            const html = buildMcpToolsHintHtml();
            // Conference topics
            expect(html).toContain('main topics');
            // Topic evolution
            expect(html).toContain('evolved over the years');
            // Search papers
            expect(html).toContain('reinforcement learning');
            // Topic relevance
            expect(html).toContain('uncertainty quantification');
            // Paper details
            expect(html).toContain('Who are the authors');
        });

        it('removeMcpToolsHint should add hide class when hint exists', () => {
            document.getElementById('chat-messages').innerHTML = buildMcpToolsHintHtml();

            removeMcpToolsHint();

            const hint = document.getElementById('mcp-tools-hint');
            expect(hint.classList.contains('mcp-tools-hint-hide')).toBe(true);
        });

        it('removeMcpToolsHint should do nothing when hint does not exist', () => {
            // Should not throw
            removeMcpToolsHint();
        });

        it('sendChatMessage should remove hint on first user message', async () => {
            document.getElementById('chat-messages').innerHTML = buildMcpToolsHintHtml();

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: { response: 'reply', papers: [], metadata: {} }
                })
            });

            await sendChatMessage();

            const hint = document.getElementById('mcp-tools-hint');
            // Hint should have the hide class (animation pending removal)
            expect(hint.classList.contains('mcp-tools-hint-hide')).toBe(true);
        });

        it('resetChat should re-show the hint', async () => {
            // First remove hint via a chat message
            document.getElementById('chat-messages').innerHTML = buildMcpToolsHintHtml();
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    response: { response: 'reply', papers: [], metadata: {} }
                })
            });
            await sendChatMessage();

            // Now reset
            global.fetch.mockResolvedValueOnce({ ok: true });
            await resetChat();

            const hint = document.getElementById('mcp-tools-hint');
            expect(hint).not.toBeNull();
            // Should NOT have hide class
            expect(hint.classList.contains('mcp-tools-hint-hide')).toBe(false);
        });

        it('initMcpToolsHint should insert hint into chat-messages', () => {
            initMcpToolsHint();

            const hint = document.getElementById('mcp-tools-hint');
            expect(hint).not.toBeNull();
            expect(hint.closest('#chat-messages')).not.toBeNull();
        });

        it('initMcpToolsHint should not duplicate hint if already present', () => {
            initMcpToolsHint();
            initMcpToolsHint();

            const hints = document.querySelectorAll('#mcp-tools-hint');
            expect(hints.length).toBe(1);
        });
    });

    describe('feedback highlight animation', () => {
        it('should add feedback-highlight class on first assistant message', () => {
            addChatMessage('First response', 'assistant');

            const feedbackDiv = document.querySelector('.chat-feedback-buttons');
            expect(feedbackDiv.classList.contains('feedback-highlight')).toBe(true);
        });

        it('should not add feedback-highlight class on second assistant message', () => {
            addChatMessage('First response', 'assistant');
            addChatMessage('Second response', 'assistant');

            const feedbackDivs = document.querySelectorAll('.chat-feedback-buttons');
            expect(feedbackDivs[0].classList.contains('feedback-highlight')).toBe(true);
            expect(feedbackDivs[1].classList.contains('feedback-highlight')).toBe(false);
        });

        it('should not add feedback-highlight on user messages', () => {
            addChatMessage('User message', 'user');

            const feedbackDiv = document.querySelector('.feedback-highlight');
            expect(feedbackDiv).toBeNull();
        });

        it('should add feedback-bounce class to buttons on first assistant message', () => {
            addChatMessage('First response', 'assistant');

            const buttons = document.querySelectorAll('.chat-feedback-btn');
            buttons.forEach(btn => {
                expect(btn.classList.contains('feedback-bounce')).toBe(true);
            });
        });

        it('should not add feedback-bounce class to buttons on second assistant message', () => {
            addChatMessage('First response', 'assistant');
            addChatMessage('Second response', 'assistant');

            const allFeedbackDivs = document.querySelectorAll('.chat-feedback-buttons');
            const secondMsgButtons = allFeedbackDivs[1].querySelectorAll('.chat-feedback-btn');
            secondMsgButtons.forEach(btn => {
                expect(btn.classList.contains('feedback-bounce')).toBe(false);
            });
        });

        it('should set staggered animation delay on buttons', () => {
            addChatMessage('First response', 'assistant');

            const buttons = document.querySelectorAll('.chat-feedback-btn');
            expect(buttons[0].style.animationDelay).toBe('0.5s');
            expect(buttons[1].style.animationDelay).toBe('0.65s');
        });

        it('should not add feedback-bounce on user messages', () => {
            addChatMessage('User message', 'user');

            const bounceBtn = document.querySelector('.feedback-bounce');
            expect(bounceBtn).toBeNull();
        });
    });
});
