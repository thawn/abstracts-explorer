/**
 * Tests for chat module
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.fetch = jest.fn();
global.marked = { parse: jest.fn((text) => text), use: jest.fn() };
global.Plotly = { newPlot: jest.fn() };

import { sendChatMessage, displayChatPapers, addChatMessage, resetChat, renderChatVisualizations } from '../static/modules/chat.js';
import * as State from '../static/modules/state.js';

describe('Chat Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorage.clear();
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
                topic: 'transformers',
                conferences: ['NeurIPS'],
                conference_data: {
                    'NeurIPS': {
                        year_relative: { '2022': 2.5, '2023': 4.0, '2024': 5.0 },
                        year_counts: { '2022': 5, '2023': 10, '2024': 15 }
                    }
                }
            }];

            renderChatVisualizations(visualizations);

            expect(global.Plotly.newPlot).toHaveBeenCalledTimes(1);
            const [plotId, traces, layout] = global.Plotly.newPlot.mock.calls[0];
            expect(plotId).toMatch(/^chat-plot-/);
            expect(traces).toHaveLength(1);
            expect(traces[0].x).toEqual(['2022', '2023', '2024']);
            expect(traces[0].y).toEqual([2.5, 4.0, 5.0]);
            expect(traces[0].type).toBe('scatter');
            expect(traces[0].mode).toBe('lines+markers');
            expect(traces[0].name).toBe('NeurIPS');
            expect(layout.title.text).toContain('transformers');
            expect(layout.title.text).toContain('NeurIPS');
            expect(layout.xaxis.title.text).toContain('Year');
            expect(layout.yaxis.title.text).toContain('Percentage');
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
                    topic: 'rl',
                    conferences: ['ICML'],
                    conference_data: {
                        'ICML': { year_relative: { '2023': 3.0 }, year_counts: { '2023': 3 } }
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
                topic: 'gnn',
                conferences: ['NeurIPS'],
                conference_data: {
                    'NeurIPS': { year_relative: { '2023': 2.0 }, year_counts: { '2023': 2 } }
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
                            topic: 'attention',
                            conferences: ['NeurIPS'],
                            conference_data: {
                                'NeurIPS': { year_relative: { '2023': 5.0 }, year_counts: { '2023': 5 } }
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
});
