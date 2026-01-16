/**
 * Tests for chat module
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.fetch = jest.fn();
global.marked = { parse: jest.fn((text) => text), use: jest.fn() };

import { sendChatMessage, displayChatPapers, addChatMessage, resetChat } from '../static/modules/chat.js';
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
});
