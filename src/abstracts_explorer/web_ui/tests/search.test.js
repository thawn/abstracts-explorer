/**
 * Tests for search module
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.fetch = jest.fn();
global.marked = { parse: jest.fn((text) => text), use: jest.fn() };
global.markedKatex = jest.fn(() => ({}));

import { searchPapers, displaySearchResults } from '../static/modules/search.js';
import * as State from '../static/modules/state.js';

describe('Search Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorage.clear();
        document.body.innerHTML = `
            <input id="search-input" value="machine learning" />
            <select id="limit-select"><option value="10" selected>10</option></select>
            <select id="session-filter" multiple>
                <option value="Session 1" selected>Session 1</option>
                <option value="Session 2" selected>Session 2</option>
            </select>
            <select id="year-selector"><option value="2025" selected>2025</option></select>
            <select id="conference-selector"><option value="NeurIPS" selected>NeurIPS</option></select>
            <div id="search-results"></div>
        `;
    });

    describe('searchPapers', () => {
        it('should show error when query is empty', async () => {
            document.getElementById('search-input').value = '';
            
            await searchPapers();
            
            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Please enter a search query');
        });

        it('should send search request with correct parameters', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    papers: [],
                    count: 0,
                    use_embeddings: true
                })
            });

            await searchPapers();

            expect(global.fetch).toHaveBeenCalledWith(
                '/api/search',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: expect.stringContaining('machine learning')
                })
            );
        });

        it('should include filters in request', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    papers: [],
                    count: 0,
                    use_embeddings: true
                })
            });

            await searchPapers();

            const callArgs = JSON.parse(global.fetch.mock.calls[0][1].body);
            expect(callArgs.query).toBe('machine learning');
            expect(callArgs.limit).toBe(10);
            expect(callArgs.years).toEqual([2025]);
            expect(callArgs.conferences).toEqual(['NeurIPS']);
        });

        it('should handle API errors', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    error: 'Search failed'
                })
            });

            await searchPapers();

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Search failed');
        });

        it('should handle network errors', async () => {
            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            await searchPapers();

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('An error occurred');
        });

        it('should set current search term', async () => {
            State.setCurrentSearchTerm('');
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    papers: [],
                    count: 0
                })
            });

            await searchPapers();

            expect(State.getCurrentSearchTerm()).toBe('machine learning');
        });
    });

    describe('displaySearchResults', () => {
        it('should show empty state when no papers found', () => {
            displaySearchResults({ papers: [], count: 0 });

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('No papers found');
        });

        it('should display papers with count', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Test Paper',
                        authors: ['Author 1', 'Author 2'],
                        abstract: 'Test abstract',
                        year: 2025,
                        conference: 'NeurIPS'
                    }
                ],
                count: 1,
                use_embeddings: true
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Found <strong>1</strong> papers');
            expect(results.innerHTML).toContain('Test Paper');
            expect(results.innerHTML).toContain('AI-Powered');
        });

        it('should display multiple papers', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Paper 1',
                        authors: ['Author 1'],
                        abstract: 'Abstract 1',
                        year: 2025
                    },
                    {
                        uid: 'paper2',
                        title: 'Paper 2',
                        authors: ['Author 2'],
                        abstract: 'Abstract 2',
                        year: 2025
                    }
                ],
                count: 2
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Paper 1');
            expect(results.innerHTML).toContain('Paper 2');
        });

        it('should handle papers with missing fields', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Minimal Paper',
                        authors: []
                    }
                ],
                count: 1
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Minimal Paper');
        });
    });
});
