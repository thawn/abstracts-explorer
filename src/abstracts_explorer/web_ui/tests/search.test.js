/**
 * Tests for search module
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.fetch = jest.fn();
global.marked = { parse: jest.fn((text) => text), use: jest.fn() };
global.markedKatex = jest.fn(() => ({}));

import { searchPapers, displaySearchResults, openAdvancedSearch, closeAdvancedSearch, applyAdvancedSearch } from '../static/modules/search.js';
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
            <div id="advanced-search-modal" class="hidden">
                <input id="adv-topic" />
                <input id="adv-authors" />
                <input id="adv-title" />
                <input id="adv-keywords" />
                <input id="adv-abstract" />
                <input id="adv-award" />
            </div>
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

        it('should include distance_threshold in request when input is present', async () => {
            document.body.innerHTML += `<input id="distance-threshold-input" value="0.8" type="number" />`;
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ papers: [], count: 0, use_embeddings: true })
            });

            await searchPapers();

            const callArgs = JSON.parse(global.fetch.mock.calls[0][1].body);
            expect(callArgs.distance_threshold).toBe(0.8);
        });

        it('should not include distance_threshold when input is absent', async () => {
            // Ensure the input is not present (default DOM has none)
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ papers: [], count: 0, use_embeddings: true })
            });

            await searchPapers();

            const callArgs = JSON.parse(global.fetch.mock.calls[0][1].body);
            expect(callArgs.distance_threshold).toBeUndefined();
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
                use_embeddings: true,
                total_similar: 42
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('best matches');
            expect(results.innerHTML).toContain('<strong>1</strong>');
            expect(results.innerHTML).toContain('<strong>42</strong>');
            expect(results.innerHTML).toContain('similar papers');
            expect(results.innerHTML).toContain('Test Paper');
            expect(results.innerHTML).toContain('LLM-Powered');
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
            expect(results.innerHTML).toContain('Found <strong>2</strong> papers');
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

        it('should display related topics when provided', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Test Paper',
                        authors: ['Author 1'],
                        abstract: 'Test abstract',
                        year: 2025,
                        conference: 'NeurIPS'
                    }
                ],
                count: 1,
                use_embeddings: true,
                related_topics: ['deep learning', 'neural networks', 'optimization']
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('Related Topics');
            expect(results.innerHTML).toContain('deep learning');
            expect(results.innerHTML).toContain('neural networks');
            expect(results.innerHTML).toContain('optimization');
        });

        it('should not display related topics section when related_topics is empty', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Test Paper',
                        authors: ['Author 1'],
                        abstract: 'Test abstract',
                        year: 2025
                    }
                ],
                count: 1,
                related_topics: []
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).not.toContain('Related Topics');
        });

        it('should not display related topics section when related_topics is absent', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Test Paper',
                        authors: ['Author 1'],
                        abstract: 'Test abstract',
                        year: 2025
                    }
                ],
                count: 1
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).not.toContain('Related Topics');
        });

        it('should render related topic keywords as clickable buttons', () => {
            const data = {
                papers: [
                    {
                        uid: 'paper1',
                        title: 'Test Paper',
                        authors: [],
                        abstract: 'Test abstract'
                    }
                ],
                count: 1,
                related_topics: ['attention mechanism']
            };

            displaySearchResults(data);

            const results = document.getElementById('search-results');
            expect(results.innerHTML).toContain('attention mechanism');
            // Should be rendered as a button element with a data-topic attribute
            const buttons = results.querySelectorAll('button');
            const topicButton = Array.from(buttons).find(btn => btn.textContent.trim() === 'attention mechanism');
            expect(topicButton).toBeTruthy();
            expect(topicButton.dataset.topic).toBe('attention mechanism');
        });
    });

    describe('Advanced Search Modal', () => {
        it('should open the modal and remove hidden class', () => {
            openAdvancedSearch();
            const modal = document.getElementById('advanced-search-modal');
            expect(modal.classList.contains('hidden')).toBe(false);
        });

        it('should close the modal and add hidden class', () => {
            openAdvancedSearch();
            closeAdvancedSearch();
            const modal = document.getElementById('advanced-search-modal');
            expect(modal.classList.contains('hidden')).toBe(true);
        });

        it('should parse existing search input into fields', () => {
            document.getElementById('search-input').value = 'authors:"John Smith" deep learning';
            openAdvancedSearch();

            expect(document.getElementById('adv-authors').value).toBe('John Smith');
            expect(document.getElementById('adv-topic').value).toBe('deep learning');
        });

        it('should parse author alias into the authors field', () => {
            document.getElementById('search-input').value = 'author:"Vaswani" attention';
            openAdvancedSearch();

            expect(document.getElementById('adv-authors').value).toBe('Vaswani');
            expect(document.getElementById('adv-topic').value).toBe('attention');
        });

        it('should build search query from advanced fields', () => {
            openAdvancedSearch();
            document.getElementById('adv-topic').value = 'uncertainty';
            document.getElementById('adv-authors').value = 'John Smith';
            document.getElementById('adv-title').value = '';
            document.getElementById('adv-keywords').value = '';
            document.getElementById('adv-abstract').value = '';
            document.getElementById('adv-award').value = '';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ papers: [], count: 0, use_embeddings: true })
            });

            applyAdvancedSearch();

            const searchInput = document.getElementById('search-input');
            expect(searchInput.value).toBe('authors:"John Smith" uncertainty');
        });

        it('should build field-only query without topic', () => {
            openAdvancedSearch();
            document.getElementById('adv-topic').value = '';
            document.getElementById('adv-authors').value = 'Doe';
            document.getElementById('adv-title').value = 'Transformer';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ papers: [], count: 0 })
            });

            applyAdvancedSearch();

            const searchInput = document.getElementById('search-input');
            expect(searchInput.value).toBe('authors:"Doe" title:"Transformer"');
        });

        it('should close the modal after applying', () => {
            openAdvancedSearch();
            document.getElementById('adv-topic').value = 'test';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ papers: [], count: 0 })
            });

            applyAdvancedSearch();
            const modal = document.getElementById('advanced-search-modal');
            expect(modal.classList.contains('hidden')).toBe(true);
        });

        it('should parse multiple field filters from existing query', () => {
            document.getElementById('search-input').value = 'title:"Transformer" authors:"Vaswani" attention';
            openAdvancedSearch();

            expect(document.getElementById('adv-title').value).toBe('Transformer');
            expect(document.getElementById('adv-authors').value).toBe('Vaswani');
            expect(document.getElementById('adv-topic').value).toBe('attention');
        });
    });
});
