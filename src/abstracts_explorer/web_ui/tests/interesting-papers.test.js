/**
 * Tests for interesting-papers.js - Interesting Papers Module
 * 
 * This module handles display, management, and export of rated/interesting papers.
 * Testing approach: Focus on DOM manipulation, sorting logic, and export functionality
 * without deeply mocking the state module.
 */

import {
    jest,
    expect,
    describe,
    test,
    beforeEach
} from '@jest/globals';

import {
    displayInterestingPapers,
    generateInterestingPapersMarkdown,
    loadInterestingPapersFromJSON,
    switchInterestingSession,
    changeInterestingPapersSortOrder
} from '../static/modules/interesting-papers.js';

import { loadPriorities } from '../static/modules/state.js';

describe('Interesting Papers Module', () => {
    beforeEach(() => {
        // Setup DOM
        document.body.innerHTML = `
            <div id="interesting-papers-list"></div>
            <div id="interesting-session-tabs"></div>
            <div id="interesting-session-tabs-nav"></div>
            <div id="interesting-count">0</div>
            <select id="year-selector">
                <option value="">All Years</option>
                <option value="2023">2023</option>
                <option value="2024">2024</option>
                <option value="2025">2025</option>
            </select>
            <select id="conference-selector">
                <option value="">All Conferences</option>
                <option value="NeurIPS">NeurIPS</option>
                <option value="ICLR">ICLR</option>
                <option value="ICML">ICML</option>
            </select>
            <input type="file" id="json-file-input" />
        `;

        // Mock localStorage with proper store reference
        const store = {};
        const localStorageMock = {
            getItem: jest.fn((key) => store[key] || null),
            setItem: jest.fn((key, value) => { store[key] = value.toString(); }),
            removeItem: jest.fn((key) => { delete store[key]; }),
            clear: jest.fn(() => { 
                for (const key in store) {
                    delete store[key];
                }
            })
        };
        global.localStorage = localStorageMock;

        // Reset state
        jest.clearAllMocks();
        localStorageMock.clear();
    });

    describe('displayInterestingPapers', () => {
        test('should display papers grouped by session', () => {
            // Setup priorities in localStorage
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3, searchTerm: 'transformers' },
                'paper2': { priority: 2, searchTerm: 'vision' }
            }));
            localStorage.setItem('interestingPapersSortOrder', 'search-rating-poster');
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 1',
                    abstract: 'Abstract 1',
                    session: 'Session A',
                    year: 2025,
                    poster_position: 'A1'
                },
                {
                    uid: 'paper2',
                    title: 'Paper 2',
                    abstract: 'Abstract 2',
                    session: 'Session B',
                    year: 2025,
                    poster_position: 'B1'
                }
            ];

            displayInterestingPapers(papers);

            const tabsNav = document.getElementById('interesting-session-tabs-nav');
            expect(tabsNav.innerHTML).toContain('Session A');
            expect(tabsNav.innerHTML).toContain('Session B');
        });

        test('should filter papers by year', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3 },
                'paper2': { priority: 2 }
            }));
            loadPriorities(); // Load from localStorage into state

            const yearSelect = document.getElementById('year-selector');
            yearSelect.value = '2025';

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 2025',
                    session: 'Session A',
                    year: 2025,
                    poster_position: 'A1'
                },
                {
                    uid: 'paper2',
                    title: 'Paper 2024',
                    session: 'Session A',
                    year: 2024,
                    poster_position: 'A2'
                }
            ];

            displayInterestingPapers(papers);

            const countElement = document.getElementById('interesting-count');
            expect(countElement.textContent).toBe('1');
        });

        test('should filter papers by conference', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3 },
                'paper2': { priority: 2 }
            }));
            loadPriorities(); // Load from localStorage into state

            const conferenceSelect = document.getElementById('conference-selector');
            conferenceSelect.value = 'NeurIPS';

            const papers = [
                {
                    uid: 'paper1',
                    title: 'NeurIPS Paper',
                    session: 'Session A',
                    conference: 'NeurIPS',
                    year: 2025,
                    poster_position: 'A1'
                },
                {
                    uid: 'paper2',
                    title: 'ICLR Paper',
                    session: 'Session A',
                    conference: 'ICLR',
                    year: 2025,
                    poster_position: 'A2'
                }
            ];

            displayInterestingPapers(papers);

            const countElement = document.getElementById('interesting-count');
            expect(countElement.textContent).toBe('1');
        });

        test('should show empty state when no papers match filters', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3 }
            }));
            loadPriorities(); // Load from localStorage into state

            const yearSelect = document.getElementById('year-selector');
            yearSelect.value = '2023';

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 2025',
                    session: 'Session A',
                    year: 2025,
                    poster_position: 'A1'
                }
            ];

            displayInterestingPapers(papers);

            const listDiv = document.getElementById('interesting-papers-list');
            expect(listDiv.innerHTML).toContain('No papers match the selected filters');
            expect(listDiv.innerHTML).toContain('fa-filter');
        });

        test('should group papers by search term when sort order is search-rating-poster', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3, searchTerm: 'transformers' }
            }));
            localStorage.setItem('currentInterestingSession', 'Session A');
            localStorage.setItem('interestingPapersSortOrder', 'search-rating-poster');
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 1',
                    session: 'Session A',
                    year: 2025,
                    poster_position: 'A1'
                }
            ];

            displayInterestingPapers(papers);

            const listDiv = document.getElementById('interesting-papers-list');
            expect(listDiv.innerHTML).toContain('transformers');
        });

        test('should group papers by rating when sort order is rating-poster-search', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3, searchTerm: 'transformers' }
            }));
            localStorage.setItem('currentInterestingSession', 'Session A');
            localStorage.setItem('interestingPapersSortOrder', 'rating-poster-search');
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 1',
                    session: 'Session A',
                    year: 2025,
                    poster_position: 'A1'
                }
            ];

            displayInterestingPapers(papers);

            const listDiv = document.getElementById('interesting-papers-list');
            expect(listDiv.innerHTML).toContain('3 stars');
        });

        test('should update count element with number of papers', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3 },
                'paper2': { priority: 2 },
                'paper3': { priority: 1 }
            }));
            loadPriorities(); // Load from localStorage into state

            const papers = [
                { uid: 'paper1', title: 'Paper 1', session: 'Session A', year: 2025, poster_position: 'A1' },
                { uid: 'paper2', title: 'Paper 2', session: 'Session A', year: 2025, poster_position: 'A2' },
                { uid: 'paper3', title: 'Paper 3', session: 'Session A', year: 2025, poster_position: 'A3' }
            ];

            displayInterestingPapers(papers);

            const countElement = document.getElementById('interesting-count');
            expect(countElement.textContent).toBe('3');
        });
    });

    describe('generateInterestingPapersMarkdown', () => {
        test('should generate markdown for papers', () => {
            // Setup priorities in localStorage for the function to read
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3, searchTerm: 'transformers' }
            }));
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Test Paper',
                    authors: ['Author 1', 'Author 2'],
                    abstract: 'Test abstract',
                    year: 2025,
                    conference: 'NeurIPS',
                    session: 'Session A',
                    poster_position: 'A1',
                    pdf_url: 'http://example.com/paper1.pdf'
                }
            ];

            const markdown = generateInterestingPapersMarkdown(papers);

            expect(markdown).toContain('# Interesting Papers');
            expect(markdown).toContain('Test Paper');
            expect(markdown).toContain('Author 1');
            expect(markdown).toContain('Session A');
            expect(markdown).toContain('⭐⭐⭐'); // 3 stars
        });

        test('should handle papers without optional fields', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 1 }
            }));
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Test Paper',
                    authors: [],
                    abstract: 'Test abstract'
                }
            ];

            const markdown = generateInterestingPapersMarkdown(papers);

            expect(markdown).toContain('Test Paper');
            expect(markdown).toContain('⭐'); // 1 star
        });

        test('should group papers by session', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 3 },
                'paper2': { priority: 2 }
            }));
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Paper 1',
                    authors: [],
                    abstract: 'Abstract 1',
                    session: 'Session A'
                },
                {
                    uid: 'paper2',
                    title: 'Paper 2',
                    authors: [],
                    abstract: 'Abstract 2',
                    session: 'Session A'
                }
            ];

            const markdown = generateInterestingPapersMarkdown(papers);

            expect(markdown).toContain('## Session A');
            expect(markdown).toContain('Paper 1');
            expect(markdown).toContain('Paper 2');
        });

        test('should include metadata for papers', () => {
            localStorage.setItem('paperPriorities', JSON.stringify({
                'paper1': { priority: 2 }
            }));
            loadPriorities(); // Load from localStorage into state

            const papers = [
                {
                    uid: 'paper1',
                    title: 'Test Paper',
                    authors: ['Author 1'],
                    abstract: 'Test abstract',
                    year: 2025,
                    conference: 'NeurIPS',
                    session: 'Poster Session',
                    poster_position: 'A1',
                    pdf_url: 'http://example.com/paper.pdf'
                }
            ];

            const markdown = generateInterestingPapersMarkdown(papers);

            // Check what the function actually includes
            expect(markdown).toContain('Test Paper');
            expect(markdown).toContain('Author 1');
            expect(markdown).toContain('**Poster:** A1');
            expect(markdown).toContain('Test abstract');
        });
    });

    describe('loadInterestingPapersFromJSON', () => {
        test('should trigger file input click', () => {
            const fileInput = document.getElementById('json-file-input');
            fileInput.click = jest.fn();

            loadInterestingPapersFromJSON();

            expect(fileInput.click).toHaveBeenCalled();
        });
    });

    describe('switchInterestingSession', () => {
        test('should switch session without errors', () => {
            // Mock the loadInterestingPapers function to prevent actual API calls
            global.fetch = jest.fn().mockResolvedValue({
                json: async () => ({ papers: [] })
            });

            // This should not throw an error
            expect(() => {
                switchInterestingSession('Session B');
            }).not.toThrow();
        });
    });

    describe('changeInterestingPapersSortOrder', () => {
        test('should update sort order and persist to localStorage', () => {
            // Mock the loadInterestingPapers function to prevent actual API calls
            global.fetch = jest.fn().mockResolvedValue({
                json: async () => ({ papers: [] })
            });

            // Call the function
            changeInterestingPapersSortOrder('rating-poster-search');

            // Verify the value was set in localStorage
            const storedOrder = localStorage.getItem('interestingPapersSortOrder');
            expect(storedOrder).toBe('rating-poster-search');
        });
    });
});
