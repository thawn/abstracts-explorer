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
    loadInterestingPapers,
    loadInterestingPapersFromJSON,
    handleJSONFileLoad,
    saveInterestingPapersAsMarkdown,
    donateInterestingPapersData,
    switchInterestingSession,
    switchInterestingConferenceFromMenu,
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
            <div id="interesting-conference-menu" class="hidden">
                <select id="interesting-conference-select">
                    <option value="">Select conference/year…</option>
                </select>
            </div>
            <select id="year-selector">
                <option value="2025">2025</option>
                <option value="2024">2024</option>
                <option value="2023">2023</option>
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


    describe('other conference menu', () => {
        const setPriorities = (ids) => {
            const priorities = {};
            ids.forEach((id) => {
                priorities[id] = { priority: 3 };
            });
            localStorage.setItem('paperPriorities', JSON.stringify(priorities));
            loadPriorities();
        };

        const getMenu = () => document.getElementById('interesting-conference-menu');
        const getSelect = () => document.getElementById('interesting-conference-select');

        test('should hide the menu when only the current combo has papers', () => {
            setPriorities(['paper1', 'paper2']);
            document.getElementById('conference-selector').value = 'NeurIPS';
            document.getElementById('year-selector').value = '2025';

            displayInterestingPapers([
                { uid: 'paper1', title: 'Paper 1', session: 'Session A', conference: 'NeurIPS', year: 2025, poster_position: 'A1' },
                { uid: 'paper2', title: 'Paper 2', session: 'Session A', conference: 'NeurIPS', year: 2025, poster_position: 'A2' }
            ]);

            expect(getMenu().classList.contains('hidden')).toBe(true);
            expect(getSelect().options).toHaveLength(1);
        });

        test('should list other conference/year combos with counts', () => {
            setPriorities(['paper1', 'paper2', 'paper3', 'paper4', 'paper5']);
            document.getElementById('conference-selector').value = 'NeurIPS';
            document.getElementById('year-selector').value = '2025';

            displayInterestingPapers([
                { uid: 'paper1', title: 'Paper 1', session: 'Session A', conference: 'NeurIPS', year: 2025, poster_position: 'A1' },
                { uid: 'paper2', title: 'Paper 2', session: 'Session A', conference: 'ICLR', year: 2025, poster_position: 'A2' },
                { uid: 'paper3', title: 'Paper 3', session: 'Session A', conference: 'ICLR', year: 2025, poster_position: 'A3' },
                { uid: 'paper4', title: 'Paper 4', session: 'Session A', conference: 'NeurIPS', year: 2024, poster_position: 'A4' },
                { uid: 'paper5', title: 'Paper 5', session: 'Session A', year: 2024, poster_position: 'A5' }
            ]);

            expect(getMenu().classList.contains('hidden')).toBe(false);
            const labels = Array.from(getSelect().options).map(option => option.textContent);
            expect(labels).toContain('ICLR 2025 (2)');
            expect(labels).toContain('NeurIPS 2024 (1)');
            expect(labels).not.toContain('NeurIPS 2025 (1)');
        });

        test('should switch header selectors and refresh when a combo is chosen', () => {
            window.conferenceYearsMap = { NeurIPS: [2025], ICLR: [2025, 2024] };
            window.handleConferenceChange = jest.fn();

            switchInterestingConferenceFromMenu(`${encodeURIComponent('ICLR')}|2024`);

            expect(document.getElementById('conference-selector').value).toBe('ICLR');
            expect(document.getElementById('year-selector').value).toBe('2024');
            expect(window.handleConferenceChange).toHaveBeenCalled();

            delete window.conferenceYearsMap;
            delete window.handleConferenceChange;
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

    describe('proxy error handling', () => {
        const html503 = () => ({
            ok: false,
            status: 503,
            json: async () => {
                throw new SyntaxError('Unexpected token <');
            }
        });

        beforeEach(() => {
            localStorage.setItem(
                'paperPriorities',
                JSON.stringify({ p1: { priority: 5, searchTerm: 'transformers' } })
            );
            loadPriorities();
            global.alert = jest.fn();
        });

        test('should show a friendly message when loading papers gets an HTML 503', async () => {
            global.fetch = jest.fn().mockResolvedValueOnce(html503());

            await loadInterestingPapers();

            const list = document.getElementById('interesting-papers-list');
            expect(list.innerHTML).toContain('The server is temporarily unavailable');
            expect(list.innerHTML).not.toContain('Unexpected token');
        });

        test('should show a friendly message when export gets an HTML 503', async () => {
            global.fetch = jest.fn().mockResolvedValueOnce(html503());
            global.confirm = jest.fn(() => false);
            const button = document.createElement('button');
            button.innerHTML = 'Download';

            await saveInterestingPapersAsMarkdown({ target: button });

            expect(global.alert).toHaveBeenCalledWith(
                expect.stringContaining('The server is temporarily unavailable')
            );
            expect(global.alert).not.toHaveBeenCalledWith(expect.stringContaining('Unexpected token'));
        });

        test('should show a friendly message when donation gets an HTML 503', async () => {
            global.fetch = jest.fn().mockResolvedValueOnce(html503());
            global.confirm = jest.fn(() => true);

            await donateInterestingPapersData();

            expect(global.alert).toHaveBeenCalledWith(
                expect.stringContaining('The server is temporarily unavailable')
            );
            expect(global.alert).not.toHaveBeenCalledWith(expect.stringContaining('Unexpected token'));
        });
    });
});


describe('Interesting Papers JSON import rating format', () => {
    beforeEach(() => {
        global.alert = jest.fn();
        global.confirm = jest.fn(() => true);
        const store = {};
        global.localStorage = {
            getItem: jest.fn((key) => store[key] || null),
            setItem: jest.fn((key, value) => { store[key] = value.toString(); }),
            removeItem: jest.fn((key) => { delete store[key]; }),
            clear: jest.fn(() => {
                for (const key in store) {
                    delete store[key];
                }
            })
        };
        loadPriorities();
        global.FileReader = class {
            readAsText(file) {
                this.result = file._content;
                this.onload({ target: { result: this.result } });
            }
        };
    });

    test('rejects legacy scalar ratings', () => {
        handleJSONFileLoad({
            target: {
                files: [{
                    name: 'ratings.json',
                    _content: JSON.stringify({
                        paperPriorities: { paper1: 5 }
                    })
                }],
                value: 'ratings.json'
            }
        });

        expect(global.alert).toHaveBeenCalledWith(
            expect.stringContaining('Invalid rating format')
        );
    });

    test('accepts dict ratings', () => {
        handleJSONFileLoad({
            target: {
                files: [{
                    name: 'ratings.json',
                    _content: JSON.stringify({
                        paperPriorities: {
                            paper1: { priority: 5, searchTerm: 'llms' }
                        }
                    })
                }],
                value: 'ratings.json'
            }
        });

        expect(global.alert).toHaveBeenCalledWith(
            expect.not.stringContaining('Invalid rating format')
        );
    });
});
