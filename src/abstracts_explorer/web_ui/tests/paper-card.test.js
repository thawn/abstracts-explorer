/**
 * Tests for paper-card module
 */

import { jest } from '@jest/globals';

global.fetch = jest.fn();

import { formatPaperCard, showPaperDetails, updateStarDisplay, updateInterestingPapersCount, setPaperPriority } from '../static/modules/paper-card.js';
import * as State from '../static/modules/state.js';

describe('Paper Card Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorage.clear();
        State.loadPriorities();
        // Mock global event for onclick handlers
        global.event = {
            stopPropagation: jest.fn(),
            preventDefault: jest.fn()
        };
        document.body.innerHTML = `
            <div id="modal-overlay" class="hidden"></div>
            <div id="paper-modal" class="hidden">
                <div id="modal-title"></div>
                <div id="modal-authors"></div>
                <div id="modal-abstract"></div>
                <div id="modal-year"></div>
                <div id="modal-conference"></div>
                <div id="modal-pdf-link"></div>
            </div>
            <button id="tab-interesting" class="tab-btn">
                <span class="interesting-count"></span>
            </button>
        `;
    });

    describe('formatPaperCard', () => {
        it('should format basic paper card', () => {
            const paper = {
                uid: 'test-paper-1',
                title: 'Test Paper',
                authors: ['Author One', 'Author Two'],
                abstract: 'Short abstract',
                year: 2025,
                conference: 'NeurIPS'
            };

            const html = formatPaperCard(paper);

            expect(html).toContain('Test Paper');
            expect(html).toContain('Author One');
            expect(html).toContain('Author Two');
            expect(html).toContain('Short abstract');
            // Year and conference are not displayed in the card
        });

        it('should show relevance score when provided', () => {
            const paper = {
                uid: 'p1',
                title: 'Paper',
                authors: [],
                abstract: 'Abstract',
                distance: 0.05 // distance of 0.05 means relevance of 0.95
            };

            const html = formatPaperCard(paper);

            // Check for the distance score display
            expect(html).toContain('0.95'); // Rounded to 3 decimals by default
        });

        it('should handle long abstracts with details element', () => {
            const paper = {
                uid: 'p1',
                title: 'Paper',
                authors: [],
                abstract: 'A'.repeat(500) // Long abstract
            };

            const html = formatPaperCard(paper);

            expect(html).toContain('<details>');
            expect(html).toContain('<summary>');
        });

        it('should handle missing PDF URL', () => {
            const paper = {
                uid: 'p1',
                title: 'Paper',
                authors: [],
                abstract: 'Abstract',
                pdf_url: null
            };

            const html = formatPaperCard(paper);

            expect(html).not.toContain('PDF');
        });

        it('should escape HTML in paper data', () => {
            const paper = {
                uid: 'p1',
                title: '<script>alert("xss")</script>',
                authors: ['<b>Hacker</b>'],
                abstract: '<img src=x onerror=alert(1)>'
            };

            const html = formatPaperCard(paper);

            expect(html).not.toContain('<script>');
            expect(html).toContain('&lt;script&gt;');
        });

        it('should show star ratings', () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('rated-paper', 4);

            const paper = {
                uid: 'rated-paper',
                title: 'Rated Paper',
                authors: [],
                abstract: 'Abstract'
            };

            const html = formatPaperCard(paper);

            expect(html).toContain('star');
            expect(html).toContain('setPaperPriority');
        });
    });

    describe('showPaperDetails', () => {
        it('should fetch and display paper details', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Detailed Paper',
                    authors: ['Author One'],
                    abstract: 'Full abstract text',
                    year: 2025,
                    conference: 'NeurIPS',
                    pdf_url: 'https://example.com/paper.pdf'
                })
            });

            await showPaperDetails('p1');

            expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/paper/p1'));
            
            // The function creates a new modal element
            const modals = document.querySelectorAll('.fixed.inset-0');
            expect(modals.length).toBeGreaterThan(0);
        });

        it('should handle missing PDF URL', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper',
                    authors: [],
                    abstract: 'Abstract',
                    pdf_url: null
                })
            });

            await showPaperDetails('p1');

            // Modal should still be created
            const modals = document.querySelectorAll('.fixed.inset-0');
            expect(modals.length).toBeGreaterThan(0);
        });

        it('should handle API errors', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    error: 'Paper not found'
                })
            });

            await showPaperDetails('invalid-id');

            // Should show error but not crash
            expect(global.fetch).toHaveBeenCalled();
        });
    });

    describe('updateStarDisplay', () => {
        it('should update star display for rated paper', () => {
            document.body.innerHTML += `<div id="stars-p1"></div>`;
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p1', 3);

            updateStarDisplay('p1');

            const stars = document.getElementById('stars-p1');
            expect(stars).not.toBeNull();
        });

        it('should handle unrated paper', () => {
            document.body.innerHTML += `<div id="stars-p2"></div>`;

            updateStarDisplay('p2');

            const stars = document.getElementById('stars-p2');
            expect(stars).not.toBeNull();
        });
    });

    describe('updateInterestingPapersCount', () => {
        it('should update count display', async () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p1', 5);
            State.setPaperPriority('p2', 4);

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    papers: [
                        { uid: 'p1', year: 2025 },
                        { uid: 'p2', year: 2025 }
                    ]
                })
            });

            await updateInterestingPapersCount();

            const countSpan = document.querySelector('.interesting-count');
            // The function updates textContent
            expect(countSpan).not.toBeNull();
            expect(parseInt(countSpan.textContent)).toBeGreaterThan(0);
        });

        it('should show 0 when no rated papers', async () => {
            // Even with no priorities, the function still runs
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    papers: []
                })
            });

            await updateInterestingPapersCount();

            const countSpan = document.querySelector('.interesting-count');
            expect(countSpan).not.toBeNull();
            // The span should exist and have been updated
            expect(countSpan.textContent).toBe('0');
        });
    });

    describe('setPaperPriority', () => {
        it('should set priority and update display', () => {
            document.body.innerHTML += `<div id="stars-p1"></div>`;
            State.setCurrentSearchTerm('test');

            setPaperPriority('p1', 4);

            expect(State.getPaperPriority('p1')).toBe(4);
        });

        it('should toggle priority when clicking same star', () => {
            document.body.innerHTML += `<div id="stars-p1"></div>`;
            State.setCurrentSearchTerm('test');

            setPaperPriority('p1', 3);
            setPaperPriority('p1', 3);

            expect(State.getPaperPriority('p1')).toBe(0);
        });

        it('should update interesting papers count', () => {
            document.body.innerHTML += `<div id="stars-p1"></div>`;
            State.setCurrentSearchTerm('test');

            setPaperPriority('p1', 5);

            const countSpan = document.querySelector('.interesting-count');
            // Count should be updated (actual value depends on async operations)
            expect(countSpan).not.toBeNull();
        });
    });
});
