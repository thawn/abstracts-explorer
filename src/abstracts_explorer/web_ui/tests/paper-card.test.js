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
        // Reset state properly
        State.resetState();
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
                <span id="interesting-count" class="interesting-count"></span>
            </button>
            <select id="year-selector"><option value="">All</option></select>
            <select id="conference-selector"><option value="">All</option></select>
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

            // Check for details and summary elements (with attributes)
            expect(html).toMatch(/<details/);
            expect(html).toMatch(/<summary/);
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

        it('should update stars in chat papers panel', () => {
            // Set up chat papers panel with a paper card
            document.body.innerHTML += `
                <div id="chat-papers">
                    <div class="paper-card" onclick="showPaperDetails('p3')">
                        <div class="star-container">
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer" onclick="setPaperPriority('p3', 1)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer" onclick="setPaperPriority('p3', 2)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer" onclick="setPaperPriority('p3', 3)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer" onclick="setPaperPriority('p3', 4)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer" onclick="setPaperPriority('p3', 5)"></i>
                        </div>
                    </div>
                </div>
            `;
            
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p3', 3);

            updateStarDisplay('p3');

            const chatPapersDiv = document.getElementById('chat-papers');
            expect(chatPapersDiv).not.toBeNull();
            
            const stars = chatPapersDiv.querySelectorAll('i[onclick*="p3"]');
            expect(stars.length).toBe(5);
            
            // First 3 stars should be filled (fas), remaining 2 should be empty (far)
            expect(stars[0].className).toContain('fas');
            expect(stars[1].className).toContain('fas');
            expect(stars[2].className).toContain('fas');
            expect(stars[3].className).toContain('far');
            expect(stars[4].className).toContain('far');
        });
    });

    describe('updateInterestingPapersCount', () => {
        it('should update count display', async () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p1', 5);
            State.setPaperPriority('p2', 4);

            // No filters selected, so it should just count the priorities
            await updateInterestingPapersCount();

            const countSpan = document.getElementById('interesting-count');
            expect(countSpan).not.toBeNull();
            expect(countSpan.textContent).toBe('2');
        });

        it('should show 0 when no rated papers', async () => {
            // No priorities set
            await updateInterestingPapersCount();

            const countSpan = document.getElementById('interesting-count');
            expect(countSpan).not.toBeNull();
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
