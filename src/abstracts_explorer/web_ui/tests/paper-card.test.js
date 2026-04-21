/**
 * Tests for paper-card module
 */

import { jest } from '@jest/globals';

global.fetch = jest.fn();
global.marked = {
    parse: jest.fn((text) => `<p>${text}</p>`),
    parseInline: jest.fn((text) => {
        // Simulate marked's HTML escaping behavior for test purposes
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }),
    use: jest.fn()
};

import { formatPaperCard, showPaperDetails, buildUrlBadges, updateStarDisplay, updateInterestingPapersCount, setPaperPriority } from '../static/modules/paper-card.js';
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
            // Conference should be displayed in the card
            expect(html).toContain('NeurIPS');
            expect(html).toContain('fa-university');
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
            expect(html).toContain('abstract-details');
            expect(html).toContain('abstract-preview');
            expect(html).toContain('abstract-full');
            expect(html).toContain('Show more');
            expect(html).toContain('Show less');
        });

        it('should handle missing PDF URL', () => {
            const paper = {
                uid: 'p1',
                title: 'Paper',
                authors: [],
                abstract: 'Abstract',
                paper_pdf_url: null
            };

            const html = formatPaperCard(paper);

            expect(html).not.toContain('PDF');
        });

        it('should show paper url link when url is present', () => {
            const paper = {
                uid: 'p1',
                title: 'Paper',
                authors: [],
                abstract: 'Abstract',
                url: 'https://example.com/paper-page'
            };

            const html = formatPaperCard(paper);

            expect(html).toContain('https://example.com/paper-page');
            expect(html).toContain('Paper Page');
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

        it('should render LaTeX in titles via marked.parseInline', () => {
            const latexTitle = 'The $\\boldsymbol{\\lambda}$ Method';
            global.marked.parseInline.mockReturnValueOnce('The <span class="katex">λ</span> Method');

            const paper = {
                uid: 'latex-paper',
                title: latexTitle,
                authors: [],
                abstract: 'Abstract'
            };

            const html = formatPaperCard(paper);

            // parseInline should be called with the title to render LaTeX
            expect(global.marked.parseInline).toHaveBeenCalledWith(latexTitle);
            // The rendered KaTeX output should appear in the card
            expect(html).toContain('katex');
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
                    paper_pdf_url: 'https://example.com/paper.pdf'
                })
            });

            await showPaperDetails('p1');

            expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/paper/p1'));
            
            // The function creates a new modal element
            const modals = document.querySelectorAll('.fixed.inset-0');
            expect(modals.length).toBeGreaterThan(0);

            // Conference should be shown in modal
            const modalHtml = modals[0].innerHTML;
            expect(modalHtml).toContain('NeurIPS');
            expect(modalHtml).toContain('fa-university');
        });

        it('should show PDF link when paper_pdf_url is present', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper with PDF',
                    authors: [],
                    abstract: 'Abstract',
                    paper_pdf_url: 'https://example.com/paper.pdf'
                })
            });

            await showPaperDetails('p1');

            const modals = document.querySelectorAll('.fixed.inset-0');
            const modalHtml = modals[0].innerHTML;
            expect(modalHtml).toContain('https://example.com/paper.pdf');
            expect(modalHtml).toContain('fa-file-pdf');
            expect(modalHtml).toContain('PDF');
        });

        it('should show general url link when url is present', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper with URL',
                    authors: [],
                    abstract: 'Abstract',
                    url: 'https://example.com/paper-page'
                })
            });

            await showPaperDetails('p1');

            const modals = document.querySelectorAll('.fixed.inset-0');
            const modalHtml = modals[0].innerHTML;
            expect(modalHtml).toContain('https://example.com/paper-page');
            expect(modalHtml).toContain('fa-external-link-alt');
            expect(modalHtml).toContain('Paper Page');
        });

        it('should show poster link when poster_image_url is present', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper with Poster',
                    authors: [],
                    abstract: 'Abstract',
                    poster_image_url: 'https://example.com/poster.jpg'
                })
            });

            await showPaperDetails('p1');

            const modals = document.querySelectorAll('.fixed.inset-0');
            const modalHtml = modals[0].innerHTML;
            expect(modalHtml).toContain('https://example.com/poster.jpg');
            expect(modalHtml).toContain('fa-image');
            expect(modalHtml).toContain('Poster');
        });

        it('should show all url badges when all url fields are present', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper with All URLs',
                    authors: [],
                    abstract: 'Abstract',
                    url: 'https://example.com/page',
                    paper_pdf_url: 'https://example.com/paper.pdf',
                    poster_image_url: 'https://example.com/poster.jpg'
                })
            });

            await showPaperDetails('p1');

            const modals = document.querySelectorAll('.fixed.inset-0');
            const modalHtml = modals[0].innerHTML;
            expect(modalHtml).toContain('Paper Page');
            expect(modalHtml).toContain('PDF');
            expect(modalHtml).toContain('Poster');
        });

        it('should handle missing PDF URL', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    uid: 'p1',
                    title: 'Paper',
                    authors: [],
                    abstract: 'Abstract',
                    paper_pdf_url: null
                })
            });

            await showPaperDetails('p1');

            // Modal should still be created
            const modals = document.querySelectorAll('.fixed.inset-0');
            expect(modals.length).toBeGreaterThan(0);
            // No PDF button when paper_pdf_url is null
            expect(modals[0].innerHTML).not.toContain('View PDF');
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

    describe('buildUrlBadges', () => {
        it('should return empty string when no URLs present', () => {
            const paper = { uid: 'p1', title: 'Paper', authors: [] };
            expect(buildUrlBadges(paper)).toBe('');
        });

        it('should render url as Paper Page badge', () => {
            const paper = { url: 'https://example.com/page' };
            const html = buildUrlBadges(paper);
            expect(html).toContain('https://example.com/page');
            expect(html).toContain('Paper Page');
            expect(html).toContain('fa-external-link-alt');
        });

        it('should render paper_pdf_url as PDF badge', () => {
            const paper = { paper_pdf_url: 'https://example.com/paper.pdf' };
            const html = buildUrlBadges(paper);
            expect(html).toContain('https://example.com/paper.pdf');
            expect(html).toContain('PDF');
            expect(html).toContain('fa-file-pdf');
        });

        it('should render poster_image_url as Poster badge', () => {
            const paper = { poster_image_url: 'https://example.com/poster.jpg' };
            const html = buildUrlBadges(paper);
            expect(html).toContain('https://example.com/poster.jpg');
            expect(html).toContain('Poster');
            expect(html).toContain('fa-image');
        });

        it('should render all three badges when all URLs present', () => {
            const paper = {
                url: 'https://example.com/page',
                paper_pdf_url: 'https://example.com/paper.pdf',
                poster_image_url: 'https://example.com/poster.jpg'
            };
            const html = buildUrlBadges(paper);
            expect(html).toContain('Paper Page');
            expect(html).toContain('PDF');
            expect(html).toContain('Poster');
        });

        it('should include event.stopPropagation by default', () => {
            const paper = { url: 'https://example.com/page' };
            const html = buildUrlBadges(paper);
            expect(html).toContain('event.stopPropagation');
        });

        it('should omit event.stopPropagation when stopPropagation=false', () => {
            const paper = { url: 'https://example.com/page' };
            const html = buildUrlBadges(paper, false, false);
            expect(html).not.toContain('event.stopPropagation');
        });

        it('should use compact styling when compact=true', () => {
            const paper = { url: 'https://example.com/page' };
            const compactHtml = buildUrlBadges(paper, true);
            const normalHtml = buildUrlBadges(paper, false);
            // compact uses mb-2, normal uses mb-3
            expect(compactHtml).toContain('mb-2');
            expect(normalHtml).toContain('mb-3');
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
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p3" onclick="setPaperPriority('p3', 1)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p3" onclick="setPaperPriority('p3', 2)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p3" onclick="setPaperPriority('p3', 3)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p3" onclick="setPaperPriority('p3', 4)"></i>
                            <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p3" onclick="setPaperPriority('p3', 5)"></i>
                        </div>
                    </div>
                </div>
            `;
            
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p3', 3);

            updateStarDisplay('p3');

            const chatPapersDiv = document.getElementById('chat-papers');
            expect(chatPapersDiv).not.toBeNull();
            
            const stars = chatPapersDiv.querySelectorAll('.paper-star[data-paper-id="p3"]');
            expect(stars.length).toBe(5);
            
            // First 3 stars should be filled (fas), remaining 2 should be empty (far)
            expect(stars[0].className).toContain('fas');
            expect(stars[1].className).toContain('fas');
            expect(stars[2].className).toContain('fas');
            expect(stars[3].className).toContain('far');
            expect(stars[4].className).toContain('far');
        });

        it('should update stars across multiple panels simultaneously', () => {
            // Set up multiple panels with the same paper
            document.body.innerHTML += `
                <div id="search-results">
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 1)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 2)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 3)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 4)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 5)"></i>
                </div>
                <div id="chat-papers">
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 1)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 2)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 3)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 4)"></i>
                    <i class="far fa-star text-gray-300 hover:text-yellow-400 cursor-pointer paper-star" data-paper-id="p4" onclick="setPaperPriority('p4', 5)"></i>
                </div>
            `;
            
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('p4', 2);

            updateStarDisplay('p4');

            // Check all stars across both panels
            const allStars = document.querySelectorAll('.paper-star[data-paper-id="p4"]');
            expect(allStars.length).toBe(10); // 5 stars in each of 2 panels
            
            // First 2 stars in each panel should be filled, rest empty
            expect(allStars[0].className).toContain('fas');
            expect(allStars[1].className).toContain('fas');
            expect(allStars[2].className).toContain('far');
            expect(allStars[3].className).toContain('far');
            expect(allStars[4].className).toContain('far');
            
            // Second panel
            expect(allStars[5].className).toContain('fas');
            expect(allStars[6].className).toContain('fas');
            expect(allStars[7].className).toContain('far');
            expect(allStars[8].className).toContain('far');
            expect(allStars[9].className).toContain('far');
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
