/**
 * Unit tests for utility modules
 */

import { escapeHtml, renderEmptyState, renderErrorBlock, showLoading, showErrorInElement, showError } from '../static/modules/utils/dom-utils.js';
import { naturalSortPosterPosition, sortClustersBySizeDesc } from '../static/modules/utils/sort-utils.js';
import { getSelectedFilters, applyYearConferenceFilters, buildFilteredRequestBody } from '../static/modules/utils/filter-utils.js';
import { getClusterLabelWithCount } from '../static/modules/utils/cluster-utils.js';
import { renderMarkdownWithLatex, configureMarkedWithKatex } from '../static/modules/utils/markdown-utils.js';
import { API_BASE, PLOTLY_COLORS } from '../static/modules/utils/constants.js';

describe('DOM Utilities', () => {
    describe('escapeHtml', () => {
        it('should escape HTML special characters', () => {
            expect(escapeHtml('<script>alert("xss")</script>'))
                .toBe('&lt;script&gt;alert("xss")&lt;/script&gt;');
        });

        it('should escape ampersands', () => {
            expect(escapeHtml('Tom & Jerry')).toBe('Tom &amp; Jerry');
        });

        it('should handle quotes', () => {
            expect(escapeHtml('"quoted"')).toContain('&quot;');
        });

        it('should handle plain text', () => {
            expect(escapeHtml('plain text')).toBe('plain text');
        });

        it('should handle empty string', () => {
            expect(escapeHtml('')).toBe('');
        });
    });

    describe('renderEmptyState', () => {
        it('should render empty state with default icon', () => {
            const html = renderEmptyState('No items', 'Try again');
            expect(html).toContain('fa-inbox');
            expect(html).toContain('No items');
            expect(html).toContain('Try again');
        });

        it('should render empty state with custom icon', () => {
            const html = renderEmptyState('No papers', 'Search for papers', 'fa-search');
            expect(html).toContain('fa-search');
            expect(html).toContain('No papers');
        });
    });

    describe('renderErrorBlock', () => {
        it('should render error block', () => {
            const html = renderErrorBlock('Something went wrong');
            expect(html).toContain('bg-red-50');
            expect(html).toContain('Something went wrong');
            expect(html).toContain('fa-exclamation-circle');
        });

        it('should escape HTML in error message', () => {
            const html = renderErrorBlock('<script>alert("xss")</script>');
            expect(html).toContain('&lt;script&gt;');
            expect(html).not.toContain('<script>alert');
        });
    });

    describe('showLoading', () => {
        beforeEach(() => {
            document.body.innerHTML = '<div id="test-element"></div>';
        });

        it('should show loading message in element', () => {
            showLoading('test-element', 'Loading data');
            const element = document.getElementById('test-element');
            expect(element.innerHTML).toContain('fa-spinner');
            expect(element.innerHTML).toContain('Loading data');
        });

        it('should handle missing element', () => {
            // Should not throw error
            showLoading('non-existent', 'Loading');
            // Just verify it doesn't crash
            expect(true).toBe(true);
        });
    });

    describe('showErrorInElement', () => {
        beforeEach(() => {
            document.body.innerHTML = '<div id="test-element"></div>';
        });

        it('should show error in element', () => {
            showErrorInElement('test-element', 'Error occurred');
            const element = document.getElementById('test-element');
            expect(element.innerHTML).toContain('Error occurred');
            expect(element.innerHTML).toContain('bg-red-50');
        });
    });

    describe('showError', () => {
        beforeEach(() => {
            document.body.innerHTML = '<div id="search-results"></div>';
        });

        it('should show error in search results', () => {
            showError('Search failed');
            const element = document.getElementById('search-results');
            expect(element.innerHTML).toContain('Search failed');
        });
    });
});

describe('Sort Utilities', () => {
    describe('naturalSortPosterPosition', () => {
        it('should sort numbers naturally', () => {
            expect(naturalSortPosterPosition('Board 9', 'Board 10')).toBeLessThan(0);
            expect(naturalSortPosterPosition('Board 99', 'Board 100')).toBeLessThan(0);
            expect(naturalSortPosterPosition('Board 100', 'Board 99')).toBeGreaterThan(0);
        });

        it('should handle plain numbers', () => {
            expect(naturalSortPosterPosition('9', '10')).toBeLessThan(0);
            expect(naturalSortPosterPosition('99', '100')).toBeLessThan(0);
        });

        it('should handle equal numbers', () => {
            expect(naturalSortPosterPosition('Board 10', 'Board 10')).toBe(0);
        });

        it('should handle strings without numbers', () => {
            expect(naturalSortPosterPosition('A', 'B')).toBeLessThan(0);
            expect(naturalSortPosterPosition('B', 'A')).toBeGreaterThan(0);
        });

        it('should handle empty strings', () => {
            expect(naturalSortPosterPosition('', '')).toBe(0);
            expect(naturalSortPosterPosition('A', '')).toBeGreaterThan(0);
        });
    });

    describe('sortClustersBySizeDesc', () => {
        it('should sort by size descending', () => {
            const clusters = [
                ['1', [1, 2, 3]],
                ['2', [1, 2]],
                ['3', [1, 2, 3, 4, 5]]
            ];
            const sorted = sortClustersBySizeDesc(clusters);
            expect(sorted[0][0]).toBe('3'); // 5 items
            expect(sorted[1][0]).toBe('1'); // 3 items
            expect(sorted[2][0]).toBe('2'); // 2 items
        });

        it('should use ID as tiebreaker', () => {
            const clusters = [
                ['3', [1, 2, 3]],
                ['1', [1, 2, 3]],
                ['2', [1, 2, 3]]
            ];
            const sorted = sortClustersBySizeDesc(clusters);
            expect(sorted[0][0]).toBe('1'); // Same size, lowest ID
            expect(sorted[1][0]).toBe('2');
            expect(sorted[2][0]).toBe('3');
        });

        it('should handle numeric counts', () => {
            const clusters = [
                ['1', 10],
                ['2', 20],
                ['3', 5]
            ];
            const sorted = sortClustersBySizeDesc(clusters);
            expect(sorted[0][0]).toBe('2'); // 20
            expect(sorted[1][0]).toBe('1'); // 10
            expect(sorted[2][0]).toBe('3'); // 5
        });
    });
});

describe('Filter Utilities', () => {
    describe('getSelectedFilters', () => {
        beforeEach(() => {
            document.body.innerHTML = `
                <select id="year-selector"><option value="2025" selected>2025</option></select>
                <select id="conference-selector"><option value="NeurIPS" selected>NeurIPS</option></select>
            `;
        });

        it('should get selected filters', () => {
            const filters = getSelectedFilters();
            expect(filters.selectedYear).toBe('2025');
            expect(filters.selectedConference).toBe('NeurIPS');
        });

        it('should handle missing selectors', () => {
            document.body.innerHTML = '';
            const filters = getSelectedFilters();
            expect(filters.selectedYear).toBe('');
            expect(filters.selectedConference).toBe('');
        });
    });

    describe('applyYearConferenceFilters', () => {
        const papers = [
            { id: 1, year: 2025, conference: 'NeurIPS' },
            { id: 2, year: 2024, conference: 'ICLR' },
            { id: 3, year: 2025, conference: 'ICLR' },
            { id: 4, year: 2024, conference: 'NeurIPS' }
        ];

        it('should filter by year', () => {
            const filtered = applyYearConferenceFilters(papers, '2025', '');
            expect(filtered.length).toBe(2);
            expect(filtered.every(p => p.year === 2025)).toBe(true);
        });

        it('should filter by conference', () => {
            const filtered = applyYearConferenceFilters(papers, '', 'NeurIPS');
            expect(filtered.length).toBe(2);
            expect(filtered.every(p => p.conference === 'NeurIPS')).toBe(true);
        });

        it('should filter by both year and conference', () => {
            const filtered = applyYearConferenceFilters(papers, '2025', 'NeurIPS');
            expect(filtered.length).toBe(1);
            expect(filtered[0].id).toBe(1);
        });

        it('should return all papers when no filters', () => {
            const filtered = applyYearConferenceFilters(papers, '', '');
            expect(filtered.length).toBe(4);
        });
    });

    describe('buildFilteredRequestBody', () => {
        it('should add session filter when partial selection', () => {
            const body = buildFilteredRequestBody(
                { query: 'test' },
                ['Session A', 'Session B'],
                3,
                '',
                ''
            );
            expect(body.sessions).toEqual(['Session A', 'Session B']);
        });

        it('should not add session filter when all selected', () => {
            const body = buildFilteredRequestBody(
                { query: 'test' },
                ['A', 'B', 'C'],
                3,
                '',
                ''
            );
            expect(body.sessions).toBeUndefined();
        });

        it('should add year filter', () => {
            const body = buildFilteredRequestBody(
                { query: 'test' },
                [],
                0,
                '2025',
                ''
            );
            expect(body.years).toEqual([2025]);
        });

        it('should add conference filter', () => {
            const body = buildFilteredRequestBody(
                { query: 'test' },
                [],
                0,
                '',
                'NeurIPS'
            );
            expect(body.conferences).toEqual(['NeurIPS']);
        });
    });
});

describe('Cluster Utilities', () => {
    describe('getClusterLabelWithCount', () => {
        it('should use provided label', () => {
            const labels = { '1': 'Machine Learning', '2': 'Computer Vision' };
            expect(getClusterLabelWithCount('1', labels, 10)).toBe('Machine Learning (10)');
        });

        it('should use default label when not provided', () => {
            const labels = {};
            expect(getClusterLabelWithCount('3', labels, 5)).toBe('Cluster 3 (5)');
        });
    });
});

describe('Markdown Utilities', () => {
    beforeEach(() => {
        // Mock marked library
        global.marked = {
            parse: jest.fn((text) => `<p>${text}</p>`),
            use: jest.fn()
        };
        global.markedKatex = jest.fn(() => ({}));
    });

    describe('renderMarkdownWithLatex', () => {
        it('should render markdown', () => {
            const result = renderMarkdownWithLatex('Hello **world**');
            expect(result).toContain('Hello **world**');
            expect(global.marked.parse).toHaveBeenCalled();
        });

        it('should handle empty string', () => {
            expect(renderMarkdownWithLatex('')).toBe('');
        });

        it('should handle parsing errors', () => {
            global.marked.parse = jest.fn(() => {
                throw new Error('Parse error');
            });
            const result = renderMarkdownWithLatex('Test');
            expect(result).toContain('Test');
        });
    });

    describe('configureMarkedWithKatex', () => {
        it('should configure marked with KaTeX', () => {
            configureMarkedWithKatex();
            expect(global.marked.use).toHaveBeenCalled();
        });

        it('should handle missing dependencies', () => {
            global.markedKatex = undefined;
            // Should not throw
            configureMarkedWithKatex();
            expect(true).toBe(true);
        });
    });
});

describe('Constants', () => {
    it('should export API_BASE', () => {
        expect(API_BASE).toBe('');
    });

    it('should export PLOTLY_COLORS', () => {
        expect(PLOTLY_COLORS).toBeInstanceOf(Array);
        expect(PLOTLY_COLORS.length).toBe(48);
        expect(PLOTLY_COLORS[0]).toBe('#2E91E5');
    });
});
