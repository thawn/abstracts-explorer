/**
 * Tests for utility modules (consolidated)
 */

import { jest } from '@jest/globals';

// Mock dependencies
global.marked = { parse: jest.fn((text) => text), parseInline: jest.fn((text) => text), use: jest.fn() };
global.markedKatex = jest.fn(() => ({}));

// Import all utility modules
import { escapeHtml } from '../static/modules/utils/dom-utils.js';
import { renderEmptyState, renderErrorBlock, showLoading, showErrorInElement, showError } from '../static/modules/utils/ui-utils.js';
import { naturalSortPosterPosition, sortClustersBySizeDesc } from '../static/modules/utils/sort-utils.js';
import { getClusterLabelWithCount } from '../static/modules/utils/cluster-utils.js';
import { renderMarkdownWithLatex, configureMarkedWithKatex, renderInlineMarkdownWithLatex } from '../static/modules/utils/markdown-utils.js';
import { parseJsonResponse } from '../static/modules/utils/api-utils.js';
import { API_BASE, PLOTLY_COLORS } from '../static/modules/utils/constants.js';

describe('Utility Modules', () => {
    describe('DOM Utils', () => {
        describe('escapeHtml', () => {
            it('should escape HTML special characters', () => {
                expect(escapeHtml('<script>alert("xss")</script>'))
                    .toBe('&lt;script&gt;alert("xss")&lt;/script&gt;');
            });

            it('should escape ampersands', () => {
                expect(escapeHtml('Tom & Jerry')).toBe('Tom &amp; Jerry');
            });

            it('should handle quotes', () => {
                const result = escapeHtml('"quoted"');
                // Note: textContent doesn't convert quotes to &quot;
                expect(result).toContain('quoted');
            });

            it('should handle plain text', () => {
                expect(escapeHtml('plain text')).toBe('plain text');
            });

            it('should handle empty string', () => {
                expect(escapeHtml('')).toBe('');
            });

            it('should handle special characters', () => {
                expect(escapeHtml("'single quotes'")).toContain("'");
                expect(escapeHtml('< > & "')).toContain('&lt;');
            });
        });
    });

    describe('UI Utils', () => {
        beforeEach(() => {
            document.body.innerHTML = '';
        });

        describe('renderEmptyState', () => {
            it('should render with default icon', () => {
                const html = renderEmptyState('No items', 'Try again');
                expect(html).toContain('fa-inbox');
                expect(html).toContain('No items');
                expect(html).toContain('Try again');
            });

            it('should render with custom icon', () => {
                const html = renderEmptyState('No papers', 'Search', 'fa-search');
                expect(html).toContain('fa-search');
            });

            it('should escape HTML in messages', () => {
                const html = renderEmptyState('<script>xss</script>', 'text');
                expect(html).toContain('&lt;script&gt;');
            });
        });

        describe('renderErrorBlock', () => {
            it('should render error block', () => {
                const html = renderErrorBlock('Something went wrong');
                expect(html).toContain('bg-red-50');
                expect(html).toContain('Something went wrong');
            });

            it('should escape HTML', () => {
                const html = renderErrorBlock('<img src=x onerror=alert(1)>');
                expect(html).not.toContain('<img');
            });
        });


        describe('showLoading', () => {
            it('should show loading in element', () => {
                document.body.innerHTML = '<div id="test"></div>';
                showLoading('test', 'Loading');
                
                const el = document.getElementById('test');
                expect(el.innerHTML).toContain('fa-spinner');
            });

            it('should handle missing element', () => {
                expect(() => showLoading('missing', 'Loading')).not.toThrow();
            });
        });

        describe('showErrorInElement', () => {
            it('should show error in element', () => {
                document.body.innerHTML = '<div id="test"></div>';
                showErrorInElement('test', 'Error!');
                
                const el = document.getElementById('test');
                expect(el.innerHTML).toContain('Error!');
            });
        });

        describe('showError', () => {
            it('should show error in search results', () => {
                document.body.innerHTML = '<div id="search-results"></div>';
                showError('Search failed');
                
                const el = document.getElementById('search-results');
                expect(el.innerHTML).toContain('Search failed');
            });
        });
    });

    describe('Sort Utils', () => {
        describe('naturalSortPosterPosition', () => {
            it('should sort numbers naturally', () => {
                expect(naturalSortPosterPosition('Board 9', 'Board 10')).toBeLessThan(0);
                expect(naturalSortPosterPosition('Board 99', 'Board 100')).toBeLessThan(0);
            });

            it('should handle plain numbers', () => {
                expect(naturalSortPosterPosition('9', '10')).toBeLessThan(0);
            });

            it('should handle equal strings', () => {
                expect(naturalSortPosterPosition('Board 10', 'Board 10')).toBe(0);
            });

            it('should handle empty strings', () => {
                expect(naturalSortPosterPosition('', '')).toBe(0);
            });

            it('should handle mixed content', () => {
                expect(naturalSortPosterPosition('Poster 5', 'Poster 10')).toBeLessThan(0);
            });
        });

        describe('sortClustersBySizeDesc', () => {
            it('should sort by size descending', () => {
                const clusters = [
                    ['1', [1, 2]],
                    ['2', [1, 2, 3, 4]],
                    ['3', [1]]
                ];
                const sorted = sortClustersBySizeDesc(clusters);
                expect(sorted[0][0]).toBe('2'); // 4 items
                expect(sorted[1][0]).toBe('1'); // 2 items
                expect(sorted[2][0]).toBe('3'); // 1 item
            });

            it('should use ID as tiebreaker', () => {
                const clusters = [
                    ['3', [1, 2]],
                    ['1', [1, 2]],
                    ['2', [1, 2]]
                ];
                const sorted = sortClustersBySizeDesc(clusters);
                expect(sorted[0][0]).toBe('1');
                expect(sorted[2][0]).toBe('3');
            });

            it('should handle numeric counts', () => {
                const clusters = [['1', 10], ['2', 5], ['3', 15]];
                const sorted = sortClustersBySizeDesc(clusters);
                expect(sorted[0][0]).toBe('3');
            });
        });
    });

    describe('Cluster Utils', () => {
        describe('getClusterLabelWithCount', () => {
            it('should use provided label', () => {
                const labels = { '1': 'ML Papers' };
                expect(getClusterLabelWithCount('1', labels, 10)).toBe('ML Papers (10)');
            });

            it('should use default label', () => {
                expect(getClusterLabelWithCount('3', {}, 5)).toBe('Cluster 3 (5)');
            });
        });
    });

    describe('Markdown Utils', () => {
        describe('renderInlineMarkdownWithLatex', () => {
            it('should render inline markdown', () => {
                global.marked.parseInline.mockReturnValueOnce('<strong>Test</strong>');
                const result = renderInlineMarkdownWithLatex('**Test**');
                expect(result).toContain('Test');
            });

            it('should handle empty string', () => {
                expect(renderInlineMarkdownWithLatex('')).toBe('');
            });

            it('should handle parsing errors gracefully', () => {
                global.marked.parseInline.mockImplementationOnce(() => {
                    throw new Error('Parse error');
                });
                const result = renderInlineMarkdownWithLatex('Title text');
                expect(result).toContain('Title text');
            });

            it('should escape HTML on parsing error fallback', () => {
                global.marked.parseInline.mockImplementationOnce(() => {
                    throw new Error('Parse error');
                });
                const result = renderInlineMarkdownWithLatex('<script>xss</script>');
                expect(result).not.toContain('<script>');
                expect(result).toContain('&lt;script&gt;');
            });
        });

        describe('renderMarkdownWithLatex', () => {
            it('should render markdown', () => {
                global.marked.parse.mockReturnValueOnce('<p>Test</p>');
                const result = renderMarkdownWithLatex('Test');
                expect(result).toContain('Test');
            });

            it('should handle empty string', () => {
                expect(renderMarkdownWithLatex('')).toBe('');
            });

            it('should handle parsing errors gracefully', () => {
                global.marked.parse.mockImplementationOnce(() => {
                    throw new Error('Parse error');
                });
                const result = renderMarkdownWithLatex('Text');
                expect(result).toContain('Text');
            });
        });

        describe('configureMarkedWithKatex', () => {
            it('should configure marked', () => {
                configureMarkedWithKatex();
                expect(global.marked.use).toHaveBeenCalled();
            });
        });
    });

    describe('API Utils', () => {
        beforeEach(() => {
            document.body.innerHTML = '';
        });





        describe('parseJsonResponse', () => {
            it('should parse a JSON response', async () => {
                const data = await parseJsonResponse({
                    ok: true,
                    status: 200,
                    json: async () => ({ value: 42 })
                });

                expect(data).toEqual({ value: 42 });
            });

            it('should report a helpful error for an HTML gateway timeout', async () => {
                const response = {
                    ok: false,
                    status: 504,
                    json: async () => {
                        throw new SyntaxError('Unexpected token <');
                    }
                };

                await expect(parseJsonResponse(response)).rejects.toThrow(
                    'The server took too long to respond'
                );
            });

            it('should surface the server error for a non-OK JSON response', async () => {
                const response = {
                    ok: false,
                    status: 500,
                    json: async () => ({ error: 'boom' })
                };

                await expect(parseJsonResponse(response)).rejects.toThrow('boom');
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
            expect(PLOTLY_COLORS[0]).toMatch(/^#/);
        });
    });
});
