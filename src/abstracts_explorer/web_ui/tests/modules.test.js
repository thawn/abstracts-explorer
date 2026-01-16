/**
 * Unit tests for refactored modules
 */

import { jest } from '@jest/globals';

// Mock global objects before importing modules
global.fetch = jest.fn();
global.marked = {
    parse: jest.fn((text) => text),
    use: jest.fn()
};
global.markedKatex = jest.fn(() => ({}));
global.Plotly = {
    newPlot: jest.fn(),
    relayout: jest.fn()
};

// Import modules
import { escapeHtml } from '../static/modules/utils/dom-utils.js';
import { renderEmptyState, renderErrorBlock } from '../static/modules/utils/ui-utils.js';
import { naturalSortPosterPosition, sortClustersBySizeDesc } from '../static/modules/utils/sort-utils.js';
import * as State from '../static/modules/state.js';

describe('Refactored Modules', () => {
    describe('DOM Utils', () => {
        it('should escape HTML', () => {
            expect(escapeHtml('<script>alert("xss")</script>'))
                .toBe('&lt;script&gt;alert("xss")&lt;/script&gt;');
        });
    });

    describe('UI Utils', () => {
        it('should render empty state', () => {
            const html = renderEmptyState('No items', 'Try again');
            expect(html).toContain('No items');
        });

        it('should render error block', () => {
            const html = renderErrorBlock('Error');
            expect(html).toContain('Error');
        });
    });

    describe('Sort Utils', () => {
        it('should sort poster positions naturally', () => {
            expect(naturalSortPosterPosition('9', '10')).toBeLessThan(0);
        });

        it('should sort clusters by size', () => {
            const clusters = [['1', [1]], ['2', [1, 2, 3]]];
            const sorted = sortClustersBySizeDesc(clusters);
            expect(sorted[0][0]).toBe('2');
        });
    });

    describe('State Management', () => {
        beforeEach(() => {
            localStorage.clear();
            State.loadPriorities();
        });

        it('should set and get current tab', () => {
            State.setCurrentTab('chat');
            expect(State.getCurrentTab()).toBe('chat');
        });

        it('should manage paper priorities', () => {
            State.setCurrentSearchTerm('test');
            const result = State.setPaperPriority('paper1', 5);
            expect(result).toBe(true);
            expect(State.getPaperPriority('paper1')).toBe(5);
        });
    });
});
