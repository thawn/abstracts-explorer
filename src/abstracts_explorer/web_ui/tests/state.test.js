/**
 * Comprehensive tests for state management module
 */

import { jest } from '@jest/globals';

import * as State from '../static/modules/state.js';

describe('State Management Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorage.clear();
        // Reset state to initial values
        State.setCurrentTab('search');
        State.setCurrentSearchTerm('');
        State.setCurrentInterestingSession(null);
        State.setInterestingPapersSortOrder('search-rating-poster');
        State.loadPriorities();
    });

    describe('Tab Management', () => {
        it('should set and get current tab', () => {
            State.setCurrentTab('chat');
            expect(State.getCurrentTab()).toBe('chat');
        });

        it('should start with search tab by default', () => {
            expect(State.getCurrentTab()).toBe('search');
        });

        it('should handle tab changes', () => {
            State.setCurrentTab('search');
            expect(State.getCurrentTab()).toBe('search');
            
            State.setCurrentTab('chat');
            expect(State.getCurrentTab()).toBe('chat');
            
            State.setCurrentTab('interesting');
            expect(State.getCurrentTab()).toBe('interesting');
        });
    });

    describe('Search Term Management', () => {
        it('should set and get current search term', () => {
            State.setCurrentSearchTerm('machine learning');
            expect(State.getCurrentSearchTerm()).toBe('machine learning');
        });

        it('should handle empty search term', () => {
            State.setCurrentSearchTerm('');
            expect(State.getCurrentSearchTerm()).toBe('');
        });

        it('should update search term', () => {
            State.setCurrentSearchTerm('first query');
            expect(State.getCurrentSearchTerm()).toBe('first query');
            
            State.setCurrentSearchTerm('second query');
            expect(State.getCurrentSearchTerm()).toBe('second query');
        });
    });

    describe('Interesting Session Management', () => {
        it('should set and get interesting session', () => {
            State.setCurrentInterestingSession('Session 1');
            expect(State.getCurrentInterestingSession()).toBe('Session 1');
        });

        it('should start with null session', () => {
            expect(State.getCurrentInterestingSession()).toBeNull();
        });

        it('should handle session changes', () => {
            State.setCurrentInterestingSession('Session A');
            expect(State.getCurrentInterestingSession()).toBe('Session A');
            
            State.setCurrentInterestingSession('Session B');
            expect(State.getCurrentInterestingSession()).toBe('Session B');
        });
    });

    describe('Sort Order Management', () => {
        it('should set and get sort order', () => {
            State.setInterestingPapersSortOrder('rating-poster-search');
            expect(State.getInterestingPapersSortOrder()).toBe('rating-poster-search');
        });

        it('should persist sort order to localStorage', () => {
            State.setInterestingPapersSortOrder('poster-search-rating');
            expect(localStorage.getItem('interestingPapersSortOrder')).toBe('poster-search-rating');
        });

        it('should load sort order from localStorage', () => {
            localStorage.setItem('interestingPapersSortOrder', 'rating-poster-search');
            State.loadPriorities();
            expect(State.getInterestingPapersSortOrder()).toBe('rating-poster-search');
        });

        it('should use default sort order when not in localStorage', () => {
            State.loadPriorities();
            expect(State.getInterestingPapersSortOrder()).toBe('search-rating-poster');
        });
    });

    describe('Paper Priorities', () => {
        beforeEach(() => {
            State.setCurrentSearchTerm('test search');
        });

        it('should set paper priority', () => {
            const result = State.setPaperPriority('paper1', 5);
            expect(result).toBe(true);
            expect(State.getPaperPriority('paper1')).toBe(5);
        });

        it('should set priority for multiple papers', () => {
            State.setPaperPriority('paper1', 5);
            State.setPaperPriority('paper2', 3);
            State.setPaperPriority('paper3', 4);

            expect(State.getPaperPriority('paper1')).toBe(5);
            expect(State.getPaperPriority('paper2')).toBe(3);
            expect(State.getPaperPriority('paper3')).toBe(4);
        });

        it('should remove priority when clicking same star', () => {
            State.setPaperPriority('paper1', 3);
            const result = State.setPaperPriority('paper1', 3);
            expect(result).toBe(false);
            expect(State.getPaperPriority('paper1')).toBe(0);
        });

        it('should remove priority when set to 0', () => {
            State.setPaperPriority('paper1', 4);
            const result = State.setPaperPriority('paper1', 0);
            expect(result).toBe(false);
            expect(State.getPaperPriority('paper1')).toBe(0);
        });

        it('should update existing priority', () => {
            State.setPaperPriority('paper1', 3);
            State.setPaperPriority('paper1', 5);
            expect(State.getPaperPriority('paper1')).toBe(5);
        });

        it('should preserve search term when updating priority', () => {
            State.setCurrentSearchTerm('first search');
            State.setPaperPriority('paper1', 3);
            
            State.setCurrentSearchTerm('second search');
            State.setPaperPriority('paper1', 5);
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('first search');
            expect(priorities['paper1'].priority).toBe(5);
        });

        it('should use current search term for new priorities', () => {
            State.setCurrentSearchTerm('my search query');
            State.setPaperPriority('paper1', 4);
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('my search query');
        });

        it('should return 0 for non-existent paper', () => {
            expect(State.getPaperPriority('non-existent')).toBe(0);
        });

        it('should handle priority range 1-5', () => {
            for (let i = 1; i <= 5; i++) {
                State.setPaperPriority(`paper${i}`, i);
                expect(State.getPaperPriority(`paper${i}`)).toBe(i);
            }
        });

        it('should get all paper priorities', () => {
            State.setPaperPriority('paper1', 5);
            State.setPaperPriority('paper2', 3);
            State.setPaperPriority('paper3', 4);
            
            const priorities = State.getAllPaperPriorities();
            expect(Object.keys(priorities).length).toBe(3);
            expect(priorities['paper1'].priority).toBe(5);
            expect(priorities['paper2'].priority).toBe(3);
            expect(priorities['paper3'].priority).toBe(4);
        });

        it('should get paper IDs with priorities', () => {
            State.setPaperPriority('paper1', 5);
            State.setPaperPriority('paper2', 3);
            State.setPaperPriority('paper3', 4);
            
            const ids = State.getPaperIdsWithPriorities();
            expect(ids).toContain('paper1');
            expect(ids).toContain('paper2');
            expect(ids).toContain('paper3');
            expect(ids.length).toBe(3);
        });

        it('should return empty array when no priorities', () => {
            const ids = State.getPaperIdsWithPriorities();
            expect(ids.length).toBe(0);
        });
    });

    describe('Priority Persistence', () => {
        it('should save priorities to localStorage', () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('paper1', 5);
            State.setPaperPriority('paper2', 3);
            
            const stored = localStorage.getItem('paperPriorities');
            expect(stored).not.toBeNull();
            
            const parsed = JSON.parse(stored);
            expect(parsed['paper1'].priority).toBe(5);
            expect(parsed['paper2'].priority).toBe(3);
        });

        it('should load priorities from localStorage', () => {
            const priorities = {
                paper1: { priority: 5, searchTerm: 'test' },
                paper2: { priority: 3, searchTerm: 'another test' },
                paper3: { priority: 4, searchTerm: 'third test' }
            };
            localStorage.setItem('paperPriorities', JSON.stringify(priorities));
            
            State.loadPriorities();
            
            expect(State.getPaperPriority('paper1')).toBe(5);
            expect(State.getPaperPriority('paper2')).toBe(3);
            expect(State.getPaperPriority('paper3')).toBe(4);
        });

        it('should handle invalid JSON in localStorage', () => {
            localStorage.setItem('paperPriorities', 'invalid json {{{');
            
            // Should not throw
            State.loadPriorities();
            
            expect(State.getPaperIdsWithPriorities().length).toBe(0);
        });

        it('should handle missing localStorage item', () => {
            State.loadPriorities();
            
            expect(State.getPaperIdsWithPriorities().length).toBe(0);
        });

        it('should persist across multiple operations', () => {
            State.setCurrentSearchTerm('query1');
            State.setPaperPriority('p1', 5);
            
            State.setCurrentSearchTerm('query2');
            State.setPaperPriority('p2', 3);
            
            State.loadPriorities();
            
            expect(State.getPaperPriority('p1')).toBe(5);
            expect(State.getPaperPriority('p2')).toBe(3);
        });

        it('should save immediately after setting priority', () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('paper1', 4);
            
            // Check localStorage was updated
            const stored = localStorage.getItem('paperPriorities');
            const parsed = JSON.parse(stored);
            expect(parsed['paper1']).toBeDefined();
        });

        it('should save immediately after removing priority', () => {
            State.setCurrentSearchTerm('test');
            State.setPaperPriority('paper1', 4);
            State.setPaperPriority('paper1', 0);
            
            const stored = localStorage.getItem('paperPriorities');
            const parsed = JSON.parse(stored);
            expect(parsed['paper1']).toBeUndefined();
        });
    });

    describe('Search Term Updates', () => {
        beforeEach(() => {
            State.setCurrentSearchTerm('initial search');
            State.setPaperPriority('paper1', 5);
            State.setPaperPriority('paper2', 4);
            State.setPaperPriority('paper3', 3);
        });

        it('should update search term for a single paper', () => {
            State.updatePaperSearchTerm('paper1', 'new search');
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('new search');
            expect(priorities['paper2'].searchTerm).toBe('initial search');
            expect(priorities['paper3'].searchTerm).toBe('initial search');
        });

        it('should not update if paper not found', () => {
            State.updatePaperSearchTerm('non-existent', 'new search');
            // Should not throw, just do nothing
            expect(true).toBe(true);
        });

        it('should persist update to localStorage', () => {
            State.updatePaperSearchTerm('paper1', 'updated query');
            
            const stored = localStorage.getItem('paperPriorities');
            const parsed = JSON.parse(stored);
            expect(parsed['paper1'].searchTerm).toBe('updated query');
        });

        it('should update search term for multiple papers', () => {
            const count = State.updateSearchTermForMultiplePapers('initial search', 'updated search');
            
            expect(count).toBe(3);
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('updated search');
            expect(priorities['paper2'].searchTerm).toBe('updated search');
            expect(priorities['paper3'].searchTerm).toBe('updated search');
        });

        it('should update only matching papers', () => {
            State.setCurrentSearchTerm('different search');
            State.setPaperPriority('paper4', 2);
            
            const count = State.updateSearchTermForMultiplePapers('initial search', 'new search');
            
            expect(count).toBe(3);
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('new search');
            expect(priorities['paper4'].searchTerm).toBe('different search');
        });

        it('should return 0 when no papers match', () => {
            const count = State.updateSearchTermForMultiplePapers('non-existent search', 'new');
            expect(count).toBe(0);
        });

        it('should persist bulk update to localStorage', () => {
            State.updateSearchTermForMultiplePapers('initial search', 'bulk updated');
            
            const stored = localStorage.getItem('paperPriorities');
            const parsed = JSON.parse(stored);
            expect(parsed['paper1'].searchTerm).toBe('bulk updated');
            expect(parsed['paper2'].searchTerm).toBe('bulk updated');
        });

        it('should handle empty search terms', () => {
            State.updateSearchTermForMultiplePapers('initial search', '');
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('');
        });
    });

    describe('Edge Cases', () => {
        it('should handle rapid priority changes', () => {
            State.setCurrentSearchTerm('test');
            
            for (let i = 0; i < 100; i++) {
                State.setPaperPriority('rapid', (i % 5) + 1);
            }
            
            // Should end with priority 1 (100 % 5 + 1)
            expect(State.getPaperPriority('rapid')).toBe(1);
        });

        it('should handle many papers', () => {
            State.setCurrentSearchTerm('bulk test');
            
            for (let i = 0; i < 100; i++) {
                State.setPaperPriority(`paper${i}`, (i % 5) + 1);
            }
            
            const ids = State.getPaperIdsWithPriorities();
            expect(ids.length).toBe(100);
        });

        it('should handle special characters in paper IDs', () => {
            State.setCurrentSearchTerm('test');
            const specialId = 'paper-with_special.chars@123';
            
            State.setPaperPriority(specialId, 4);
            expect(State.getPaperPriority(specialId)).toBe(4);
        });

        it('should handle special characters in search terms', () => {
            const specialTerm = 'search with "quotes" & symbols <>';
            State.setCurrentSearchTerm(specialTerm);
            State.setPaperPriority('paper1', 3);
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe(specialTerm);
        });

        it('should handle very long search terms', () => {
            const longTerm = 'a'.repeat(1000);
            State.setCurrentSearchTerm(longTerm);
            State.setPaperPriority('paper1', 5);
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe(longTerm);
        });

        it('should maintain data integrity after many operations', () => {
            State.setCurrentSearchTerm('query1');
            State.setPaperPriority('p1', 5);
            
            State.setCurrentSearchTerm('query2');
            State.setPaperPriority('p2', 4);
            
            State.updatePaperSearchTerm('p1', 'updated');
            State.setPaperPriority('p2', 3);
            State.updateSearchTermForMultiplePapers('query2', 'bulk');
            
            State.loadPriorities();
            
            const priorities = State.getAllPaperPriorities();
            expect(priorities['p1'].priority).toBe(5);
            expect(priorities['p1'].searchTerm).toBe('updated');
            expect(priorities['p2'].priority).toBe(3);
        });
    });
});
