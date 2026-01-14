/**
 * Unit tests for state management module
 */

import {
    currentTab,
    setCurrentTab,
    getCurrentTab,
    setCurrentSearchTerm,
    getCurrentSearchTerm,
    setCurrentInterestingSession,
    getCurrentInterestingSession,
    setInterestingPapersSortOrder,
    getInterestingPapersSortOrder,
    loadPriorities,
    savePriorities,
    setPaperPriority,
    getPaperPriority,
    getAllPaperPriorities,
    getPaperIdsWithPriorities,
    updatePaperSearchTerm,
    updateSearchTermForMultiplePapers
} from '../static/modules/state.js';

describe('State Management', () => {
    beforeEach(() => {
        // Clear localStorage before each test
        localStorage.clear();
        // Reset state by reloading priorities
        loadPriorities();
    });

    describe('Tab Management', () => {
        it('should set and get current tab', () => {
            setCurrentTab('chat');
            expect(getCurrentTab()).toBe('chat');
        });

        it('should start with search tab', () => {
            expect(getCurrentTab()).toBe('search');
        });
    });

    describe('Search Term Management', () => {
        it('should set and get current search term', () => {
            setCurrentSearchTerm('machine learning');
            expect(getCurrentSearchTerm()).toBe('machine learning');
        });
    });

    describe('Interesting Session Management', () => {
        it('should set and get current interesting session', () => {
            setCurrentInterestingSession('Session 1');
            expect(getCurrentInterestingSession()).toBe('Session 1');
        });

        it('should start with null session', () => {
            expect(getCurrentInterestingSession()).toBeNull();
        });
    });

    describe('Sort Order Management', () => {
        it('should set and get sort order', () => {
            setInterestingPapersSortOrder('rating-poster-search');
            expect(getInterestingPapersSortOrder()).toBe('rating-poster-search');
        });

        it('should persist sort order to localStorage', () => {
            setInterestingPapersSortOrder('poster-search-rating');
            expect(localStorage.getItem('interestingPapersSortOrder')).toBe('poster-search-rating');
        });

        it('should load sort order from localStorage', () => {
            localStorage.setItem('interestingPapersSortOrder', 'rating-poster-search');
            loadPriorities();
            expect(getInterestingPapersSortOrder()).toBe('rating-poster-search');
        });
    });

    describe('Paper Priorities', () => {
        beforeEach(() => {
            setCurrentSearchTerm('test search');
        });

        it('should set paper priority', () => {
            const result = setPaperPriority('paper1', 5);
            expect(result).toBe(true);
            expect(getPaperPriority('paper1')).toBe(5);
        });

        it('should remove priority when clicking same star', () => {
            setPaperPriority('paper1', 3);
            const result = setPaperPriority('paper1', 3);
            expect(result).toBe(false);
            expect(getPaperPriority('paper1')).toBe(0);
        });

        it('should remove priority when set to 0', () => {
            setPaperPriority('paper1', 4);
            const result = setPaperPriority('paper1', 0);
            expect(result).toBe(false);
            expect(getPaperPriority('paper1')).toBe(0);
        });

        it('should preserve search term when updating priority', () => {
            setCurrentSearchTerm('first search');
            setPaperPriority('paper1', 3);
            
            setCurrentSearchTerm('second search');
            setPaperPriority('paper1', 5);
            
            const priorities = getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('first search');
        });

        it('should use current search term for new priorities', () => {
            setCurrentSearchTerm('my search');
            setPaperPriority('paper1', 4);
            
            const priorities = getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('my search');
        });

        it('should return 0 for non-existent paper', () => {
            expect(getPaperPriority('non-existent')).toBe(0);
        });

        it('should get all paper priorities', () => {
            setPaperPriority('paper1', 5);
            setPaperPriority('paper2', 3);
            
            const priorities = getAllPaperPriorities();
            expect(Object.keys(priorities).length).toBe(2);
            expect(priorities['paper1'].priority).toBe(5);
            expect(priorities['paper2'].priority).toBe(3);
        });

        it('should get paper IDs with priorities', () => {
            setPaperPriority('paper1', 5);
            setPaperPriority('paper2', 3);
            
            const ids = getPaperIdsWithPriorities();
            expect(ids).toContain('paper1');
            expect(ids).toContain('paper2');
            expect(ids.length).toBe(2);
        });
    });

    describe('Priority Persistence', () => {
        it('should save priorities to localStorage', () => {
            setCurrentSearchTerm('test');
            setPaperPriority('paper1', 5);
            
            const stored = localStorage.getItem('paperPriorities');
            expect(stored).not.toBeNull();
            
            const parsed = JSON.parse(stored);
            expect(parsed['paper1'].priority).toBe(5);
        });

        it('should load priorities from localStorage', () => {
            const priorities = {
                paper1: { priority: 5, searchTerm: 'test' },
                paper2: { priority: 3, searchTerm: 'another test' }
            };
            localStorage.setItem('paperPriorities', JSON.stringify(priorities));
            
            loadPriorities();
            
            expect(getPaperPriority('paper1')).toBe(5);
            expect(getPaperPriority('paper2')).toBe(3);
        });

        it('should handle invalid JSON in localStorage', () => {
            localStorage.setItem('paperPriorities', 'invalid json');
            
            // Should not throw
            loadPriorities();
            
            expect(getPaperIdsWithPriorities().length).toBe(0);
        });
    });

    describe('Search Term Updates', () => {
        beforeEach(() => {
            setCurrentSearchTerm('initial search');
            setPaperPriority('paper1', 5);
            setPaperPriority('paper2', 4);
            setPaperPriority('paper3', 3);
        });

        it('should update search term for a single paper', () => {
            updatePaperSearchTerm('paper1', 'new search');
            
            const priorities = getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('new search');
            expect(priorities['paper2'].searchTerm).toBe('initial search');
        });

        it('should not update if paper not found', () => {
            updatePaperSearchTerm('non-existent', 'new search');
            // Should not throw, just do nothing
            expect(true).toBe(true);
        });

        it('should update search term for multiple papers', () => {
            const count = updateSearchTermForMultiplePapers('initial search', 'updated search');
            
            expect(count).toBe(3);
            const priorities = getAllPaperPriorities();
            expect(priorities['paper1'].searchTerm).toBe('updated search');
            expect(priorities['paper2'].searchTerm).toBe('updated search');
            expect(priorities['paper3'].searchTerm).toBe('updated search');
        });

        it('should return 0 when no papers match', () => {
            const count = updateSearchTermForMultiplePapers('non-existent', 'new');
            expect(count).toBe(0);
        });

        it('should persist after updating multiple papers', () => {
            updateSearchTermForMultiplePapers('initial search', 'updated search');
            
            const stored = localStorage.getItem('paperPriorities');
            const parsed = JSON.parse(stored);
            expect(parsed['paper1'].searchTerm).toBe('updated search');
        });
    });
});
