/**
 * Tests for filters module
 */

import { jest } from '@jest/globals';

global.fetch = jest.fn();

import { 
    loadFilterOptions, 
    selectAllFilter, 
    deselectAllFilter,
    openSearchSettings,
    openChatSettings,
    closeSettings,
    syncFiltersToModal,
    syncFiltersFromModal
} from '../static/modules/filters.js';

describe('Filters Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        document.body.innerHTML = `
            <select id="session-filter" multiple>
                <option value="Session 1">Session 1</option>
                <option value="Session 2">Session 2</option>
            </select>
            <select id="chat-session-filter" multiple>
                <option value="Session 1">Session 1</option>
                <option value="Session 2">Session 2</option>
            </select>
            <select id="year-selector">
                <option value="">All Years</option>
            </select>
            <select id="conference-selector">
                <option value="">All Conferences</option>
            </select>
            <div id="settings-modal" class="hidden" data-context=""></div>
            <div id="modal-title"></div>
            <div id="search-settings-section"></div>
            <div id="chat-settings-section"></div>
            <select id="modal-session-filter" multiple></select>
        `;
        document.body.style.overflow = '';
    });

    describe('loadFilterOptions', () => {
        it('should load filter options from API', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    sessions: ['Session 1', 'Session 2', 'Session 3']
                })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS', 'ICLR'],
                    years: [2024, 2025],
                    conference_years: {
                        'NeurIPS': [2024, 2025],
                        'ICLR': [2024]
                    }
                })
            });

            await loadFilterOptions();

            expect(global.fetch).toHaveBeenCalledTimes(2);
            const sessionSelect = document.getElementById('session-filter');
            expect(sessionSelect.options.length).toBe(3);
        });

        it('should populate year selector', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [2024, 2025]
                })
            });

            await loadFilterOptions();

            const yearSelect = document.getElementById('year-selector');
            expect(yearSelect.options.length).toBeGreaterThan(1);
        });
    });

    describe('selectAllFilter', () => {
        it('should select all options', () => {
            const select = document.getElementById('session-filter');
            Array.from(select.options).forEach(opt => opt.selected = false);

            selectAllFilter('session-filter');

            expect(Array.from(select.options).every(opt => opt.selected)).toBe(true);
        });
    });

    describe('deselectAllFilter', () => {
        it('should deselect all options', () => {
            const select = document.getElementById('session-filter');
            Array.from(select.options).forEach(opt => opt.selected = true);

            deselectAllFilter('session-filter');

            expect(Array.from(select.options).every(opt => !opt.selected)).toBe(true);
        });
    });

    describe('openSearchSettings', () => {
        it('should open modal with search context', () => {
            openSearchSettings();

            const modal = document.getElementById('settings-modal');
            expect(modal.classList.contains('hidden')).toBe(false);
            expect(modal.dataset.context).toBe('search');
            expect(document.body.style.overflow).toBe('hidden');
        });

        it('should show search-specific settings', () => {
            openSearchSettings();

            const searchSection = document.getElementById('search-settings-section');
            const chatSection = document.getElementById('chat-settings-section');
            expect(searchSection.classList.contains('hidden')).toBe(false);
            expect(chatSection.classList.contains('hidden')).toBe(true);
        });
    });

    describe('openChatSettings', () => {
        it('should open modal with chat context', () => {
            openChatSettings();

            const modal = document.getElementById('settings-modal');
            expect(modal.classList.contains('hidden')).toBe(false);
            expect(modal.dataset.context).toBe('chat');
        });

        it('should show chat-specific settings', () => {
            openChatSettings();

            const searchSection = document.getElementById('search-settings-section');
            const chatSection = document.getElementById('chat-settings-section');
            expect(searchSection.classList.contains('hidden')).toBe(true);
            expect(chatSection.classList.contains('hidden')).toBe(false);
        });
    });

    describe('closeSettings', () => {
        it('should close modal', () => {
            const modal = document.getElementById('settings-modal');
            modal.classList.remove('hidden');
            modal.dataset.context = 'search';

            closeSettings();

            expect(modal.classList.contains('hidden')).toBe(true);
            expect(document.body.style.overflow).toBe('');
        });
    });

    describe('syncFiltersToModal', () => {
        it('should sync search filters to modal', () => {
            const sessionSelect = document.getElementById('session-filter');
            const modalSelect = document.getElementById('modal-session-filter');
            
            // Setup source
            sessionSelect.options[0].selected = true;
            sessionSelect.options[1].selected = false;
            
            // Setup target with options
            modalSelect.innerHTML = sessionSelect.innerHTML;

            syncFiltersToModal('search');

            expect(modalSelect.options[0].selected).toBe(true);
            expect(modalSelect.options[1].selected).toBe(false);
        });
    });

    describe('syncFiltersFromModal', () => {
        it('should sync modal filters back to search', () => {
            const sessionSelect = document.getElementById('session-filter');
            const modalSelect = document.getElementById('modal-session-filter');
            
            // Setup both with options
            modalSelect.innerHTML = sessionSelect.innerHTML;
            
            // Change modal
            modalSelect.options[0].selected = false;
            modalSelect.options[1].selected = true;

            syncFiltersFromModal('search');

            expect(sessionSelect.options[0].selected).toBe(false);
            expect(sessionSelect.options[1].selected).toBe(true);
        });
    });
});
