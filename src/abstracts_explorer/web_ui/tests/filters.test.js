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
    syncFiltersFromModal,
    handleYearChange,
    handleConferenceChange,
    updateYearsForConference
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
            <select id="year-selector" multiple>
                <option value="">All Years</option>
            </select>
            <select id="conference-selector">
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

        it('should call window.loadStats when a default conference is applied', async () => {
            const mockLoadStats = jest.fn();
            window.loadStats = mockLoadStats;

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS', 'ICLR'],
                    years: [2024, 2025],
                    conference_years: { 'NeurIPS': [2024, 2025], 'ICLR': [2024] },
                    default_conference: 'NeurIPS',
                    default_year: null
                })
            });

            await loadFilterOptions();

            const conferenceSelect = document.getElementById('conference-selector');
            expect(conferenceSelect.value).toBe('NeurIPS');
            expect(mockLoadStats).toHaveBeenCalled();

            delete window.loadStats;
        });

        it('should call window.loadStats when a default year is applied', async () => {
            const mockLoadStats = jest.fn();
            window.loadStats = mockLoadStats;

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [2024, 2025],
                    conference_years: {},
                    default_conference: '',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            const yearSelect = document.getElementById('year-selector');
            expect(yearSelect.value).toBe('2025');
            expect(mockLoadStats).toHaveBeenCalled();

            delete window.loadStats;
        });

        it('should NOT override "All Years" selection with default_year when user explicitly selected All Years', async () => {
            // Simulate: dropdown already populated (more than 1 option), user selected "All Years"
            const yearSelect = document.getElementById('year-selector');
            const opt2024 = document.createElement('option');
            opt2024.value = '2024'; opt2024.textContent = '2024';
            const opt2025 = document.createElement('option');
            opt2025.value = '2025'; opt2025.textContent = '2025';
            yearSelect.appendChild(opt2024);
            yearSelect.appendChild(opt2025);
            yearSelect.value = '';  // User selected "All Years"

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [2024, 2025],
                    conference_years: {},
                    default_conference: '',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            // The year should remain "All Years" (empty string), not jump to 2025
            expect(yearSelect.value).toBe('');
        });

        it('should select the first conference and call window.loadStats even when no default_conference is configured', async () => {
            const mockLoadStats = jest.fn();
            window.loadStats = mockLoadStats;

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS'],
                    years: [2024],
                    conference_years: { 'NeurIPS': [2024] },
                    default_conference: '',
                    default_year: null
                })
            });

            await loadFilterOptions();

            // Without "All Conferences", a conference is always selected on initial load
            expect(mockLoadStats).toHaveBeenCalled();

            delete window.loadStats;
        });

        it('should store db_conference_years in window.dbConferenceYearsMap', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS', 'ICLR'],
                    years: [2024, 2025],
                    conference_years: { 'NeurIPS': [2024, 2025], 'ICLR': [2024] },
                    db_conference_years: { 'NeurIPS': [2025], 'ICLR': [2024] },
                    default_conference: 'NeurIPS',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            expect(window.dbConferenceYearsMap).toEqual({ 'NeurIPS': [2025], 'ICLR': [2024] });
        });
    });

    describe('updateYearsForConference', () => {
        beforeEach(() => {
            window.conferenceYearsMap = { 'NeurIPS': [2023, 2024, 2025], 'ICLR': [2024] };
            window.allYears = [2023, 2024, 2025];

            const conferenceSelect = document.getElementById('conference-selector');
            const option = document.createElement('option');
            option.value = 'NeurIPS';
            option.textContent = 'NeurIPS';
            conferenceSelect.appendChild(option);
            conferenceSelect.value = 'NeurIPS';
        });

        it('should use db_conference_years years when available for the selected conference', () => {
            window.dbConferenceYearsMap = { 'NeurIPS': [2025] };

            updateYearsForConference();

            const yearSelect = document.getElementById('year-selector');
            const yearValues = Array.from(yearSelect.options).map(o => o.value);
            // db_conference_years only has 2025 for NeurIPS; plugin has 2023, 2024, 2025
            expect(yearValues).toContain('2025');
            expect(yearValues).not.toContain('2023');
            expect(yearValues).not.toContain('2024');
        });

        it('should fall back to conference_years when db_conference_years has no entry for the selected conference', () => {
            window.dbConferenceYearsMap = {};  // no DB data for NeurIPS

            updateYearsForConference();

            const yearSelect = document.getElementById('year-selector');
            const yearValues = Array.from(yearSelect.options).map(o => o.value);
            // falls back to plugin-based years: 2023, 2024, 2025
            expect(yearValues).toContain('2025');
            expect(yearValues).toContain('2024');
            expect(yearValues).toContain('2023');
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

    describe('handleYearChange', () => {
        beforeEach(() => {
            document.body.innerHTML = `
                <select id="year-selector" multiple>
                    <option value="">All Years</option>
                </select>
                <select id="conference-selector"></select>
                <select id="session-filter" multiple></select>
                <div id="search-results">old results</div>
            `;
            window.loadStats = jest.fn();
            window.loadInterestingPapers = jest.fn();
            window.updateInterestingPapersCount = jest.fn();
            window.resetClusters = jest.fn();
            window.loadClusters = jest.fn();
            global.fetch.mockResolvedValue({
                ok: true,
                json: async () => ({ sessions: [], topics: [] })
            });
        });

        afterEach(() => {
            delete window.loadStats;
            delete window.loadInterestingPapers;
            delete window.updateInterestingPapersCount;
            delete window.resetClusters;
            delete window.loadClusters;
        });

        it('should call resetClusters but not loadClusters when not on clusters tab', () => {
            document.body.innerHTML += '<button id="tab-search" class="tab-btn border-purple-600"></button>';
            handleYearChange();
            expect(window.resetClusters).toHaveBeenCalled();
            expect(window.loadClusters).not.toHaveBeenCalled();
        });

        it('should call resetClusters and loadClusters when on clusters tab', () => {
            document.body.innerHTML += '<button id="tab-clusters" class="tab-btn border-purple-600"></button>';
            handleYearChange();
            expect(window.resetClusters).toHaveBeenCalled();
            expect(window.loadClusters).toHaveBeenCalled();
        });
    });

    describe('handleConferenceChange', () => {
        beforeEach(() => {
            document.body.innerHTML = `
                <select id="year-selector" multiple>
                    <option value="">All Years</option>
                </select>
                <select id="conference-selector"></select>
                <select id="session-filter" multiple></select>
                <div id="search-results">old results</div>
            `;
            window.loadStats = jest.fn();
            window.loadInterestingPapers = jest.fn();
            window.updateInterestingPapersCount = jest.fn();
            window.resetClusters = jest.fn();
            window.loadClusters = jest.fn();
            global.fetch.mockResolvedValue({
                ok: true,
                json: async () => ({ sessions: [], topics: [] })
            });
        });

        afterEach(() => {
            delete window.loadStats;
            delete window.loadInterestingPapers;
            delete window.updateInterestingPapersCount;
            delete window.resetClusters;
            delete window.loadClusters;
        });

        it('should call resetClusters but not loadClusters when not on clusters tab', () => {
            document.body.innerHTML += '<button id="tab-search" class="tab-btn border-purple-600"></button>';
            handleConferenceChange();
            expect(window.resetClusters).toHaveBeenCalled();
            expect(window.loadClusters).not.toHaveBeenCalled();
        });

        it('should call resetClusters and loadClusters when on clusters tab', () => {
            document.body.innerHTML += '<button id="tab-clusters" class="tab-btn border-purple-600"></button>';
            handleConferenceChange();
            expect(window.resetClusters).toHaveBeenCalled();
            expect(window.loadClusters).toHaveBeenCalled();
        });
    });
});
