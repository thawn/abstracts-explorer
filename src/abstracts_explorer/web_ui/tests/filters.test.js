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
    updateYearsForConference,
    dismissConferenceError
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
            </select>
            <select id="conference-selector">
            </select>
            <div id="settings-modal" class="hidden" data-context=""></div>
            <div id="modal-title"></div>
            <div id="search-settings-section"></div>
            <div id="chat-settings-section"></div>
            <select id="modal-session-filter" multiple></select>
            <div id="conference-error-banner" class="hidden"></div>
            <p id="conference-error-message"></p>
            <div id="conference-error-available"></div>
        `;
        document.body.style.overflow = '';
        // Clear URL conference globals
        delete window.urlConference;
        delete window.urlConferenceError;
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

        it('should populate conferences and years from conference_years', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS', 'ICLR'],
                    years: [2024, 2025],
                    conference_years: { 'NeurIPS': [2025], 'ICLR': [2024] },
                    default_conference: 'NeurIPS',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            expect(window.conferenceYearsMap).toEqual({ 'NeurIPS': [2025], 'ICLR': [2024] });
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

        it('should use conference_years for the selected conference', () => {
            updateYearsForConference();

            const yearSelect = document.getElementById('year-selector');
            const yearValues = Array.from(yearSelect.options).map(o => o.value);
            expect(yearValues).toContain('2023');
            expect(yearValues).toContain('2024');
            expect(yearValues).toContain('2025');
        });

        it('should show empty years when conference has no entry in conference_years', () => {
            window.conferenceYearsMap = {};

            updateYearsForConference();

            const yearSelect = document.getElementById('year-selector');
            const yearValues = Array.from(yearSelect.options).map(o => o.value);
            expect(yearValues).toEqual([]);
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

    describe('URL conference override', () => {
        it('should select the URL-specified conference over the API default', async () => {
            const mockLoadStats = jest.fn();
            window.loadStats = mockLoadStats;
            window.urlConference = 'ICLR';

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
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            const conferenceSelect = document.getElementById('conference-selector');
            expect(conferenceSelect.value).toBe('ICLR');
            expect(mockLoadStats).toHaveBeenCalled();

            delete window.loadStats;
        });

        it('should clear window.urlConference after applying it', async () => {
            window.urlConference = 'NeurIPS';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS'],
                    years: [2025],
                    conference_years: { 'NeurIPS': [2025] },
                    default_conference: 'NeurIPS',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            expect(window.urlConference).toBeUndefined();
        });

        it('should fall back to default when URL conference is not in options', async () => {
            window.urlConference = 'UnknownConf';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS'],
                    years: [2025],
                    conference_years: { 'NeurIPS': [2025] },
                    default_conference: 'NeurIPS',
                    default_year: 2025
                })
            });

            await loadFilterOptions();

            const conferenceSelect = document.getElementById('conference-selector');
            // URL conference not found in options, so API default should be used
            expect(conferenceSelect.value).toBe('NeurIPS');
        });
    });

    describe('conference error banner', () => {
        it('should show error banner when urlConferenceError is set', async () => {
            window.urlConferenceError = {
                message: "Conference 'xyz' not found.",
                available_conferences: ['NeurIPS', 'ICLR']
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: ['NeurIPS'],
                    years: [2025],
                    conference_years: { 'NeurIPS': [2025] }
                })
            });

            await loadFilterOptions();

            const banner = document.getElementById('conference-error-banner');
            expect(banner.classList.contains('hidden')).toBe(false);

            const message = document.getElementById('conference-error-message');
            expect(message.textContent).toContain('xyz');

            const available = document.getElementById('conference-error-available');
            expect(available.innerHTML).toContain('NeurIPS');
            expect(available.innerHTML).toContain('ICLR');
        });

        it('should clear urlConferenceError after showing', async () => {
            window.urlConferenceError = {
                message: 'Test error',
                available_conferences: []
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [],
                    conference_years: {}
                })
            });

            await loadFilterOptions();

            expect(window.urlConferenceError).toBeUndefined();
        });
    });

    describe('dismissConferenceError', () => {
        it('should hide the conference error banner', () => {
            const banner = document.getElementById('conference-error-banner');
            banner.classList.remove('hidden');

            dismissConferenceError();

            expect(banner.classList.contains('hidden')).toBe(true);
        });

        it('should do nothing when banner does not exist', () => {
            document.getElementById('conference-error-banner').remove();

            // Should not throw
            expect(() => dismissConferenceError()).not.toThrow();
        });
    });

    describe('distance threshold localStorage persistence', () => {
        beforeEach(() => {
            // Add distance threshold input to DOM
            document.body.innerHTML += `<input id="distance-threshold-input" type="number" value="1.2" />`;
            localStorage.clear();
        });

        it('should set distance threshold from localStorage if stored', async () => {
            localStorage.setItem('searchDistanceThreshold', '0.9');

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [],
                    conference_years: {},
                    default_distance_threshold: 1.2
                })
            });

            await loadFilterOptions();

            const dtInput = document.getElementById('distance-threshold-input');
            expect(dtInput.value).toBe('0.9');
        });

        it('should set distance threshold from API default when localStorage is empty', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ sessions: [] })
            }).mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    conferences: [],
                    years: [],
                    conference_years: {},
                    default_distance_threshold: 1.2
                })
            });

            await loadFilterOptions();

            const dtInput = document.getElementById('distance-threshold-input');
            expect(dtInput.value).toBe('1.2');
        });
    });
});
