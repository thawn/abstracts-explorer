/**
 * Tests for tabs module
 */

import { jest } from '@jest/globals';

global.fetch = jest.fn();

import { switchTab, loadStats, checkEmbeddingModelCompatibility, dismissEmbeddingWarning } from '../static/modules/tabs.js';
import * as State from '../static/modules/state.js';

describe('Tabs Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        document.body.innerHTML = `
            <button id="tab-search" class="tab-btn border-purple-600 text-gray-700"></button>
            <button id="tab-chat" class="tab-btn border-transparent text-gray-500"></button>
            <button id="tab-interesting" class="tab-btn border-transparent text-gray-500"></button>
            <div id="search-tab" class="tab-content"></div>
            <div id="chat-tab" class="tab-content hidden"></div>
            <div id="interesting-tab" class="tab-content hidden"></div>
            <div id="stats"></div>
            <select id="year-selector"><option value="">All</option></select>
            <select id="conference-selector"><option value="">All</option></select>
            <div id="embedding-warning-banner" class="hidden"></div>
            <div id="embedding-warning-message"></div>
            <div id="warning-current-model"></div>
            <div id="warning-stored-model"></div>
        `;
    });

    describe('switchTab', () => {
        it('should switch to chat tab', () => {
            switchTab('chat');

            expect(State.getCurrentTab()).toBe('chat');
            expect(document.getElementById('tab-chat').classList.contains('border-purple-600')).toBe(true);
            expect(document.getElementById('chat-tab').classList.contains('hidden')).toBe(false);
        });

        it('should hide previous tab content', () => {
            switchTab('chat');

            expect(document.getElementById('search-tab').classList.contains('hidden')).toBe(true);
        });

        it('should update tab button styles', () => {
            switchTab('interesting');

            const interestingBtn = document.getElementById('tab-interesting');
            expect(interestingBtn.classList.contains('border-purple-600')).toBe(true);
            expect(interestingBtn.classList.contains('text-gray-700')).toBe(true);
        });
    });

    describe('loadStats', () => {
        it('should fetch and display stats', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    total_papers: 1234,
                    conference: 'NeurIPS',
                    year: 2025
                })
            });

            await loadStats();

            expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/stats'));
            const stats = document.getElementById('stats');
            expect(stats.innerHTML).toContain('1,234');
        });

        it('should include filters in request', async () => {
            document.getElementById('year-selector').value = '2025';
            document.getElementById('conference-selector').value = 'NeurIPS';
            
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    total_papers: 500,
                    year: 2025,
                    conference: 'NeurIPS'
                })
            });

            await loadStats();

            // Check that fetch was called with the API endpoint
            expect(global.fetch).toHaveBeenCalled();
            const callUrl = global.fetch.mock.calls[0][0];
            expect(callUrl).toContain('/api/stats');
        });

        it('should handle error response', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    error: 'Failed to load stats'
                })
            });

            await loadStats();

            const stats = document.getElementById('stats');
            expect(stats.innerHTML).toContain('Failed to load stats');
        });
    });

    describe('checkEmbeddingModelCompatibility', () => {
        it('should check compatibility on load', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    compatible: true
                })
            });

            await checkEmbeddingModelCompatibility();

            expect(global.fetch).toHaveBeenCalledWith('/api/embedding-model-check');
        });

        it('should show warning banner when incompatible', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    compatible: false,
                    warning: 'Model mismatch',
                    current_model: 'model-a',
                    stored_model: 'model-b'
                })
            });

            await checkEmbeddingModelCompatibility();

            const banner = document.getElementById('embedding-warning-banner');
            expect(banner.classList.contains('hidden')).toBe(false);
            expect(document.getElementById('embedding-warning-message').textContent).toBe('Model mismatch');
        });

        it('should not show warning when compatible', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    compatible: true
                })
            });

            await checkEmbeddingModelCompatibility();

            const banner = document.getElementById('embedding-warning-banner');
            expect(banner.classList.contains('hidden')).toBe(true);
        });
    });

    describe('dismissEmbeddingWarning', () => {
        it('should hide warning banner', () => {
            const banner = document.getElementById('embedding-warning-banner');
            banner.classList.remove('hidden');

            dismissEmbeddingWarning();

            expect(banner.classList.contains('hidden')).toBe(true);
        });
    });
});
