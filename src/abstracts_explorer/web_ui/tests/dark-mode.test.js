/**
 * Tests for dark mode / light mode switching in Plotly charts.
 *
 * Verifies that the matchMedia 'change' listeners registered by clustering.js
 * and chat.js correctly call Plotly.relayout with updated font colours when
 * the OS colour scheme changes while the page is open.
 */

import { jest } from '@jest/globals';

// ---------------------------------------------------------------------------
// matchMedia mock
// Must be set up BEFORE any module imports so the listener captured at
// module-evaluation time is stored here.
// ---------------------------------------------------------------------------
const _mediaListeners = {};   // query -> { change: [fn, ...] }
const _mediaState = { dark: false };

function _createMQL(query) {
    return {
        get matches() {
            return query === '(prefers-color-scheme: dark)' ? _mediaState.dark : false;
        },
        addEventListener(type, listener) {
            if (!_mediaListeners[query]) _mediaListeners[query] = {};
            if (!_mediaListeners[query][type]) _mediaListeners[query][type] = [];
            _mediaListeners[query][type].push(listener);
        },
        removeEventListener(type, listener) {
            const list = _mediaListeners[query]?.[type];
            if (list) {
                const idx = list.indexOf(listener);
                if (idx >= 0) list.splice(idx, 1);
            }
        }
    };
}

window.matchMedia = jest.fn(_createMQL);

/**
 * Simulate an OS colour-scheme change.
 * Sets the internal `matches` state and fires all registered 'change' listeners
 * for `(prefers-color-scheme: dark)`.
 * @param {boolean} dark - true = dark mode, false = light mode
 */
function fireColorSchemeChange(dark) {
    _mediaState.dark = dark;
    const listeners = _mediaListeners['(prefers-color-scheme: dark)']?.change ?? [];
    listeners.forEach((fn) => fn({ matches: dark }));
}

// ---------------------------------------------------------------------------
// Global Plotly mock (must exist before module imports)
// ---------------------------------------------------------------------------
global.Plotly = {
    newPlot: jest.fn(() => Promise.resolve()),
    relayout: jest.fn(() => Promise.resolve()),
    restyle: jest.fn(() => Promise.resolve()),
    react: jest.fn(() => Promise.resolve())
};

global.marked = { parse: jest.fn((t) => t), use: jest.fn() };
global.alert = jest.fn();

// ---------------------------------------------------------------------------
// Mock modules required by clustering.js
// ---------------------------------------------------------------------------
jest.unstable_mockModule('../static/modules/utils/constants.js', () => ({
    API_BASE: '',
    PLOTLY_COLORS: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}));

jest.unstable_mockModule('../static/modules/utils/ui-utils.js', () => ({
    showLoading: jest.fn(),
    showErrorInElement: jest.fn(),
    renderEmptyState: jest.fn()
}));

jest.unstable_mockModule('../static/modules/utils/sort-utils.js', () => ({
    sortClustersBySizeDesc: jest.fn((e) => e)
}));

jest.unstable_mockModule('../static/modules/utils/cluster-utils.js', () => ({
    getClusterLabelWithCount: jest.fn((id, labels, n) => `Cluster ${id} (${n})`)
}));

jest.unstable_mockModule('../static/modules/utils/dom-utils.js', () => ({
    getSelectedConference: jest.fn(() => ''),
    getSelectedYears: jest.fn(() => []),
    escapeHtml: jest.fn((s) => s)
}));

jest.unstable_mockModule('../static/modules/paper-card.js', () => ({
    formatPaperCard: jest.fn((p) => `<div>${p.title}</div>`)
}));

jest.unstable_mockModule('../static/modules/utils/markdown-utils.js', () => ({
    renderInlineMarkdownWithLatex: jest.fn((s) => s)
}));

jest.unstable_mockModule('../static/modules/state.js', () => ({
    setCurrentSearchTerm: jest.fn(),
    getState: jest.fn(() => ({})),
    setState: jest.fn()
}));

// ---------------------------------------------------------------------------
// Import modules — the matchMedia mock is already in place so each module's
// top-level `addEventListener('change', ...)` call is captured above.
// ---------------------------------------------------------------------------
const { resetClusters } = await import('../static/modules/clustering.js');
const { _resetChatState } = await import('../static/modules/chat.js');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const DARK_COLOR = '#e5e7eb';   // gray-200
const LIGHT_COLOR = '#374151';  // gray-700

// ============================================================================
// Tests
// ============================================================================

describe('Dark mode / light mode colour-scheme switching', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        resetClusters();
        _resetChatState();
        // Reset to light mode by default
        _mediaState.dark = false;
    });

    // -------------------------------------------------------------------------
    // clustering.js — _refreshClusteringPlotColors
    // -------------------------------------------------------------------------
    describe('clustering.js — Plotly charts inside #clusters-tab', () => {
        it('updates font colour to dark when switching to dark mode', () => {
            document.body.innerHTML = `
                <div id="clusters-tab">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(1);
            expect(global.Plotly.relayout).toHaveBeenCalledWith(plotEl, { 'font.color': DARK_COLOR });
        });

        it('updates font colour to light when switching to light mode', () => {
            _mediaState.dark = true;   // start in dark mode
            document.body.innerHTML = `
                <div id="clusters-tab">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(false);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(1);
            expect(global.Plotly.relayout).toHaveBeenCalledWith(plotEl, { 'font.color': LIGHT_COLOR });
        });

        it('updates all plots when multiple charts exist', () => {
            document.body.innerHTML = `
                <div id="clusters-tab">
                    <div class="js-plotly-plot" id="p1"></div>
                    <div class="js-plotly-plot" id="p2"></div>
                    <div class="js-plotly-plot" id="p3"></div>
                </div>
            `;

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(3);
            ['p1', 'p2', 'p3'].forEach((id) => {
                expect(global.Plotly.relayout).toHaveBeenCalledWith(
                    document.getElementById(id),
                    { 'font.color': DARK_COLOR }
                );
            });
        });

        it('does not call Plotly.relayout when #clusters-tab has no plots', () => {
            document.body.innerHTML = `<div id="clusters-tab"></div>`;

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).not.toHaveBeenCalled();
        });

        it('does not call Plotly.relayout when #clusters-tab is absent', () => {
            document.body.innerHTML = '';

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).not.toHaveBeenCalled();
        });

        it('refreshes correctly on repeated switches dark → light → dark', () => {
            document.body.innerHTML = `
                <div id="clusters-tab">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(true);
            fireColorSchemeChange(false);
            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(3);
            // The last call should use the dark colour
            expect(global.Plotly.relayout).toHaveBeenNthCalledWith(3, plotEl, { 'font.color': DARK_COLOR });
        });
    });

    // -------------------------------------------------------------------------
    // chat.js — _refreshChatPlotColors
    // -------------------------------------------------------------------------
    describe('chat.js — Plotly charts inside #chat-messages', () => {
        it('updates font colour to dark when switching to dark mode', () => {
            document.body.innerHTML = `
                <div id="chat-messages">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(1);
            expect(global.Plotly.relayout).toHaveBeenCalledWith(plotEl, { 'font.color': DARK_COLOR });
        });

        it('updates font colour to light when switching to light mode', () => {
            _mediaState.dark = true;   // start in dark mode
            document.body.innerHTML = `
                <div id="chat-messages">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(false);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(1);
            expect(global.Plotly.relayout).toHaveBeenCalledWith(plotEl, { 'font.color': LIGHT_COLOR });
        });

        it('updates all chat plots when multiple charts are present', () => {
            document.body.innerHTML = `
                <div id="chat-messages">
                    <div class="js-plotly-plot" id="c1"></div>
                    <div class="js-plotly-plot" id="c2"></div>
                </div>
            `;

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(2);
        });

        it('does not call Plotly.relayout when #chat-messages has no plots', () => {
            document.body.innerHTML = `<div id="chat-messages"></div>`;

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).not.toHaveBeenCalled();
        });

        it('does not call Plotly.relayout when #chat-messages is absent', () => {
            document.body.innerHTML = '';

            fireColorSchemeChange(true);

            expect(global.Plotly.relayout).not.toHaveBeenCalled();
        });

        it('refreshes correctly on repeated switches light → dark → light', () => {
            document.body.innerHTML = `
                <div id="chat-messages">
                    <div class="js-plotly-plot"></div>
                </div>
            `;
            const plotEl = document.querySelector('.js-plotly-plot');

            fireColorSchemeChange(true);
            fireColorSchemeChange(false);

            expect(global.Plotly.relayout).toHaveBeenCalledTimes(2);
            expect(global.Plotly.relayout).toHaveBeenNthCalledWith(2, plotEl, { 'font.color': LIGHT_COLOR });
        });
    });

    // -------------------------------------------------------------------------
    // Both modules respond to the same event
    // -------------------------------------------------------------------------
    describe('simultaneous refresh of both clustering and chat plots', () => {
        it('relayouts both #clusters-tab and #chat-messages plots on one change event', () => {
            document.body.innerHTML = `
                <div id="clusters-tab">
                    <div class="js-plotly-plot" id="stat-plot"></div>
                </div>
                <div id="chat-messages">
                    <div class="js-plotly-plot" id="chat-plot"></div>
                </div>
            `;

            fireColorSchemeChange(true);

            // One call per module, one element each
            expect(global.Plotly.relayout).toHaveBeenCalledTimes(2);
            expect(global.Plotly.relayout).toHaveBeenCalledWith(
                document.getElementById('stat-plot'), { 'font.color': DARK_COLOR }
            );
            expect(global.Plotly.relayout).toHaveBeenCalledWith(
                document.getElementById('chat-plot'), { 'font.color': DARK_COLOR }
            );
        });
    });
});
