/**
 * Tests for hierarchy-level slider behavior in the clustering module.
 */

import { jest } from '@jest/globals';

global.fetch = jest.fn();
global.Plotly = {
    newPlot: jest.fn(() => Promise.resolve()),
    relayout: jest.fn(() => Promise.resolve()),
    restyle: jest.fn(() => Promise.resolve()),
    react: jest.fn(() => Promise.resolve()),
    addTraces: jest.fn(() => Promise.resolve())
};
global.console = {
    ...console,
    log: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
};

jest.unstable_mockModule('../static/modules/utils/constants.js', () => ({
    API_BASE: '',
    PLOTLY_COLORS: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
}));

jest.unstable_mockModule('../static/modules/utils/ui-utils.js', () => ({
    showLoading: jest.fn(),
    showErrorInElement: jest.fn()
}));

jest.unstable_mockModule('../static/modules/paper-card.js', () => ({
    formatPaperCard: jest.fn((paper) => `<div>${paper.title}</div>`)
}));

const { loadClusters, setHierarchyLevel, resetClusters } = await import(
    '../static/modules/clustering.js'
);

function buildData(levels) {
    return {
        points: [
            { id: 'p1', x: 1, y: 1, cluster: 0, title: 'Paper 1', year: 2025, conference: 'NeurIPS' },
            { id: 'p2', x: 2, y: 2, cluster: 1, title: 'Paper 2', year: 2025, conference: 'NeurIPS' }
        ],
        statistics: { total_papers: 2, n_clusters: 2, n_noise: 0, cluster_sizes: { 0: 1, 1: 1 } },
        cluster_labels: { 0: 'Cat', 1: 'Dog' },
        cluster_centers: { 0: { x: 1, y: 1 }, 1: { x: 2, y: 2 } },
        hierarchy: { levels }
    };
}

const flatLevel = {
    level: 0,
    max_distance: 0.1,
    clusters: [
        { id: 0, label: 'Cat', size: 1, center: { x: 1, y: 1 }, keywords: [] },
        { id: 1, label: 'Dog', size: 1, center: { x: 2, y: 2 }, keywords: [] }
    ],
    mapping: { 0: 0, 1: 1 }
};

const mergedLevel = {
    level: 1,
    max_distance: 0.2,
    clusters: [{ id: 100, label: 'Animals', size: 2, center: { x: 1.5, y: 1.5 }, keywords: [] }],
    mapping: { 0: 100, 1: 100 }
};

describe('Hierarchy slider', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        resetClusters();
        document.body.innerHTML = `
            <div id="cluster-plot"></div>
            <div id="cluster-legend"></div>
            <div id="hierarchy-controls" class="hidden">
                <input id="hierarchy-level-slider" type="range" min="0" max="9" value="4" />
                <span id="hierarchy-level-label"></span>
            </div>
        `;
        const plotDiv = document.getElementById('cluster-plot');
        plotDiv.on = jest.fn();
        plotDiv.data = [];
    });

    it('hides controls when the cache has no hierarchy levels', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([]) });

        await loadClusters();

        expect(document.getElementById('hierarchy-controls').classList.contains('hidden')).toBe(true);
        const traces = global.Plotly.newPlot.mock.calls[0][1];
        expect(traces).toHaveLength(2);
        expect(traces[1].x).toEqual([1, 2]);
    });

    it('clamps the default level when fewer than four levels exist', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();

        const slider = document.getElementById('hierarchy-level-slider');
        expect(slider.max).toBe('1');
        expect(slider.value).toBe('1');
        expect(document.getElementById('hierarchy-level-label').textContent).toContain('Level 1');
    });

    it('merges document colors and stars at a merged level', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();

        const mergedTraces = global.Plotly.newPlot.mock.calls[0][1];
        expect(mergedTraces[0].marker.color[0]).toBe(mergedTraces[0].marker.color[1]);
        expect(mergedTraces[1].text).toEqual(['Animals']);
        expect(mergedTraces[1].x).toEqual([1.5]);

        global.Plotly.newPlot.mockClear();
        setHierarchyLevel(0);

        const flatTraces = global.Plotly.newPlot.mock.calls[0][1];
        expect(flatTraces[1].text).toEqual(['Cat', 'Dog']);
        expect(flatTraces[1].x).toEqual([1, 2]);
    });

    it('clamps invalid slider indices', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();
        setHierarchyLevel(99);

        expect(document.getElementById('hierarchy-level-slider').value).toBe('1');
        setHierarchyLevel(-5);
        expect(document.getElementById('hierarchy-level-slider').value).toBe('0');
    });

    it('hides cluster points when toggled off and restores them when toggled on', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();
        setHierarchyLevel(0);
        await flushPromises();
        document.getElementById('cluster-plot').data = [
            { x: [1, 2], y: [1, 2] },
            { x: [1.5], y: [1.5] }
        ];
        global.Plotly.restyle.mockClear();

        const items = document.querySelectorAll('#cluster-legend [data-cluster-id]');
        expect(items).toHaveLength(2);

        items[0].click();

        const hiddenCall = getDocumentRestyleCall();
        expect(hiddenCall[1].x[0]).toEqual([2]);
        expect(hiddenCall[1].y[0]).toEqual([2]);
        expect(hiddenCall[1].text[0]).toEqual(['Paper 2']);
        expect(hiddenCall[1].customdata[0]).toHaveLength(1);
        expect(hiddenCall[1].customdata[0][0].id).toBe('p2');
        expect(hiddenCall[1]['marker.color'][0]).toEqual(['#ff7f0e']);
        expect(hiddenCall[1]['marker.color'][0]).not.toContain('#e5e7eb');

        global.Plotly.restyle.mockClear();
        items[0].click();

        const restoredCall = getDocumentRestyleCall();
        expect(restoredCall[1].x[0]).toEqual([1, 2]);
        expect(restoredCall[1].text[0]).toEqual(['Paper 1', 'Paper 2']);
        expect(restoredCall[1]['marker.color'][0]).toEqual(['#1f77b4', '#ff7f0e']);
    });

    it('keeps noise points visible until all clusters are cleared', async () => {
        const data = buildData([flatLevel, mergedLevel]);
        data.points.push({
            id: 'noise-1',
            x: 3,
            y: 3,
            cluster: -1,
            title: 'Noise paper',
            year: 2025,
            conference: 'NeurIPS'
        });
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => data });

        await loadClusters();
        setHierarchyLevel(0);
        await flushPromises();
        document.getElementById('cluster-plot').data = [
            { x: [1, 2, 3], y: [1, 2, 3] },
            { x: [1.5], y: [1.5] }
        ];
        global.Plotly.restyle.mockClear();

        const items = document.querySelectorAll('#cluster-legend [data-cluster-id]');
        items[0].click();

        const hiddenCall = getDocumentRestyleCall();
        expect(hiddenCall[1].customdata[0].map(point => point.id)).toEqual(['p2', 'noise-1']);

        const noneButton = Array.from(document.querySelectorAll('#cluster-legend button'))
            .find(button => button.textContent === 'None');
        expect(noneButton).toBeDefined();
        global.Plotly.restyle.mockClear();
        noneButton.click();

        const clearedCall = getDocumentRestyleCall();
        expect(clearedCall[1].x[0]).toEqual([]);
        expect(clearedCall[1].customdata[0]).toEqual([]);
    });
});

async function flushPromises() {
    await Promise.resolve();
    await Promise.resolve();
}

function getDocumentRestyleCall() {
    return global.Plotly.restyle.mock.calls.find(call => (
        Array.isArray(call[2])
        && call[2][0] === 0
        && call[1]
        && Array.isArray(call[1].x)
    ));
}
