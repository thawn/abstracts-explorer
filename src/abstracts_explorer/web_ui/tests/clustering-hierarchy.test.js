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
});
