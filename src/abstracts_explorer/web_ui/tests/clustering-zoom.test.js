/**
 * Tests for zoom-driven hierarchy levels in the clustering module.
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
global.alert = jest.fn();
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
    formatCard: jest.fn(),
    formatPaperCard: jest.fn((paper) => `<div>${paper.title}</div>`)
}));

const { loadClusters, setHierarchyLevel, resetClusters, searchCustomCluster } = await import(
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
        cluster_descriptions: { 0: 'Papers about cats.', 1: 'Papers about dogs.' },
        cluster_centers: { 0: { x: 1, y: 1 }, 1: { x: 2, y: 2 } },
        hierarchy: { levels }
    };
}

function buildLevels(count) {
    return Array.from({ length: count }, (_, index) => {
        const clusterId = 100 + index;
        return {
            level: index,
            max_distance: 0.1 + index * 0.1,
            clusters: [
                {
                    id: clusterId,
                    label: `Level ${index} cluster`,
                    description: `Cluster at level ${index}.`,
                    size: 2,
                    center: { x: 1.5, y: 1.5 },
                    keywords: []
                }
            ],
            mapping: { 0: clusterId, 1: clusterId }
        };
    });
}

function installDom() {
    document.body.innerHTML = `
        <div id="cluster-plot"></div>
        <div id="cluster-legend"></div>
        <div id="hierarchy-controls" class="hidden">
            <input id="hierarchy-level-slider" type="range" min="0" max="9" value="4" />
            <span id="hierarchy-level-label"></span>
        </div>
        <div id="custom-query-input"></div>
        <div id="custom-query-distance"></div>
        <button id="search-custom-btn"></button>
    `;
    const plotDiv = document.getElementById('cluster-plot');
    plotDiv.on = jest.fn();
    plotDiv.removeAllListeners = jest.fn();
    plotDiv.data = [];
    return plotDiv;
}

async function loadWithLevels(plotDiv, levels) {
    plotDiv.layout = {
        xaxis: { range: [0, 100] },
        yaxis: { range: [0, 100] }
    };
    global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData(levels) });
    await loadClusters();
    await flushPromises();
}

function getRelayoutHandler(plotDiv) {
    const call = plotDiv.on.mock.calls
        .filter(([event]) => event === 'plotly_relayout')
        .pop();
    expect(call).toBeTruthy();
    return call[1];
}

function setRanges(plotDiv, xRange, yRange = xRange) {
    plotDiv.layout = {
        xaxis: { range: xRange },
        yaxis: { range: yRange }
    };
}

describe('Zoom-driven hierarchy levels', () => {
    let plotDiv;

    beforeEach(() => {
        jest.clearAllMocks();
        resetClusters();
        plotDiv = installDom();
    });

    it('starts at the coarsest level when fully zoomed out', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));

        const slider = document.getElementById('hierarchy-level-slider');
        expect(slider.value).toBe('9');
        expect(document.getElementById('hierarchy-level-label').textContent).toContain('Level 9');
    });

    it('enables scroll zoom for the hierarchy plot', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));

        const config = global.Plotly.newPlot.mock.calls[0][3];
        expect(config.scrollZoom).toBe(true);
    });

    it('enables scroll zoom for custom-topic plots', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                query: 'cats',
                distance: 0.5,
                papers: [{ id: 'p1', title: 'Paper 1', year: 2025, conference: 'NeurIPS' }],
                count: 1,
                query_embedding: [0.1, 0.2]
            })
        });

        document.getElementById('custom-query-input').value = 'cats';
        document.getElementById('custom-query-distance').value = '0.5';
        await searchCustomCluster();

        const config = global.Plotly.react.mock.calls.at(-1)[3];
        expect(config.scrollZoom).toBe(true);
    });

    it('zooms all the way in to the finest level at 10x', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        const handler = getRelayoutHandler(plotDiv);

        setRanges(plotDiv, [45, 55]);
        handler({ 'xaxis.range[0]': 45, 'xaxis.range[1]': 55 });

        expect(document.getElementById('hierarchy-level-slider').value).toBe('0');
    });

    it('maps intermediate zoom levels to intermediate hierarchy levels', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        const handler = getRelayoutHandler(plotDiv);

        // 3.25x zoom: t = (3.25 - 1) / 9 = 0.25 -> level round(9 * 0.75) = 7
        setRanges(plotDiv, [34.6153846, 65.3846154]);
        handler({});

        expect(document.getElementById('hierarchy-level-slider').value).toBe('7');
    });

    it('returns to the coarsest level when zooming back out', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        const handler = getRelayoutHandler(plotDiv);

        setRanges(plotDiv, [45, 55]);
        handler({});
        expect(document.getElementById('hierarchy-level-slider').value).toBe('0');
        await flushPromises();

        setRanges(plotDiv, [0, 100]);
        getRelayoutHandler(plotDiv)({});
        expect(document.getElementById('hierarchy-level-slider').value).toBe('9');
    });

    it('does not override a manual level when the plot is only panned', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        let handler = getRelayoutHandler(plotDiv);

        setRanges(plotDiv, [33.3333, 66.6667]);
        handler({});
        expect(document.getElementById('hierarchy-level-slider').value).toBe('7');
        await flushPromises();

        setHierarchyLevel(3);
        await flushPromises();
        handler = getRelayoutHandler(plotDiv);

        setRanges(plotDiv, [43.3333, 76.6667]);
        handler({});

        expect(document.getElementById('hierarchy-level-slider').value).toBe('3');
    });

    it('does not change the zoom range when the manual slider is moved', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));
        setRanges(plotDiv, [33.3333, 66.6667]);
        getRelayoutHandler(plotDiv)({});
        await flushPromises();

        const before = JSON.parse(JSON.stringify(plotDiv.layout));
        global.Plotly.relayout.mockClear();

        setHierarchyLevel(2);

        expect(plotDiv.layout).toEqual(before);
        expect(global.Plotly.relayout.mock.calls.some(call => (
            call[1] && (call[1]['xaxis.range'] || call[1]['yaxis.range'])
        ))).toBe(false);
    });

    it('preserves the current zoom when a level change rebuilds the plot', async () => {
        await loadWithLevels(plotDiv, buildLevels(10));

        plotDiv.data = [{ x: [1], y: [1] }, { x: [] }];
        setRanges(plotDiv, [45, 55]);
        global.Plotly.newPlot.mockClear();

        getRelayoutHandler(plotDiv)({});

        expect(global.Plotly.newPlot).toHaveBeenCalled();
        const layout = global.Plotly.newPlot.mock.calls.at(-1)[2];
        expect(layout.xaxis.range).toEqual([45, 55]);
        expect(layout.yaxis.range).toEqual([45, 55]);
    });

    it('ignores zoom events when there are no hierarchy levels', async () => {
        plotDiv.layout = {
            xaxis: { range: [0, 100] },
            yaxis: { range: [0, 100] }
        };
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([]) });
        await loadClusters();
        await flushPromises();

        const handler = getRelayoutHandler(plotDiv);
        setRanges(plotDiv, [45, 55]);
        handler({});

        expect(document.getElementById('hierarchy-controls').classList.contains('hidden')).toBe(true);
    });
});

async function flushPromises() {
    await Promise.resolve();
    await Promise.resolve();
}
