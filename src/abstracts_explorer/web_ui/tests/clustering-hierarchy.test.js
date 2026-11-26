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
        cluster_descriptions: { 0: 'Papers about cats.', 1: 'Papers about dogs.' },
        cluster_centers: { 0: { x: 1, y: 1 }, 1: { x: 2, y: 2 } },
        hierarchy: { levels }
    };
}

const flatLevel = {
    level: 0,
    max_distance: 0.1,
    clusters: [
        {
            id: 0,
            label: 'Cat',
            description: 'Papers about cats.',
            size: 1,
            center: { x: 1, y: 1 },
            keywords: []
        },
        {
            id: 1,
            label: 'Dog',
            description: 'Papers about dogs.',
            size: 1,
            center: { x: 2, y: 2 },
            keywords: []
        }
    ],
    mapping: { 0: 0, 1: 1 }
};

const mergedLevel = {
    level: 1,
    max_distance: 0.2,
    clusters: [
        {
            id: 100,
            label: 'Animals',
            description: 'Papers about domestic animals.',
            size: 2,
            center: { x: 1.5, y: 1.5 },
            keywords: []
        }
    ],
    mapping: { 0: 100, 1: 100 }
};

const fourTopicFlatLevel = {
    level: 0,
    max_distance: 0.1,
    clusters: [0, 1, 2, 3].map(id => ({
        id,
        label: `Topic ${id}`,
        description: `Papers about topic ${id}.`,
        size: 1,
        center: { x: id + 1, y: id + 1 },
        keywords: []
    })),
    mapping: { 0: 0, 1: 1, 2: 2, 3: 3 }
};

const fourTopicMidLevel = {
    level: 1,
    max_distance: 0.2,
    clusters: [
        {
            id: 10,
            label: 'Topics 0 + 1',
            description: 'Papers about topics 0 and 1.',
            size: 2,
            center: { x: 1.5, y: 1.5 },
            keywords: []
        },
        {
            id: 2,
            label: 'Topic 2',
            description: 'Papers about topic 2.',
            size: 1,
            center: { x: 3, y: 3 },
            keywords: []
        },
        {
            id: 3,
            label: 'Topic 3',
            description: 'Papers about topic 3.',
            size: 1,
            center: { x: 4, y: 4 },
            keywords: []
        }
    ],
    mapping: { 0: 10, 1: 10, 2: 2, 3: 3 }
};

const fourTopicHighLevel = {
    level: 2,
    max_distance: 0.3,
    clusters: [
        {
            id: 100,
            label: 'Topics 0 + 1',
            description: 'Papers about topics 0 and 1.',
            size: 2,
            center: { x: 1.5, y: 1.5 },
            keywords: []
        },
        {
            id: 200,
            label: 'Topics 2 + 3',
            description: 'Papers about topics 2 and 3.',
            size: 2,
            center: { x: 3.5, y: 3.5 },
            keywords: []
        }
    ],
    mapping: { 0: 100, 1: 100, 2: 200, 3: 200 }
};

const fourTopicLevels = [fourTopicFlatLevel, fourTopicMidLevel, fourTopicHighLevel];

function buildFourTopicData(levels) {
    const points = [0, 1, 2, 3].map(id => ({
        id: `p${id}`,
        x: id + 1,
        y: id + 1,
        cluster: id,
        title: `Paper ${id}`,
        year: 2025,
        conference: 'NeurIPS'
    }));
    return {
        points,
        statistics: { total_papers: 4, n_clusters: 4, n_noise: 0, cluster_sizes: { 0: 1, 1: 1, 2: 1, 3: 1 } },
        cluster_labels: { 0: 'Topic 0', 1: 'Topic 1', 2: 'Topic 2', 3: 'Topic 3' },
        cluster_descriptions: {
            0: 'Papers about topic 0.',
            1: 'Papers about topic 1.',
            2: 'Papers about topic 2.',
            3: 'Papers about topic 3.'
        },
        cluster_centers: { 0: { x: 1, y: 1 }, 1: { x: 2, y: 2 }, 2: { x: 3, y: 3 }, 3: { x: 4, y: 4 } },
        hierarchy: { levels }
    };
}

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
        expect(mergedTraces[1].customdata[0].description).toBe('Papers about domestic animals.');
        expect(mergedTraces[1].hovertemplate).toContain('%{customdata.description}');

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

    it('assigns stable ids to document and star traces', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();
        setHierarchyLevel(0);
        await flushPromises();

        const traces = global.Plotly.newPlot.mock.calls.at(-1)[1];
        expect(traces[0].ids).toEqual(['p1', 'p2']);
        expect(traces[1].ids).toEqual(['0', '1']);
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
        expect(hiddenCall[1].ids[0]).toEqual(['p2']);
        expect(hiddenCall[1].customdata[0]).toHaveLength(1);
        expect(hiddenCall[1].customdata[0][0].id).toBe('p2');
        expect(hiddenCall[1]['marker.color'][0]).toEqual(['#ff7f0e']);
        expect(hiddenCall[1]['marker.color'][0]).not.toContain('#e5e7eb');

        const starHiddenCall = getStarRestyleCall();
        expect(starHiddenCall[1].ids[0]).toEqual(['1']);

        global.Plotly.restyle.mockClear();
        items[0].click();

        const restoredCall = getDocumentRestyleCall();
        expect(restoredCall[1].x[0]).toEqual([1, 2]);
        expect(restoredCall[1].text[0]).toEqual(['Paper 1', 'Paper 2']);
        expect(restoredCall[1].ids[0]).toEqual(['p1', 'p2']);
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
        expect(hiddenCall[1].ids[0]).toEqual(['p2', 'noise-1']);

        const noneButton = Array.from(document.querySelectorAll('#cluster-legend button'))
            .find(button => button.textContent === 'None');
        expect(noneButton).toBeDefined();
        global.Plotly.restyle.mockClear();
        noneButton.click();

        const clearedCall = getDocumentRestyleCall();
        expect(clearedCall[1].x[0]).toEqual([]);
        expect(clearedCall[1].ids[0]).toEqual([]);
        expect(clearedCall[1].customdata[0]).toEqual([]);
    });

    it('only splits visible clusters into children when lowering the level', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildFourTopicData(fourTopicLevels) });

        await loadClusters();
        await flushPromises();
        primePlotData(4);

        clickLegendItem('200');
        global.Plotly.restyle.mockClear();

        setHierarchyLevel(1);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['10']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p0', 'p1']);
        expect(getLegendItem('10').style.opacity).toBe('1');
        expect(getLegendItem('2').style.opacity).toBe('0.4');
        expect(getLegendItem('3').style.opacity).toBe('0.4');

        setHierarchyLevel(0);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['0', '1']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p0', 'p1']);
    });

    it('shows whole parent clusters with visible children when raising the level', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildFourTopicData(fourTopicLevels) });

        await loadClusters();
        await flushPromises();
        primePlotData(4);

        setHierarchyLevel(0);
        clickLegendButton('None');
        clickLegendItem('2');
        global.Plotly.restyle.mockClear();

        setHierarchyLevel(1);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['2']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p2']);

        setHierarchyLevel(2);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['200']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p2', 'p3']);
        expect(getLegendItem('200').style.opacity).toBe('1');
        expect(getLegendItem('100').style.opacity).toBe('0.4');
    });

    it('raises every parent with a visible child', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildFourTopicData(fourTopicLevels) });

        await loadClusters();
        await flushPromises();
        primePlotData(4);

        setHierarchyLevel(0);
        clickLegendButton('None');
        clickLegendItem('0');
        clickLegendItem('2');
        global.Plotly.restyle.mockClear();

        setHierarchyLevel(2);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['100', '200']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p0', 'p1', 'p2', 'p3']);
    });

    it('preserves the selection when lowering and raising back', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildFourTopicData(fourTopicLevels) });

        await loadClusters();
        await flushPromises();
        primePlotData(4);

        clickLegendItem('200');
        setHierarchyLevel(0);
        global.Plotly.restyle.mockClear();

        setHierarchyLevel(2);

        expect(getSelectionRestyleCall(1)[1].ids[0]).toEqual(['100']);
        expect(getSelectionRestyleCall(0)[1].ids[0]).toEqual(['p0', 'p1']);
    });

    it('isolates a cluster when its star is clicked and toggles back', async () => {
        let clickHandler;
        const plotDiv = document.getElementById('cluster-plot');
        plotDiv.on = jest.fn((event, handler) => {
            if (event === 'plotly_click') clickHandler = handler;
        });
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();
        setHierarchyLevel(0);
        await flushPromises();
        primePlotData(2);

        expect(clickHandler).toBeDefined();

        const starPoint = {
            customdata: { cluster_id: 0, size: 1, label: 'Cat', description: '' },
            data: { name: 'Clusters' }
        };

        clickHandler({ points: [starPoint] });

        expect(getLegendItem('0').style.opacity).toBe('1');
        expect(getLegendItem('1').style.opacity).toBe('0.4');
        expect(getStarRestyleCall()[1].ids[0]).toEqual(['0']);

        clickHandler({ points: [starPoint] });

        expect(getLegendItem('1').style.opacity).toBe('1');
    });

    it('live-filters legend items and persists the query across level changes', async () => {
        global.fetch.mockResolvedValueOnce({ ok: true, json: async () => buildData([flatLevel, mergedLevel]) });

        await loadClusters();
        setHierarchyLevel(0);
        await flushPromises();

        const input = document.getElementById('cluster-legend-search');
        expect(input).toBeTruthy();

        input.value = 'cat';
        input.dispatchEvent(new Event('input'));

        expect(getLegendItem('0').style.display).toBe('');
        expect(getLegendItem('1').style.display).toBe('none');

        setHierarchyLevel(1);

        const persistedInput = document.getElementById('cluster-legend-search');
        expect(persistedInput.value).toBe('cat');
        expect(getLegendItem('100').style.display).toBe('none');

        persistedInput.value = '';
        persistedInput.dispatchEvent(new Event('input'));
        expect(getLegendItem('100').style.display).toBe('');
    });
});

async function flushPromises() {
    await Promise.resolve();
    await Promise.resolve();
}

function primePlotData(pointCount) {
    const plotDiv = document.getElementById('cluster-plot');
    const xs = Array.from({ length: pointCount }, (_, index) => index);
    plotDiv.data = [
        { x: xs, y: xs },
        { x: [] }
    ];
}

function getLegendItem(clusterId) {
    return document.querySelector(`#cluster-legend [data-cluster-id="${clusterId}"]`);
}

function clickLegendItem(clusterId) {
    const item = getLegendItem(clusterId);
    expect(item).toBeTruthy();
    item.click();
    return item;
}

function clickLegendButton(label) {
    const button = Array.from(document.querySelectorAll('#cluster-legend button'))
        .find(candidate => candidate.textContent === label);
    expect(button).toBeDefined();
    button.click();
    return button;
}

function getSelectionRestyleCall(traceIndex) {
    const calls = global.Plotly.restyle.mock.calls.filter(call => (
        Array.isArray(call[2])
        && call[2][0] === traceIndex
        && call[1]
        && Array.isArray(call[1].ids)
    ));
    return calls[calls.length - 1];
}

function getDocumentRestyleCall() {
    return getRestyleCall(0);
}

function getStarRestyleCall() {
    return getRestyleCall(1);
}

function getRestyleCall(traceIndex) {
    return global.Plotly.restyle.mock.calls.find(call => (
        Array.isArray(call[2])
        && call[2][0] === traceIndex
        && call[1]
        && Array.isArray(call[1].x)
    ));
}
