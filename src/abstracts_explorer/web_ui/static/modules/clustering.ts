// @ts-nocheck -- staged conversion: this Plotly-heavy module is typed incrementally.
/**
 * Clustering Module
 *
 * Handles cluster visualization, analysis, and management using Plotly.
 */

import { API_BASE, PLOTLY_COLORS } from './utils/constants.js';
import { showLoading, showErrorInElement } from './utils/ui-utils.js';
import { parseJsonResponse } from './utils/api-utils.js';
import { getSelectedConference, getSelectedLocation, getSelectedYears, escapeHtml, escapeAttr } from './utils/dom-utils.js';
import { formatPaperCard } from './paper-card.js';
import { renderInlineMarkdownWithLatex } from './utils/markdown-utils.js';
import { ensurePlotly, getPlotly } from './plotly-loader.js';

// Plotly is loaded lazily (and tests inject a global mock). The proxy defers
// resolution to call time so the existing `Plotly.xxx` call sites keep working.
const Plotly: any = new Proxy({}, {
    get(_target, prop) {
        const impl = getPlotly();
        const value = impl[prop];
        return typeof value === 'function' ? value.bind(impl) : value;
    }
});

/**
 * Return the median of a numeric array.
 * @param {number[]} values
 * @returns {number}
 */
function _median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2
        ? sorted[middle]
        : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Returns Plotly background and font colours matching the current colour scheme.
 * Backgrounds are always transparent so the parent card's CSS (dark:bg-gray-800)
 * handles the surface colour automatically.
 * @returns {{ plot_bgcolor: string, paper_bgcolor: string, font: { color: string } }}
 */
function getPlotColors() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return {
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: isDark ? '#e5e7eb' : '#374151' }  // gray-200 vs gray-700
    };
}

/**
 * Re-apply font colour to every active Plotly chart inside the clusters tab.
 * Called automatically when the OS colour scheme changes.
 */
function _refreshClusteringPlotColors() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const fontColor = isDark ? '#e5e7eb' : '#374151';
    const container = document.getElementById('clusters-tab');
    if (!container || typeof Plotly === 'undefined') return;
    /* global Plotly */
    container.querySelectorAll('.js-plotly-plot').forEach(function (el) {
        Plotly.relayout(el, { 'font.color': fontColor });
    });
}

// Keep clustering charts in sync when the OS colour scheme changes
if (typeof window !== 'undefined' && window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', _refreshClusteringPlotColors);
}

// Cluster state
let clusterData = null;
let currentClusterConfig = {
    reduction_method: 'umap',
    n_components: 2,
    clustering_method: 'bertopic-agglomerative',
    distance_threshold: 0.5,
    linkage: 'ward',
    use_llm_labels: true
};
// Track selected clusters for multi-select
let selectedClusters = new Set();
let currentLevelIndex = 4;
let currentLevelState = null;
// Persisted search query for the cluster legend (survives re-renders)
let legendSearchQuery = '';

// Zoom-driven hierarchy state
const ZOOM_LEVEL_MIN = 1;
const ZOOM_LEVEL_MAX = 10;
const MIN_ZOOM_DELTA = 0.01;
const ZOOM_HYSTERESIS = 0.1;
const ZOOM_SYNC_DELAY_MS = 120;
let baselineRanges = null;
let lastAutoZoom = 1;
let suppressZoomSync = false;
let resetViewOnNextRender = true;
let zoomSyncTimer = null;
let pendingZoomPlot = null;
let renderedDocumentCount = 0;

// Custom topic clustering state
let customQueryClusters = [];  // Array of custom topic cluster objects
let customClusterMode = false;  // Whether custom topic mode is active
let customClusterVisibility = {};  // Track visibility of each custom topic by ID

// Color palette for topic evolution charts
const TOPIC_COLORS = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#6366f1', '#0891b2', '#be185d'];

// Fixed DOM id for the single topic evolution Plotly chart
const TOPIC_EVOLUTION_PLOT_ID = 'topic-evolution-plot';

// Topic evolution chart state – accumulated across multiple queries
let topicEvolutionTraces = [];   // All traces added so far
let topicEvolutionColorIdx = 0;  // Running colour index
let topicEvolutionTopics = [];   // Topic names already loaded (for the chart title)

const DEFAULT_HIERARCHY_LEVEL = 4;

/**
 * Check if clusters are loaded
 * @returns {boolean} True if clusters are loaded
 */
export function areClustersLoaded() {
    return clusterData !== null;
}

/**
 * Reset the loaded cluster data so the next tab switch will trigger a reload.
 * Call this when the conference/year filter changes.
 */
export function resetClusters() {
    clusterData = null;
    customQueryClusters = [];
    customClusterMode = false;
    customClusterVisibility = {};
    selectedClusters.clear();
    legendSearchQuery = '';
    currentLevelIndex = DEFAULT_HIERARCHY_LEVEL;
    currentLevelState = null;
    _cancelZoomSync();
    baselineRanges = null;
    lastAutoZoom = 1;
    suppressZoomSync = false;
    resetViewOnNextRender = true;
    renderedDocumentCount = 0;
    // Reset topic evolution state
    topicEvolutionTraces = [];
    topicEvolutionColorIdx = 0;
    topicEvolutionTopics = [];
    const topicContainer = document.getElementById('topic-evolution-container');
    if (topicContainer) {
        topicContainer.innerHTML = '';
    }
    _setHierarchyControlsVisible(false);
}

/**
 * Load and visualize clusters
 * @async
 */
export async function loadClusters() {
    try {
        await ensurePlotly();
        showLoading('cluster-plot', 'Loading clusters...');

        // Build request body with conference/year filters
        const computeBody = {};
        const selectedConference = getSelectedConference();
        const selectedYears = getSelectedYears();
        if (selectedConference) computeBody.conferences = [selectedConference];
        if (selectedYears.length > 0) computeBody.years = selectedYears;
        const selectedLocation = getSelectedLocation();
        if (selectedLocation) computeBody.locations = [selectedLocation];

        // Fetch pre-computed clustering data
        const response = await fetch(`${API_BASE}/api/clusters/compute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(computeBody)
        });

        clusterData = await parseJsonResponse(response);

        if (clusterData.error) {
            showErrorInElement('cluster-plot', clusterData.error);
            return;
        }

        const levels = _getHierarchyLevels();
        currentLevelIndex = levels.length ? levels.length - 1 : 0;
        selectedClusters.clear();
        _cancelZoomSync();
        baselineRanges = null;
        lastAutoZoom = 1;
        resetViewOnNextRender = true;
        renderedDocumentCount = 0;
        _bindHierarchySlider();
        renderCurrentLevel();
    } catch (error) {
        console.error('Error loading clusters:', error);
        showErrorInElement('cluster-plot', error.message || 'Could not load clusters. Please try again.');
    }
}

/**
 * Return hierarchy levels from the cached payload.
 * @returns {Array}
 */
function _getHierarchyLevels() {
    return (clusterData && clusterData.hierarchy && clusterData.hierarchy.levels) || [];
}

/**
 * Return the active hierarchy level, or null for the flat fallback.
 * @returns {Object|null}
 */
function _getActiveLevel() {
    const levels = _getHierarchyLevels();
    if (!levels.length) return null;
    const index = Math.min(Math.max(currentLevelIndex, 0), levels.length - 1);
    return levels[index];
}

/**
 * Read the current x/y axis ranges from a Plotly plot.
 * @param {HTMLElement} plotDiv
 * @returns {{x: number[], y: number[]}|null}
 */
function _capturePlotRanges(plotDiv) {
    if (!plotDiv) return null;
    const fullLayout = plotDiv._fullLayout || {};
    const layout = plotDiv.layout || {};
    const fullX = fullLayout.xaxis && fullLayout.xaxis.range;
    const fullY = fullLayout.yaxis && fullLayout.yaxis.range;
    const layoutX = layout.xaxis && layout.xaxis.range;
    const layoutY = layout.yaxis && layout.yaxis.range;
    const xRange = Array.isArray(fullX) ? fullX : (Array.isArray(layoutX) ? layoutX : null);
    const yRange = Array.isArray(fullY) ? fullY : (Array.isArray(layoutY) ? layoutY : null);
    if (!xRange || !yRange) return null;
    const x = xRange.map(Number);
    const y = yRange.map(Number);
    if (!x.every(Number.isFinite) || !y.every(Number.isFinite)) return null;
    return { x, y };
}

/**
 * Compute the current zoom factor relative to the initial full view.
 * @param {HTMLElement} plotDiv
 * @returns {number}
 */
function _getZoomFactor(plotDiv) {
    if (!baselineRanges) return 1;
    const current = _capturePlotRanges(plotDiv);
    if (!current) return 1;
    const baseX = Math.abs(baselineRanges.x[1] - baselineRanges.x[0]);
    const baseY = Math.abs(baselineRanges.y[1] - baselineRanges.y[0]);
    const curX = Math.abs(current.x[1] - current.x[0]);
    const curY = Math.abs(current.y[1] - current.y[0]);
    if (!(baseX > 0) || !(baseY > 0) || !(curX > 0) || !(curY > 0)) return 1;
    const zoom = Math.sqrt((baseX / curX) * (baseY / curY));
    return Number.isFinite(zoom) ? Math.max(1, zoom) : 1;
}

/**
 * Map a zoom factor to a hierarchy level index.
 *
 * A small deadband around each level boundary prevents rapid flip-flopping
 * when the zoom factor hovers right at a threshold.
 * @param {number} zoom
 * @returns {number}
 */
function _levelForZoom(zoom) {
    const levels = _getHierarchyLevels();
    if (!levels.length) return 0;
    const maxIndex = levels.length - 1;
    const span = ZOOM_LEVEL_MAX - ZOOM_LEVEL_MIN;
    const t = Math.min(Math.max((zoom - ZOOM_LEVEL_MIN) / span, 0), 1);
    const continuous = maxIndex * (1 - t);
    const target = Math.round(continuous);
    if (target === currentLevelIndex) return currentLevelIndex;
    if (Math.abs(continuous - currentLevelIndex) < 0.5 + ZOOM_HYSTERESIS) {
        return currentLevelIndex;
    }
    return target;
}

/**
 * Adjust the hierarchy level to match the current zoom factor.
 * @param {HTMLElement} plotDiv
 */
function _syncLevelToZoom(plotDiv) {
    if (customClusterMode || suppressZoomSync) return;
    if (!_getHierarchyLevels().length || !baselineRanges) return;
    const zoom = _getZoomFactor(plotDiv);
    if (Math.abs(zoom - lastAutoZoom) <= MIN_ZOOM_DELTA) return;
    lastAutoZoom = zoom;
    const level = _levelForZoom(zoom);
    if (level !== currentLevelIndex) setHierarchyLevel(level);
}

/**
 * Cancel a pending zoom-driven level sync.
 */
function _cancelZoomSync() {
    if (zoomSyncTimer !== null) {
        clearTimeout(zoomSyncTimer);
    }
    zoomSyncTimer = null;
    pendingZoomPlot = null;
}

/**
 * Coalesce rapid relayout events into a single level sync once the zoom
 * gesture settles. Rendering the new level for every wheel tick is what makes
 * the plot stutter on large datasets.
 * @param {HTMLElement} plotDiv
 */
function _scheduleZoomSync(plotDiv) {
    pendingZoomPlot = plotDiv;
    if (zoomSyncTimer !== null) {
        clearTimeout(zoomSyncTimer);
    }
    zoomSyncTimer = setTimeout(function () {
        zoomSyncTimer = null;
        const target = pendingZoomPlot;
        pendingZoomPlot = null;
        if (target) _syncLevelToZoom(target);
    }, ZOOM_SYNC_DELAY_MS);
}

/**
 * Keep the hierarchy level in sync with Plotly zoom changes.
 * @param {HTMLElement} plotDiv
 */
function _bindZoomSyncHandler(plotDiv) {
    if (!plotDiv || typeof plotDiv.on !== 'function') return;
    if (typeof plotDiv.removeAllListeners === 'function') {
        plotDiv.removeAllListeners('plotly_relayout');
    }
    plotDiv.on('plotly_relayout', function () {
        _scheduleZoomSync(plotDiv);
    });
}

/**
 * Translate a cluster selection from one hierarchy level to another.
 *
 * Selections are resolved to their flat topic IDs and then re-grouped at the
 * target level. Lowering a level therefore reveals only the children of the
 * visible clusters, while raising it reveals the whole parent cluster for any
 * visible child.
 * @param {Object|null} fromLevel
 * @param {Set<string>} selection
 * @param {Object|null} toLevel
 * @returns {Set<string>}
 */
function _convertSelection(fromLevel, selection, toLevel) {
    const fromMapping = (fromLevel && fromLevel.mapping) || {};
    const toMapping = (toLevel && toLevel.mapping) || {};
    const visibleTopics = new Set();
    Object.keys(fromMapping).forEach(topic => {
        if (selection.has(String(fromMapping[topic]))) {
            visibleTopics.add(topic);
        }
    });
    const converted = new Set();
    Object.keys(toMapping).forEach(topic => {
        if (visibleTopics.has(topic)) {
            converted.add(String(toMapping[topic]));
        }
    });
    return converted;
}

/**
 * Bind the hierarchy range slider.
 */
function _bindHierarchySlider() {
    const slider = document.getElementById('hierarchy-level-slider');
    if (!slider) return;
    slider.oninput = function () {
        const index = parseInt(slider.value, 10);
        if (!isNaN(index)) setHierarchyLevel(index);
    };
}

/**
 * Select a hierarchy level and re-render, preserving the visible clusters.
 *
 * Lowering a level splits the visible clusters into their children; raising it
 * shows the whole parent cluster for any visible child.
 * @param {number} index
 */
export function setHierarchyLevel(index) {
    const levels = _getHierarchyLevels();
    if (!levels.length) return;
    _cancelZoomSync();
    const previousLevel = _getActiveLevel();
    currentLevelIndex = Math.min(Math.max(parseInt(index, 10) || 0, 0), levels.length - 1);
    const nextLevel = levels[currentLevelIndex];
    if (previousLevel !== nextLevel) {
        selectedClusters = _convertSelection(previousLevel, selectedClusters, nextLevel);
    }
    renderCurrentLevel(true);
}

/**
 * Show or hide the hierarchy controls.
 * @param {boolean} visible
 */
function _setHierarchyControlsVisible(visible) {
    const controls = document.getElementById('hierarchy-controls');
    if (!controls) return;
    if (visible) {
        controls.classList.remove('hidden');
    } else {
        controls.classList.add('hidden');
    }
}

/**
 * Build the current level state: cluster entries, colors, document arrays, and stars.
 * @returns {Object}
 */
function _buildLevelState() {
    const level = _getActiveLevel();
    const mapping = (level && level.mapping) || {};
    const metaById = {};
    if (level && level.clusters) {
        level.clusters.forEach(cluster => {
            metaById[Number(cluster.id)] = cluster;
        });
    }

    const flatCenters = (clusterData && clusterData.cluster_centers) || {};
    const flatLabels = (clusterData && clusterData.cluster_labels) || {};
    const flatDescriptions = (clusterData && clusterData.cluster_descriptions) || {};
    const groups = {};
    const points = (clusterData && clusterData.points) || [];

    points.forEach(point => {
        const rawCluster = Number(point.cluster);
        const mapped = mapping[String(rawCluster)];
        const clusterId = Number(mapped !== undefined ? mapped : rawCluster);
        if (!groups[clusterId]) groups[clusterId] = [];
        groups[clusterId].push(point);
    });

    const clusterIds = Object.keys(groups)
        .map(Number)
        .filter(id => id >= 0)
        .sort((a, b) => (groups[b].length - groups[a].length) || (a - b));

    const entries = clusterIds.map((id, index) => {
        const meta = metaById[id] || {};
        const clusterPoints = groups[id];
        let center = meta.center;
        if (!center && flatCenters[String(id)]) center = flatCenters[String(id)];
        if (!center) {
            center = {
                x: _median(clusterPoints.map(point => point.x)),
                y: _median(clusterPoints.map(point => point.y))
            };
        }
        const label = meta.label
            || flatLabels[String(id)]
            || flatLabels[id]
            || `Cluster ${id}`;
        const description = meta.description
            || flatDescriptions[String(id)]
            || flatDescriptions[id]
            || '';
        return {
            id,
            points: clusterPoints,
            label,
            description,
            size: meta.size !== undefined ? meta.size : clusterPoints.length,
            color: PLOTLY_COLORS[index % PLOTLY_COLORS.length],
            center
        };
    });

    const colorById = {};
    entries.forEach(entry => {
        colorById[entry.id] = entry.color;
    });

    const xs = [];
    const ys = [];
    const texts = [];
    const customdata = [];
    const colors = [];
    const ids = [];
    const clusterIdsByPoint = [];
    points.forEach((point, index) => {
        const rawCluster = Number(point.cluster);
        const mapped = mapping[String(rawCluster)];
        const clusterId = Number(mapped !== undefined ? mapped : rawCluster);
        xs.push(point.x);
        ys.push(point.y);
        texts.push(point.title || point.id);
        clusterIdsByPoint.push(clusterId);
        ids.push(point.id !== undefined && point.id !== null ? String(point.id) : `point-${index}`);
        customdata.push({
            id: point.id,
            title: point.title || '',
            year: point.year || '',
            conference: point.conference || '',
            session: point.session || '',
            cluster_id: clusterId
        });
        colors.push(colorById[clusterId] || '#9ca3af');
    });

    const starX = [];
    const starY = [];
    const starText = [];
    const starCustomdata = [];
    const starColors = [];
    const starIds = [];
    entries.forEach(entry => {
        starX.push(entry.center.x);
        starY.push(entry.center.y);
        starText.push(entry.label);
        starIds.push(String(entry.id));
        starCustomdata.push({
            cluster_id: entry.id,
            size: entry.size,
            label: entry.label,
            description: entry.description
        });
        starColors.push(entry.color);
    });

    const hasDescriptions = entries.some(entry => Boolean(entry.description));
    const starHoverTemplate = hasDescriptions
        ? '<b>%{text}</b><br>' +
          '%{customdata.description}<br>' +
          '%{customdata.size} papers<br>' +
          '<extra></extra>'
        : '<b>%{text}</b><br>' +
          '%{customdata.size} papers<br>' +
          '<extra></extra>';

    return {
        level,
        entries,
        colorById,
        clusterIdsByPoint,
        xs,
        ys,
        texts,
        customdata,
        colors,
        ids,
        starX,
        starY,
        starText,
        starCustomdata,
        starColors,
        starIds,
        starHoverTemplate
    };
}

/**
 * Render the active hierarchy level with one document trace and one star trace.
 * @param {boolean} preserveSelection - Keep the current selection instead of selecting every cluster.
 */
export function renderCurrentLevel(preserveSelection = false) {
    if (!clusterData || !clusterData.points) {
        console.error('No cluster data to visualize');
        return;
    }

    const state = _buildLevelState();
    currentLevelState = state;
    if (!preserveSelection) {
        selectedClusters = new Set(state.entries.map(entry => String(entry.id)));
    }

    const shouldResetView = resetViewOnNextRender;
    resetViewOnNextRender = false;
    if (shouldResetView) {
        baselineRanges = null;
        lastAutoZoom = 1;
    }

    const plotColors = getPlotColors();
    const layout = {
        title: '',
        xaxis: {
            title: '',
            zeroline: false,
            showgrid: false,
            showticklabels: false,
            ticks: ''
        },
        yaxis: {
            title: '',
            zeroline: false,
            showgrid: false,
            showticklabels: false,
            ticks: ''
        },
        hovermode: 'closest',
        showlegend: false,
        plot_bgcolor: plotColors.plot_bgcolor,
        paper_bgcolor: plotColors.paper_bgcolor,
        font: plotColors.font,
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hoverlabel: {
            namelength: -1,
            align: 'left'
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true
    };

    const documentTrace = {
        x: state.xs,
        y: state.ys,
        mode: 'markers',
        type: 'scatter',
        name: 'Papers',
        text: state.texts,
        customdata: state.customdata,
        ids: state.ids,
        marker: {
            color: state.colors,
            size: 8,
            opacity: 0.7,
            line: {
                color: 'white',
                width: 0.5
            }
        },
        hovertemplate: '<b>%{text}</b><br>' +
                      'Year: %{customdata.year}<br>' +
                      'Conference: %{customdata.conference}<br>' +
                      '<extra></extra>',
        showlegend: false
    };

    const starTrace = {
        x: state.starX,
        y: state.starY,
        mode: 'markers',
        type: 'scatter',
        name: 'Clusters',
        text: state.starText,
        customdata: state.starCustomdata,
        ids: state.starIds,
        marker: {
            color: state.starColors,
            symbol: 'star',
            size: 16,
            opacity: 1.0,
            line: {
                color: 'white',
                width: 2
            }
        },
        hovertemplate: state.starHoverTemplate,
        showlegend: false
    };

    const plotElement = document.getElementById('cluster-plot');
    if (!plotElement) return;

    const previousRanges = shouldResetView ? null : _capturePlotRanges(plotElement);

    const canRestyle = !shouldResetView
        && plotElement.data
        && plotElement.data.length === 2;

    if (canRestyle) {
        suppressZoomSync = true;
        try {
            if (!baselineRanges) {
                baselineRanges = _capturePlotRanges(plotElement);
            }
            lastAutoZoom = _getZoomFactor(plotElement);
            createLevelLegend(state);
            _updateHierarchyControls(state);
            applySelectionToPlot();
        } finally {
            suppressZoomSync = false;
        }
        return;
    }

    if (previousRanges) {
        layout.xaxis.range = previousRanges.x.slice();
        layout.yaxis.range = previousRanges.y.slice();
    }

    suppressZoomSync = true;
    plotElement.innerHTML = '';
    Plotly.newPlot(plotElement, [documentTrace, starTrace], layout, config).then(function () {
        try {
            Plotly.relayout(plotElement, {
                'xaxis.fixedrange': false,
                'yaxis.fixedrange': false
            });
            if (!baselineRanges) {
                baselineRanges = _capturePlotRanges(plotElement);
            }
            lastAutoZoom = _getZoomFactor(plotElement);
            renderedDocumentCount = state.xs.length;
            _bindClusterClickHandler(plotElement);
            _bindZoomSyncHandler(plotElement);
            applySelectionToPlot();
        } finally {
            suppressZoomSync = false;
        }
    }).catch(function (error) {
        suppressZoomSync = false;
        console.error('Error rendering cluster plot:', error);
    });

    createLevelLegend(state);
    _updateHierarchyControls(state);
}

/**
 * Keep the legacy visualizeClusters name working.
 */
export function visualizeClusters() {
    renderCurrentLevel();
}

/**
 * Bind the cluster plot click handler and remove any previous one.
 * @param {HTMLElement} plotDiv
 */
function _bindClusterClickHandler(plotDiv) {
    if (!plotDiv || typeof plotDiv.on !== 'function') return;
    if (typeof plotDiv.removeAllListeners === 'function') {
        plotDiv.removeAllListeners('plotly_click');
    }
    plotDiv.on('plotly_click', function (data) {
        const point = data.points && data.points[0];
        if (!point) return;
        const customdata = point.customdata || {};
        const isStar = (point.data && point.data.name === 'Clusters')
            || (customdata.cluster_id !== undefined && customdata.id == null);
        if (isStar && customdata.cluster_id !== undefined) {
            _toggleClusterIsolation(customdata.cluster_id);
            return;
        }
        if (customdata.id) {
            showClusterPaperDetails(customdata.id, customdata);
        }
    });
}

/**
 * Isolate a cluster (hide all others), toggling back to all clusters when the
 * cluster is already the only visible one.
 * @param {number|string} clusterId - ID of the cluster whose star was clicked
 */
function _toggleClusterIsolation(clusterId) {
    const state = currentLevelState;
    if (!state || !state.entries) return;
    const key = String(clusterId);
    const allIds = state.entries.map(entry => String(entry.id));
    const isIsolated = selectedClusters.size === 1 && selectedClusters.has(key);
    selectedClusters = isIsolated ? new Set(allIds) : new Set([key]);
    applySelectionToPlot();
}

/**
 * Update hierarchy slider state and label.
 * @param {Object} state
 */
function _updateHierarchyControls(state) {
    const levels = _getHierarchyLevels();
    if (!levels.length || customClusterMode) {
        _setHierarchyControlsVisible(false);
        return;
    }
    _setHierarchyControlsVisible(true);
    const slider = document.getElementById('hierarchy-level-slider');
    const label = document.getElementById('hierarchy-level-label');
    const levelCount = levels.length;
    const clusterCount = state && state.entries ? state.entries.length : 0;
    if (slider) {
        slider.max = String(levelCount - 1);
        slider.value = String(Math.min(currentLevelIndex, levelCount - 1));
    }
    if (label) {
        label.textContent = `Level ${currentLevelIndex} · ${clusterCount} clusters`;
    }
}

/**
 * Apply cluster selection by hiding documents and filtering stars.
 */
function applySelectionToPlot() {
    const state = currentLevelState;
    const plotDiv = document.getElementById('cluster-plot');
    if (!state || !plotDiv || !plotDiv.data || !plotDiv.data.length) return;

    const visibleIndices = [];
    state.clusterIdsByPoint.forEach((clusterId, index) => {
        const isVisible = clusterId < 0
            ? selectedClusters.size > 0
            : selectedClusters.has(String(clusterId));
        if (isVisible) visibleIndices.push(index);
    });

    const allVisible = visibleIndices.length === state.xs.length;
    const traceHasAllPoints = renderedDocumentCount === state.xs.length;

    if (allVisible && traceHasAllPoints) {
        Plotly.restyle(plotDiv, {
            'marker.color': [state.colors]
        }, [0]);
    } else {
        Plotly.restyle(plotDiv, {
            x: [visibleIndices.map(index => state.xs[index])],
            y: [visibleIndices.map(index => state.ys[index])],
            text: [visibleIndices.map(index => state.texts[index])],
            customdata: [visibleIndices.map(index => state.customdata[index])],
            ids: [visibleIndices.map(index => state.ids[index])],
            'marker.color': [visibleIndices.map(index => state.colors[index])]
        }, [0]);
    }
    renderedDocumentCount = visibleIndices.length;

    const visibleEntries = state.entries.filter(entry => selectedClusters.has(String(entry.id)));
    Plotly.restyle(plotDiv, {
        x: [visibleEntries.map(entry => entry.center.x)],
        y: [visibleEntries.map(entry => entry.center.y)],
        text: [visibleEntries.map(entry => entry.label)],
        customdata: [visibleEntries.map(entry => ({
            cluster_id: entry.id,
            size: entry.size,
            label: entry.label,
            description: entry.description
        }))],
        ids: [visibleEntries.map(entry => String(entry.id))],
        hovertemplate: [state.starHoverTemplate],
        'marker.color': [visibleEntries.map(entry => entry.color)]
    }, [1]);

    _updateLegendSelectionStyles();
}

/**
 * Update legend item opacity to reflect selection.
 */
function _updateLegendSelectionStyles() {
    document.querySelectorAll('#cluster-legend [data-cluster-id]').forEach(item => {
        const isSelected = selectedClusters.has(String(item.dataset.clusterId));
        item.style.opacity = isSelected ? '1' : '0.4';
        item.style.backgroundColor = isSelected ? 'rgb(249 250 251)' : 'transparent';
    });
}

/**
 * Format cluster statistics as HTML string for the legend title.
 * @param {Object} statistics
 * @param {Object} labels
 * @returns {string}
 */
function formatClusterStats(statistics, labels) {
    let statsHTML = 'Clusters';
    if (statistics) {
        statsHTML += `<br><span class="text-xs font-normal">${statistics.total_papers} papers in ${statistics.n_clusters} clusters`;
        if (statistics.n_noise > 0) {
            statsHTML += ` (<span class="text-red-600">${statistics.n_noise}</span> noise)`;
        }
        statsHTML += '</span>';
        if (labels && Object.keys(labels).length > 0) {
            statsHTML += '<br><span class="text-xs font-normal text-green-600">✓ Labels generated</span>';
        }
    }
    return statsHTML;
}

/**
 * Filter legend elements live based on a search query.
 *
 * Matching is a case-insensitive substring search over each element's
 * `data-search-text`. Non-matching elements are hidden but kept in the DOM so
 * their selection state is preserved.
 *
 * @param {HTMLElement} container - Container holding the legend elements
 * @param {string} selector - Selector identifying the filterable elements
 * @param {string} query - Search query entered by the user
 */
function applyLegendSearch(container, selector, query) {
    if (!container) return;
    const normalized = (query || '').trim().toLowerCase();
    container.querySelectorAll(selector).forEach(element => {
        const haystack = (element.dataset.searchText || element.textContent || '').toLowerCase();
        const matches = normalized === '' || haystack.includes(normalized);
        element.style.display = matches ? '' : 'none';
    });
}

/**
 * Create the search input used to live-filter the legend.
 *
 * @param {Function} onInput - Callback invoked with the current query on input
 * @returns {HTMLDivElement} Wrapper element containing the input and clear button
 */
function createLegendSearchField(onInput) {
    const wrapper = document.createElement('div');
    wrapper.className = 'relative mb-2';

    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'cluster-legend-search';
    input.placeholder = 'Search clusters...';
    input.value = legendSearchQuery;
    input.className = 'w-full pr-10 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-600 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500';
    input.addEventListener('input', () => {
        legendSearchQuery = input.value;
        onInput(input.value);
    });

    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.setAttribute('aria-label', 'Clear cluster search');
    clearBtn.title = 'Clear search';
    clearBtn.className = 'absolute right-2 top-1/2 -translate-y-1/2 px-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200';
    const clearIcon = document.createElement('i');
    clearIcon.className = 'fas fa-times';
    clearBtn.appendChild(clearIcon);
    clearBtn.addEventListener('click', () => {
        legendSearchQuery = '';
        input.value = '';
        onInput('');
        input.focus();
    });

    wrapper.appendChild(input);
    wrapper.appendChild(clearBtn);
    return wrapper;
}

/**
 * Create the custom legend for the active level.
 * @param {Object} state
 */
function createLevelLegend(state) {
    const legendContainer = document.getElementById('cluster-legend');
    if (!legendContainer) return;

    legendContainer.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'mb-3';

    const title = document.createElement('h4');
    title.className = 'text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2';
    title.innerHTML = formatClusterStats(clusterData && clusterData.statistics, clusterData && clusterData.cluster_labels);
    header.appendChild(title);

    const levelInfo = document.createElement('p');
    levelInfo.className = 'text-xs text-gray-600 dark:text-gray-400 mb-2';
    levelInfo.textContent = `Level ${currentLevelIndex} · ${state.entries.length} clusters`;
    header.appendChild(levelInfo);

    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'space-y-1 overflow-y-auto pr-2 flex-1';
    itemsContainer.style.minHeight = '0';
    itemsContainer.style.maxHeight = '100%';

    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'flex gap-2';

    const selectAllBtn = document.createElement('button');
    selectAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors';
    selectAllBtn.textContent = 'All';
    selectAllBtn.addEventListener('click', () => {
        const visibleIds = [];
        itemsContainer.querySelectorAll('[data-cluster-id]').forEach((item: HTMLElement) => {
            if (item.style.display !== 'none') {
                visibleIds.push(String(item.dataset.clusterId));
            }
        });
        selectedClusters = new Set(visibleIds);
        applySelectionToPlot();
    });

    const clearAllBtn = document.createElement('button');
    clearAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors';
    clearAllBtn.textContent = 'None';
    clearAllBtn.addEventListener('click', () => {
        const hiddenIds = [];
        itemsContainer.querySelectorAll('[data-cluster-id]').forEach((item: HTMLElement) => {
            if (item.style.display === 'none') {
                hiddenIds.push(String(item.dataset.clusterId));
            }
        });
        selectedClusters = new Set(hiddenIds);
        applySelectionToPlot();
    });

    buttonContainer.appendChild(selectAllBtn);
    buttonContainer.appendChild(clearAllBtn);
    header.appendChild(buttonContainer);

    header.appendChild(createLegendSearchField(query => {
        applyLegendSearch(itemsContainer, '[data-cluster-id]', query);
    }));

    legendContainer.appendChild(header);

    state.entries.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 p-2 rounded cursor-pointer transition-all';
        item.dataset.clusterId = String(entry.id);
        item.dataset.searchText = `${entry.label} ${entry.description || ''}`;
        item.title = entry.description
            ? `${entry.label} (${entry.size} papers)\n${entry.description}`
            : `${entry.label} (${entry.size} papers)`;
        item.style.opacity = '1';
        item.style.backgroundColor = 'rgb(249 250 251)';

        const colorBox = document.createElement('div');
        colorBox.className = 'w-4 h-4 rounded flex-shrink-0';
        colorBox.style.backgroundColor = entry.color;

        const labelText = document.createElement('span');
        labelText.className = 'text-sm text-gray-700 dark:text-gray-300 flex-1 truncate max-w-[14rem]';
        labelText.textContent = `${entry.label} (${entry.size})`;

        item.appendChild(colorBox);
        item.appendChild(labelText);

        item.addEventListener('click', () => {
            const key = String(entry.id);
            if (selectedClusters.has(key)) {
                selectedClusters.delete(key);
            } else {
                selectedClusters.add(key);
            }
            applySelectionToPlot();
        });

        itemsContainer.appendChild(item);
    });

    legendContainer.appendChild(itemsContainer);
    applyLegendSearch(itemsContainer, '[data-cluster-id]', legendSearchQuery);
    _updateLegendSelectionStyles();
}

/**
 * Show details of selected paper from cluster
 * @param {string} paperId - Paper ID
 * @param {Object} basicInfo - Basic paper info
 * @async
 */
export async function showClusterPaperDetails(paperId, basicInfo) {
    const detailsDiv = document.getElementById('selected-paper-details');
    const contentDiv = document.getElementById('selected-paper-content');
    
    if (!detailsDiv || !contentDiv) {
        console.warn('Cluster paper details elements not found');
        return;
    }
    
    detailsDiv.classList.remove('hidden');
    
    // Show loading state with basic info
    const loadingPaper = {
        uid: paperId,
        title: basicInfo.title,
        authors: ['Loading...'],
        year: basicInfo.year,
        conference: basicInfo.conference,
        abstract: 'Loading full details...'
    };
    
    try {
        contentDiv.innerHTML = formatPaperCard(loadingPaper, { 
            compact: false,
            idPrefix: 'cluster-paper-detail'
        });
    } catch (error) {
        console.error('Error formatting loading state:', error);
        contentDiv.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h4 class="text-lg font-semibold text-gray-800 dark:text-gray-100">${renderInlineMarkdownWithLatex(basicInfo.title)}</h4>
                <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">Loading full details...</p>
            </div>
        `;
    }
    
    // Fetch full paper details
    try {
        const response = await fetch(`${API_BASE}/api/paper/${paperId}`);
        const paper = await parseJsonResponse(response);
        
        const formattedPaper = {
            uid: paperId,
            title: paper.title,
            authors: paper.authors || [],
            year: paper.year,
            conference: paper.conference,
            session: paper.session,
            poster_position: paper.poster_position,
            abstract: paper.abstract,
            paper_url: paper.url,
            keywords: paper.keywords
        };
        
        contentDiv.innerHTML = formatPaperCard(formattedPaper, { 
            compact: false,
            idPrefix: 'cluster-paper-detail'
        });
    } catch (error) {
        console.error('Error fetching paper details:', error);
        contentDiv.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <p class="text-sm text-red-500">${escapeHtml(error.message || 'Could not load paper details. Please try again.')}</p>
            </div>
        `;
    }
}

/**
 * Export cluster data as JSON
 */
export function exportClusters() {
    if (!clusterData) {
        alert('No cluster data to export');
        return;
    }
    
    const dataStr = JSON.stringify(clusterData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json;charset=utf-8' });
    const objectUrl = URL.createObjectURL(blob);

    const exportFileDefaultName = 'clusters.json';

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', objectUrl);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    URL.revokeObjectURL(objectUrl);
}

/**
 * Get current cluster data (for external access)
 * @returns {Object|null} Cluster data
 */
export function getClusterData() {
    return clusterData;
}

/**
 * Search for papers within distance of a custom topic
 * @async
 */
export async function searchCustomCluster() {
    const queryInput = document.getElementById('custom-query-input');
    const distanceInput = document.getElementById('custom-query-distance');
    const searchBtn = document.getElementById('search-custom-btn');
    
    const query = queryInput.value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    const distance = parseFloat(distanceInput.value);
    if (isNaN(distance) || distance <= 0) {
        alert('Please enter a valid distance value');
        return;
    }
    
    // Disable button and show loading
    searchBtn.disabled = true;
    searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';
    
    try {
        // Get selected filters from UI
        const yearSelect = document.getElementById('year-selector');
        const conferenceSelect = document.getElementById('conference-selector');
        const selectedYear = yearSelect ? yearSelect.value : '';
        const selectedConference = conferenceSelect ? conferenceSelect.value : '';
        
        // Build request body with optional filters
        const requestBody = { query, distance };
        
        // Add year filter if selected
        if (selectedYear) {
            requestBody.years = [parseInt(selectedYear)];
        }
        
        // Add conference filter if selected
        if (selectedConference) {
            requestBody.conferences = [selectedConference];
        }

        const selectedLocation = getSelectedLocation();
        if (selectedLocation) {
            requestBody.locations = [selectedLocation];
        }
        
        const response = await fetch(`${API_BASE}/api/clusters/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        const data = await parseJsonResponse(response);
        
        // Generate unique ID using timestamp and random component
        const uniqueId = `custom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Add custom cluster to state
        const customCluster = {
            id: uniqueId,
            query: data.query,
            distance: data.distance,
            papers: data.papers,
            count: data.count,
            queryEmbedding: data.query_embedding
        };
        
        customQueryClusters.push(customCluster);
        customClusterMode = true;
        _setHierarchyControlsVisible(false);
        
        // Initialize visibility to true (visible by default)
        customClusterVisibility[uniqueId] = true;
        
        // Re-visualize in custom cluster mode
        visualizeClustersWithCustomQueries();
        
        console.log(`Found ${data.count} papers within distance ${distance} for query: "${query}"`);
        
        // Fetch and display topic evolution for this custom topic
        fetchAndDisplayTopicEvolution(query, distance, selectedConference);
        
    } catch (error) {
        console.error('Error searching custom cluster:', error);
        alert(error.message || 'Could not run that search. Please try again.');
    } finally {
        // Re-enable button
        searchBtn.disabled = false;
        searchBtn.innerHTML = '<i class="fas fa-search mr-2"></i>Search';
    }
}

/**
 * Visualize clusters with custom queries in custom cluster mode
 * In this mode:
 * - All non-matching papers are colored blue
 * - Each custom query's papers get a unique color
 * - Query terms are shown as cluster centers (stars)
 * - Normal legend is replaced with custom query legend
 */
function visualizeClustersWithCustomQueries() {
    if (!clusterData || !clusterData.points) {
        console.error('No cluster data to visualize');
        return;
    }
    _setHierarchyControlsVisible(false);
    
    const points = clusterData.points;
    const traces = [];
    
    // Build a set of all paper IDs that match any custom query
    const matchingPaperIds = new Set();
    customQueryClusters.forEach(cluster => {
        cluster.papers.forEach(paper => {
            // Use uid for matching
            matchingPaperIds.add(paper.uid);
        });
    });
    
    // Create a map from paper id to point
    const paperIdToPoint = {};
    points.forEach(point => {
        paperIdToPoint[point.id] = point;
    });
    
    // Collect all non-matching papers (will be colored blue)
    const nonMatchingPoints = points.filter(p => !matchingPaperIds.has(p.id));
    
    // Add trace for non-matching papers (blue)
    if (nonMatchingPoints.length > 0) {
        traces.push({
            x: nonMatchingPoints.map(p => p.x),
            y: nonMatchingPoints.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: 'Other papers',
            text: nonMatchingPoints.map(p => p.title || p.id),
            customdata: nonMatchingPoints.map(p => ({
                id: p.id,
                title: p.title || '',
                year: p.year || '',
                conference: p.conference || ''
            })),
            marker: {
                color: '#3B82F6',  // Blue color
                size: 8,
                opacity: 0.5,
                line: {
                    color: 'white',
                    width: 0.5
                }
            },
            hovertemplate: '<b>%{text}</b><br>' +
                          'Year: %{customdata.year}<br>' +
                          'Conference: %{customdata.conference}<br>' +
                          '<extra></extra>',
            showlegend: false
        });
    }
    
    // Add traces for each custom query cluster with unique colors
    customQueryClusters.forEach((cluster, idx) => {
        // Skip blue (first color) - start from index 1 to avoid confusion with background
        const colorIndex = (idx + 1) % PLOTLY_COLORS.length;
        const clusterColor = PLOTLY_COLORS[colorIndex];
        
        // Get the points for papers in this cluster
        const clusterPoints = cluster.papers
            .map(paper => paperIdToPoint[paper.uid])
            .filter(p => p !== undefined);
        
        if (clusterPoints.length === 0) {
            console.warn(`No points found for custom cluster: ${cluster.query}`);
            return;
        }
        
        // Add trace for cluster papers
        traces.push({
            x: clusterPoints.map(p => p.x),
            y: clusterPoints.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: `${cluster.query} (${cluster.count})`,
            text: clusterPoints.map(p => p.title || p.id),
            customdata: clusterPoints.map(p => ({
                id: p.id,
                title: p.title || '',
                year: p.year || '',
                conference: p.conference || '',
                distance: cluster.papers.find(cp => cp.uid === p.id)?.distance || 0
            })),
            marker: {
                color: clusterColor,
                size: 10,
                opacity: 0.8,
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '<b>%{text}</b><br>' +
                          'Year: %{customdata.year}<br>' +
                          'Conference: %{customdata.conference}<br>' +
                          'Distance: %{customdata.distance:.3f}<br>' +
                          '<extra></extra>',
            legendgroup: cluster.id
        });
        
        // Calculate robust center position as median of cluster points
        const centerX = _median(clusterPoints.map(p => p.x));
        const centerY = _median(clusterPoints.map(p => p.y));
        
        // Add cluster center as star marker
        traces.push({
            x: [centerX],
            y: [centerY],
            mode: 'markers+text',
            type: 'scatter',
            name: `${cluster.query} (center)`,
            text: [cluster.query],
            textposition: 'top center',
            customdata: [{ cluster_id: cluster.id }],
            marker: {
                color: clusterColor,
                symbol: 'star',
                size: 20,
                opacity: 1.0,
                line: {
                    color: 'white',
                    width: 2
                }
            },
            hovertemplate: '<b>Query: %{text}</b><br>' +
                          `Papers: ${cluster.count}<br>` +
                          '<extra></extra>',
            showlegend: false,
            legendgroup: cluster.id
        });
    });
    
    // Layout configuration
    const plotColors4 = getPlotColors();
    const layout = {
        title: '',
        xaxis: {
            title: '',
            zeroline: false,
            showgrid: false,
            showticklabels: false,
            ticks: ''
        },
        yaxis: {
            title: '',
            zeroline: false,
            showgrid: false,
            showticklabels: false,
            ticks: ''
        },
        hovermode: 'closest',
        showlegend: false,  // We'll use custom legend
        plot_bgcolor: plotColors4.plot_bgcolor,
        paper_bgcolor: plotColors4.paper_bgcolor,
        font: plotColors4.font,
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hoverlabel: {
            namelength: -1,
            align: 'left'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true,
        toImageButtonOptions: {
            format: 'png',
            filename: 'custom_cluster_plot',
            height: 800,
            width: 800,
            scale: 2
        }
    };
    
    const plotDiv = document.getElementById('cluster-plot');
    Plotly.react(plotDiv, traces, layout, config);
    
    // Add click handler
    plotDiv.on('plotly_click', function(data) {
        if (!data.points || data.points.length === 0) return;
        const point = data.points[0];
        const customdata = point.customdata || {};
        const isStar = point.data && point.data.marker && point.data.marker.symbol === 'star';
        if (isStar && customdata.cluster_id) {
            _toggleCustomClusterIsolation(customdata.cluster_id);
            return;
        }
        const paperId = customdata.id || point.data.customdata?.[point.pointIndex]?.id;
        if (paperId) {
            showClusterPaperDetails(paperId);
        }
    });
    
    // Update custom legend
    updateCustomQueryLegend();
}

/**
 * Update custom legend showing only custom topics
 */
function updateCustomQueryLegend() {
    const legendDiv = document.getElementById('cluster-legend');
    if (!legendDiv) return;

    if (customQueryClusters.length === 0) {
        legendDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-sm">No custom topics</p>';
        return;
    }

    legendDiv.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'space-y-3';

    const title = document.createElement('h4');
    title.className = 'text-md font-bold text-gray-700 dark:text-gray-300 mb-3';
    title.textContent = 'Custom Topics';
    wrapper.appendChild(title);

    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'space-y-3';

    wrapper.appendChild(createLegendSearchField(query => {
        applyLegendSearch(itemsContainer, '[data-custom-topic]', query);
    }));

    let html = '';
    customQueryClusters.forEach((cluster, idx) => {
        // Skip blue (first color) - start from index 1 to avoid confusion with background
        const colorIndex = (idx + 1) % PLOTLY_COLORS.length;
        const clusterColor = PLOTLY_COLORS[colorIndex];
        const escapedQuery = escapeHtml(cluster.query);
        const searchText = escapeAttr(cluster.query);

        // Check if this cluster is visible (default: true)
        const isVisible = customClusterVisibility[cluster.id] !== false;
        const opacityClass = isVisible ? 'opacity-100' : 'opacity-50';

        html += `
            <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                data-custom-topic data-search-text="${searchText}">
                <div class="flex items-center gap-3 flex-1 cursor-pointer ${opacityClass}" onclick="toggleCustomClusterVisibility('${cluster.id}')">
                    <div class="w-4 h-4 rounded-full" style="background-color: ${clusterColor}"></div>
                    <div class="flex-1">
                        <div class="font-semibold text-sm text-gray-800 dark:text-gray-200">${escapedQuery}</div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">${cluster.count} papers (d=${cluster.distance.toFixed(2)})</div>
                    </div>
                </div>
                <button onclick="deleteCustomCluster('${cluster.id}')" 
                    class="ml-2 px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
    });

    itemsContainer.innerHTML = html;
    wrapper.appendChild(itemsContainer);
    legendDiv.appendChild(wrapper);

    applyLegendSearch(itemsContainer, '[data-custom-topic]', legendSearchQuery);
}

/**
 * Display statistics for a custom query cluster
 * @param {Object} customCluster - Custom cluster data
 */
function displayCustomQueryStats(customCluster) {
    // Statistics removed per user request
    // This function is kept for compatibility but does nothing
}

/**
 * Update visualization to highlight papers in custom cluster
 * @param {Object} customCluster - Custom cluster data
 * @deprecated This function is no longer used in custom cluster mode
 */
function updateVisualizationWithCustomCluster(customCluster) {
    // No longer used - replaced by visualizeClustersWithCustomQueries
}

/**
 * Apply the current custom-cluster visibility flags to all cluster traces.
 */
function _applyCustomClusterVisibility() {
    const plotDiv = document.getElementById('cluster-plot');
    if (!plotDiv || !plotDiv.data) return;

    const traceIndices = [];
    const visibility = [];
    plotDiv.data.forEach((trace, idx) => {
        if (trace.legendgroup) {
            traceIndices.push(idx);
            visibility.push(customClusterVisibility[trace.legendgroup] !== false);
        }
    });

    if (traceIndices.length > 0) {
        Plotly.restyle(plotDiv, { visible: visibility }, traceIndices);
    }
    updateCustomQueryLegend();
}

/**
 * Isolate a custom cluster (hide all others), toggling back to all clusters
 * when the cluster is already the only visible one.
 * @param {string} clusterId - ID of the custom cluster whose star was clicked
 */
function _toggleCustomClusterIsolation(clusterId) {
    const clusterIds = customQueryClusters.map(cluster => cluster.id);
    const visibleIds = clusterIds.filter(id => customClusterVisibility[id] !== false);
    const isIsolated = visibleIds.length === 1 && visibleIds[0] === clusterId;

    clusterIds.forEach(id => {
        customClusterVisibility[id] = isIsolated ? true : id === clusterId;
    });

    _applyCustomClusterVisibility();
}

/**
 * Toggle visibility of a custom cluster
 * @param {string} clusterId - ID of the custom cluster to toggle
 */
export function toggleCustomClusterVisibility(clusterId) {
    // Toggle visibility state
    const currentVisibility = customClusterVisibility[clusterId] !== false;
    customClusterVisibility[clusterId] = !currentVisibility;
    
    // Get the plot div
    const plotDiv = document.getElementById('cluster-plot');
    if (!plotDiv || !plotDiv.data) {
        console.warn('Plot not initialized');
        return;
    }
    
    // Find traces with matching legendgroup and toggle visibility
    const updates = {
        visible: []
    };
    
    const traceIndices = [];
    plotDiv.data.forEach((trace, idx) => {
        if (trace.legendgroup === clusterId) {
            traceIndices.push(idx);
            updates.visible.push(!currentVisibility);
        }
    });
    
    if (traceIndices.length > 0) {
        Plotly.restyle(plotDiv, updates, traceIndices);
    }
    
    // Update legend to show opacity change
    updateCustomQueryLegend();
}

/**
 * Update legend to include custom clusters with delete buttons
 * @deprecated Replaced by updateCustomQueryLegend
 */
function updateLegendWithCustomClusters() {
    // No longer used - replaced by updateCustomQueryLegend
}

/**
 * Delete a custom cluster
 * @param {string} clusterId - ID of the custom cluster to delete
 */
export async function deleteCustomCluster(clusterId) {
    // Remove from state
    const index = customQueryClusters.findIndex(c => c.id === clusterId);
    if (index === -1) {
        console.warn('Custom cluster not found:', clusterId);
        return;
    }
    
    customQueryClusters.splice(index, 1);
    
    // Clean up visibility state
    delete customClusterVisibility[clusterId];
    
    // If no more custom clusters, exit custom cluster mode
    if (customQueryClusters.length === 0) {
        customClusterMode = false;
        customClusterVisibility = {};  // Reset visibility state
        renderCurrentLevel();
    } else {
        // Re-render the visualization with remaining custom clusters
        visualizeClustersWithCustomQueries();
    }
}

/**
 * Fetch topic evolution data and add a new line to the unified topic evolution chart.
 * The first call creates the chart; subsequent calls append additional traces so that
 * all queries are visible as separate lines in the same plot.
 * @param {string} topic - Topic keywords
 * @param {number} distance - Distance threshold used for the custom topic search
 * @param {string} conference - Selected conference (may be empty)
 * @async
 */
async function fetchAndDisplayTopicEvolution(topic, distance, conference) {
    const container = document.getElementById('topic-evolution-container');
    if (!container) return;

    // Topic evolution requires a conference; skip when none is selected
    if (!conference) {
        return;
    }

    // Build request body
    const requestBody = {
        topic_keywords: topic,
        distance_threshold: distance,
        conferences: [conference]
    };

    const isFirstLoad = topicEvolutionTraces.length === 0;
    const loadingId = `${TOPIC_EVOLUTION_PLOT_ID}-loading`;

    if (isFirstLoad) {
        // Build the single chart wrapper with a loading spinner
        container.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6" id="topic-evolution-wrapper">
                <div id="${loadingId}" class="text-center text-gray-500 dark:text-gray-400 py-8">
                    <i class="fas fa-spinner fa-spin text-4xl mb-4 opacity-20"></i>
                    <p class="text-sm">Loading topic evolution for "${escapeHtml(topic)}"...</p>
                </div>
                <div id="${TOPIC_EVOLUTION_PLOT_ID}" style="width: 100%; height: 350px;"></div>
            </div>
        `;
        container.classList.remove('hidden');
    } else {
        // Append a small inline loading indicator below the existing chart
        const wrapper = document.getElementById('topic-evolution-wrapper');
        if (wrapper) {
            let loadingEl = document.getElementById(loadingId);
            if (!loadingEl) {
                loadingEl = document.createElement('div');
                loadingEl.id = loadingId;
                wrapper.appendChild(loadingEl);
            }
            loadingEl.className = 'mt-3 text-sm text-gray-500 dark:text-gray-400';
            loadingEl.innerHTML = `<i class="fas fa-spinner fa-spin mr-2 text-purple-600 opacity-70"></i>Loading topic evolution for "${escapeHtml(topic)}"...`;
        }
    }

    try {
        const response = await fetch(`${API_BASE}/api/topic-evolution`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const data = await parseJsonResponse(response);

        if (data.error) {
            throw new Error(data.error);
        }

        // Add trace(s) to the unified chart
        _addTopicEvolutionTrace(data);

    } catch (error) {
        console.error('Error fetching topic evolution:', error);
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) {
            if (isFirstLoad) {
                loadingEl.innerHTML = `
                    <div class="text-center text-red-500 py-8">
                        <i class="fas fa-exclamation-triangle text-4xl mb-4 opacity-50"></i>
                        <p class="text-sm">${escapeHtml(error.message)}</p>
                    </div>
                `;
            } else {
                loadingEl.className = 'mt-3 text-sm text-red-500';
                loadingEl.innerHTML = `<i class="fas fa-exclamation-triangle mr-1"></i>${escapeHtml(error.message)}`;
            }
        }
    }
}

/**
 * Add a new trace from topic evolution API data to the unified chart.
 * Creates the chart on the first call; uses Plotly.addTraces on subsequent calls.
 * @param {Object} data - Topic evolution data from the API
 */
function _addTopicEvolutionTrace(data) {
    const conferenceData = data.conference_data || {};
    const conferences = Object.keys(conferenceData);
    const topic = data.topic || '';

    const loadingEl = document.getElementById(`${TOPIC_EVOLUTION_PLOT_ID}-loading`);
    const plotEl = document.getElementById(TOPIC_EVOLUTION_PLOT_ID);

    const newTraces = [];
    for (const conf of conferences) {
        const cdata = conferenceData[conf] || {};
        const yearRelative = cdata.year_relative || {};
        const years = Object.keys(yearRelative).sort();
        const values = years.map(y => yearRelative[y]);

        // When a single conference is requested, label by topic; otherwise add "(conf)" suffix
        const traceName = conferences.length > 1 ? `${topic} (${conf})` : topic;
        newTraces.push({
            x: years.map(Number),
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            name: traceName,
            line: { color: TOPIC_COLORS[topicEvolutionColorIdx % TOPIC_COLORS.length], width: 2 },
            marker: { size: 6 }
        });
        topicEvolutionColorIdx++;
    }

    // Remove the loading indicator regardless of whether we got data
    if (loadingEl) {
        loadingEl.remove();
    }

    if (newTraces.length === 0) {
        if (plotEl && topicEvolutionTraces.length === 0) {
            plotEl.innerHTML = `
                <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                    <p class="text-sm">No topic evolution data available for "${escapeHtml(topic)}"</p>
                </div>
            `;
        }
        return;
    }

    const prevTraceCount = topicEvolutionTraces.length;
    topicEvolutionTraces.push(...newTraces);
    topicEvolutionTopics.push(topic);

    if (typeof Plotly === 'undefined' || !plotEl) return;

    const plotColors = getPlotColors();
    const chartTitle = `Topic Evolution: ${topicEvolutionTopics.join(', ')}`;
    const layout = {
        title: { text: chartTitle },
        xaxis: { title: { text: 'Year' }, type: 'linear', automargin: true, dtick: 1, showgrid: false, zeroline: false },
        yaxis: { title: { text: 'Percentage of Papers (%)' }, automargin: true, showgrid: false, zeroline: false },
        margin: { t: 50, b: 60, l: 80, r: 20 },
        paper_bgcolor: plotColors.paper_bgcolor,
        plot_bgcolor: plotColors.plot_bgcolor,
        font: plotColors.font,
        showlegend: topicEvolutionTraces.length > 1
    };
    const config = { responsive: true, displayModeBar: false };

    if (prevTraceCount === 0) {
        // First data: create the Plotly chart from scratch
        Plotly.newPlot(plotEl, topicEvolutionTraces, layout, config);
    } else {
        // Subsequent data: rebuild the chart atomically with all accumulated traces.
        // Using react() instead of addTraces()+relayout() avoids a Plotly rendering
        // quirk where the new trace would appear twice in the legend.
        Plotly.react(plotEl, topicEvolutionTraces, layout, config);
    }
}

/**
 * Load and display papers-per-year bar chart for the currently selected conference.
 * Highlights the currently selected year.
 * @async
 */
export async function loadPapersPerYear() {
    const plotDiv = document.getElementById('papers-per-year-plot');
    if (!plotDiv) return;

    const selectedConference = getSelectedConference();
    const selectedYears = getSelectedYears();
    const highlightYear = selectedYears.length > 0 ? selectedYears[0] : null;

    // Build query params
    let url = `${API_BASE}/api/papers-per-year`;
    if (selectedConference) {
        url += `?conference=${encodeURIComponent(selectedConference)}`;
    }

    try {
        await ensurePlotly();
        const response = await fetch(url);
        const data = await parseJsonResponse(response);
        if (data.error) {
            throw new Error(data.error);
        }

        const yearCounts = data.year_counts || {};
        const years = Object.keys(yearCounts).map(Number).sort();
        const counts = years.map(y => yearCounts[y]);

        if (years.length === 0) {
            plotDiv.innerHTML = '<div class="text-center text-gray-500 dark:text-gray-400 py-8"><p class="text-sm">No data available</p></div>';
            return;
        }

        // Build colors array: highlight the selected year
        const colors = years.map(y =>
            y === highlightYear ? '#7c3aed' : '#c4b5fd'
        );

        const traces = [{
            x: years,
            y: counts,
            type: 'scatter',
            mode: 'markers',
            marker: { color: colors, size: 12, line: { width: 2, color: '#7c3aed' }, symbol: 'diamond' },
            hovertemplate: '<b>%{x}</b><br>Papers: %{y}<extra></extra>'
        }];

        const confTitle = selectedConference || 'All Conferences';
        const plotColors6 = getPlotColors();
        const layout = {
            title: { text: `Total Papers Per Year — ${confTitle}` },
            xaxis: { title: { text: 'Year' }, type: 'linear', dtick: 1, automargin: true, showgrid: false, zeroline: false },
            yaxis: { title: { text: 'Number of Papers' }, automargin: true, showgrid: false, zeroline: false },
            margin: { t: 50, b: 50, l: 70, r: 20 },
            paper_bgcolor: plotColors6.paper_bgcolor,
            plot_bgcolor: plotColors6.plot_bgcolor,
            font: plotColors6.font,
            showlegend: false,
            bargap: 0.2
        };

        plotDiv.innerHTML = '';
        Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: false });

    } catch (error) {
        console.error('Error loading papers per year:', error);
        plotDiv.innerHTML = `<div class="text-center text-red-500 py-8"><p class="text-sm">Failed to load: ${escapeHtml(error.message)}</p></div>`;
    }
}

// Make custom cluster functions globally accessible for onclick handlers
if (typeof window !== 'undefined') {
    window.searchCustomCluster = searchCustomCluster;
    window.deleteCustomCluster = deleteCustomCluster;
    window.toggleCustomClusterVisibility = toggleCustomClusterVisibility;
}
