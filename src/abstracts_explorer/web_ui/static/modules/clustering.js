/**
 * Clustering Module
 * 
 * Handles cluster visualization, analysis, and management using Plotly.
 */

import { API_BASE, PLOTLY_COLORS } from './utils/constants.js';
import { showLoading, showErrorInElement } from './utils/ui-utils.js';
import { sortClustersBySizeDesc } from './utils/sort-utils.js';
import { getClusterLabelWithCount } from './utils/cluster-utils.js';
import { getSelectedConference, getSelectedYears, escapeHtml } from './utils/dom-utils.js';
import { formatPaperCard } from './paper-card.js';
import { renderInlineMarkdownWithLatex } from './utils/markdown-utils.js';

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
    reduction_method: 'tsne',
    n_components: 2,
    clustering_method: 'agglomerative',
    distance_threshold: 150,
    linkage: 'ward',
    use_llm_labels: true
};
// Track selected clusters for multi-select
let selectedClusters = new Set();

// Hierarchical clustering state
let hierarchyMode = false;
let currentHierarchyLevel = 0;
let maxHierarchyLevel = 0;
let currentParentId = null;

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
    // Reset topic evolution state
    topicEvolutionTraces = [];
    topicEvolutionColorIdx = 0;
    topicEvolutionTopics = [];
}

/**
 * Load and visualize clusters
 * @async
 */
export async function loadClusters() {
    try {
        showLoading('cluster-plot', 'Loading clusters...');

        // Build request body with conference/year filters
        const computeBody = {};
        const selectedConference = getSelectedConference();
        const selectedYears = getSelectedYears();
        if (selectedConference) computeBody.conferences = [selectedConference];
        if (selectedYears.length > 0) computeBody.years = selectedYears;
        
        // Fetch pre-computed clustering data
        const response = await fetch(`${API_BASE}/api/clusters/compute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(computeBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        clusterData = await response.json();
        
        if (clusterData.error) {
            showErrorInElement('cluster-plot', clusterData.error);
            return;
        }
        
        // Initialize selected clusters (start with all selected)
        selectedClusters.clear();
        
        // Create visualization
        visualizeClusters();
        
    } catch (error) {
        console.error('Error loading clusters:', error);
        showErrorInElement('cluster-plot', `Failed to load clusters: ${error.message}`);
    }
}

/**
 * Enable hierarchical clustering mode
 * @async
 */
export async function enableHierarchyMode() {
    if (!clusterData || !clusterData.cluster_hierarchy || !clusterData.cluster_hierarchy.tree) {
        console.warn('Hierarchy not available for current clustering');
        alert('Hierarchical view is only available for agglomerative clustering. Please run agglomerative clustering first.');
        return;
    }
    
    const tree = clusterData.cluster_hierarchy.tree;
    if (!tree.nodes || Object.keys(tree.nodes).length === 0) {
        console.error('Invalid hierarchy tree: no nodes found');
        alert('Invalid hierarchy data. Please re-run clustering.');
        return;
    }
    
    hierarchyMode = true;
    currentHierarchyLevel = tree?.max_level || 0;  // Start at top level
    maxHierarchyLevel = tree?.max_level || 0;
    currentParentId = null;
    
    // Build level data from local cluster hierarchy
    const levelData = buildLevelDataFromHierarchy(currentHierarchyLevel, null);
    
    if (levelData.clusters.length === 0) {
        console.error('No clusters found at level', currentHierarchyLevel);
        alert('No clusters found at the selected hierarchy level.');
        hierarchyMode = false;
        return;
    }
    
    // Visualize with hierarchy
    visualizeHierarchyLevel(levelData);
    
    // Create hierarchy legend
    createHierarchyLegend(levelData.clusters);
}

/**
 * Build level data from local hierarchy structure
 * @param {number} level - Hierarchy level to build
 * @param {number|null} parentId - Optional parent node ID to filter by
 * @returns {Object} Level data structure
 */
function buildLevelDataFromHierarchy(level, parentId = null) {
    const tree = clusterData.cluster_hierarchy.tree;
    if (!tree || !tree.nodes) {
        console.error('Invalid hierarchy tree structure:', tree);
        return { clusters: [], level: 0, max_level: 0, labels: {} };
    }
    
    // Get all nodes at this level by filtering through all nodes
    const nodesAtLevel = [];
    for (const [nodeId, nodeInfo] of Object.entries(tree.nodes)) {
        if (nodeInfo.level === level) {
            nodesAtLevel.push(parseInt(nodeId));
        }
    }
    
    // Filter by parent if specified
    let filteredNodes = nodesAtLevel;
    if (parentId !== null && tree.nodes[parentId]) {
        const parentChildren = tree.nodes[parentId].children || [];
        filteredNodes = nodesAtLevel.filter(nodeId => parentChildren.includes(nodeId));
    }
    
    // Build cluster info for each node
    const clusters = filteredNodes.map(nodeId => {
        const node = tree.nodes[nodeId];
        return {
            cluster_id: nodeId,
            node_id: nodeId,
            label: node.label || `Cluster ${nodeId}`,
            size: node.samples ? node.samples.length : 0,
            samples: node.samples || [],
            is_leaf: node.is_leaf || false,
            has_children: (node.children && node.children.length > 0) || false
        };
    });
    
    return {
        clusters: clusters,
        level: level,
        max_level: tree.max_level || 0,
        labels: {} // Labels are already in cluster objects
    };
}

/**
 * Disable hierarchical clustering mode
 */
export function disableHierarchyMode() {
    hierarchyMode = false;
    currentHierarchyLevel = 0;
    currentParentId = null;
    
    // Re-visualize normal view
    visualizeClusters();
}

/**
 * Load and visualize a specific hierarchy level
 * @async
 * @param {number} level - Hierarchy level to load
 * @param {number|null} parentId - Optional parent cluster ID to filter by
 */
export async function loadHierarchyLevel(level, parentId = null) {
    if (!hierarchyMode) {
        console.warn('Not in hierarchy mode');
        return;
    }
    
    try {
        showLoading('cluster-plot', `Loading hierarchy level ${level}...`);
        
        // Build level data from local hierarchy structure
        const levelData = buildLevelDataFromHierarchy(level, parentId);
        
        // Update state
        currentHierarchyLevel = levelData.level;
        maxHierarchyLevel = levelData.max_level;
        currentParentId = parentId;
        
        // Visualize this level
        visualizeHierarchyLevel(levelData);
        
        // Update legend
        createHierarchyLegend(levelData.clusters);
        
    } catch (error) {
        console.error('Error loading hierarchy level:', error);
        showErrorInElement('cluster-plot', `Failed to load hierarchy level: ${error.message}`);
    }
}

/**
 * Navigate up in hierarchy (more merged clusters)
 * @async
 */
export async function navigateHierarchyUp() {
    if (currentHierarchyLevel < maxHierarchyLevel) {
        await loadHierarchyLevel(currentHierarchyLevel + 1, null);
    }
}

/**
 * Navigate down in hierarchy (more detailed clusters)
 * @async
 */
export async function navigateHierarchyDown() {
    if (currentHierarchyLevel > 0) {
        await loadHierarchyLevel(currentHierarchyLevel - 1, currentParentId);
    }
}

/**
 * Visualize hierarchy level data
 * @param {Object} levelData - Level data from API
 */
function visualizeHierarchyLevel(levelData) {
    const traces = [];
    const clusters = levelData.clusters || [];
    
    // For each cluster at this level, create a trace showing:
    // - A center marker (star) for the cluster
    // - All member abstracts as smaller points
    clusters.forEach((cluster, idx) => {
        const clusterColor = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        const label = cluster.label || `Cluster ${cluster.cluster_id}`;
        
        // Get points for all samples in this cluster
        // samples now contains paper IDs (not indices), so we can directly compare
        const samples = cluster.samples || [];
        const clusterPoints = clusterData.points.filter(p => 
            samples.includes(p.id)
        );
        
        if (clusterPoints.length > 0) {
            // Calculate center point as mean of all points
            const centerX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
            const centerY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
            
            // Trace for cluster center (star marker)
            const centerTrace = {
                x: [centerX],
                y: [centerY],
                mode: 'markers',  // Changed from 'markers+text' to show label only on hover
                type: 'scatter',
                name: `${label} (${cluster.size})`,
                text: [label],
                customdata: [{
                    cluster_id: cluster.cluster_id,
                    node_id: cluster.node_id,
                    has_children: !cluster.is_leaf,
                    size: cluster.size
                }],
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
                hovertemplate: '<b>%{text}</b><br>' +
                              `${cluster.size} papers<br>` +
                              (cluster.is_leaf ? '' : 'Click to drill down<br>') +
                              '<extra></extra>',
                legendgroup: `cluster-${cluster.cluster_id}`,
                showlegend: false  // We'll use custom legend
            };
            traces.push(centerTrace);
            
            // Trace for all abstracts in this cluster (smaller markers)
            const abstractTrace = {
                x: clusterPoints.map(p => p.x),
                y: clusterPoints.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: label,
                text: clusterPoints.map(p => p.title || p.id),
                customdata: clusterPoints.map(p => ({
                    id: p.id,
                    title: p.title || '',
                    cluster_id: cluster.cluster_id,
                    node_id: cluster.node_id
                })),
                marker: {
                    color: clusterColor,
                    size: 6,
                    opacity: 0.5,
                    line: {
                        color: 'white',
                        width: 0.5
                    }
                },
                hovertemplate: '<b>%{text}</b><br>' +
                              '<extra></extra>',
                legendgroup: `cluster-${cluster.cluster_id}`,
                showlegend: false
            };
            traces.push(abstractTrace);
        }
    });
    
    const plotColors = getPlotColors();
    const layout = {
        title: '',
        hovermode: 'closest',
        showlegend: false,  // Use custom legend
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
    
    // Clear and create plot
    const plotElement = document.getElementById('cluster-plot');
    plotElement.innerHTML = '';
    
    Plotly.newPlot('cluster-plot', traces, layout, config).then(function() {
        Plotly.relayout('cluster-plot', {
            'xaxis.fixedrange': false,
            'yaxis.fixedrange': false
        });
    });
    
    // Create custom legend with hierarchy controls
    createHierarchyLegend(clusters);
    
    // Add click handler for drilling down
    const plotDiv = document.getElementById('cluster-plot');
    if (plotDiv) {
        plotDiv.on('plotly_click', async function(data) {
            if (hierarchyMode && data.points && data.points.length > 0) {
                const point = data.points[0];
                const hasChildren = point.customdata?.has_children;
                const nodeId = point.customdata?.node_id;
                
                // Only drill down if clicking on a center (star) marker and it has children
                if (hasChildren && nodeId !== undefined && point.data.marker.symbol === 'star') {
                    if (currentHierarchyLevel > 0) {
                        // Drill down to children of this cluster
                        await loadHierarchyLevel(currentHierarchyLevel - 1, nodeId);
                    }
                }
            }
        });
    }
}

/**
 * Visualize clusters using Plotly
 */
export function visualizeClusters() {
    if (!clusterData || !clusterData.points) {
        console.error('No cluster data to visualize');
        return;
    }
    
    const points = clusterData.points;
    const centers = clusterData.cluster_centers || {};
    
    // Group points by cluster
    const clusterGroups = {};
    points.forEach(point => {
        const cluster = point.cluster;
        if (!clusterGroups[cluster]) {
            clusterGroups[cluster] = [];
        }
        clusterGroups[cluster].push(point);
    });
    
    // Sort clusters by size (descending), then by ID (ascending) as tiebreaker
    const sortedClusterEntries = sortClustersBySizeDesc(Object.entries(clusterGroups));
    
    // Initialize selected clusters with all clusters
    selectedClusters.clear();
    
    // Create traces for each cluster
    const labels = clusterData.cluster_labels || {};
    const traces = [];
    
    // For each cluster, create both the points trace and center trace with matching colors
    sortedClusterEntries.forEach(([clusterId, clusterPoints], idx) => {
        const paperCount = clusterPoints.length;
        
        // Assign explicit color from Plotly's default palette
        const clusterColor = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        
        // Get the label with count for this cluster
        const label = getClusterLabelWithCount(clusterId, labels, paperCount);
        
        // Main cluster points trace
        const pointsTrace = {
            x: clusterPoints.map(p => p.x),
            y: clusterPoints.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: label,
            text: clusterPoints.map(p => p.title || p.id),
            customdata: clusterPoints.map(p => ({
                id: p.id,
                title: p.title || '',
                year: p.year || '',
                conference: p.conference || '',
                session: p.session || ''
            })),
            marker: {
                color: clusterColor,
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
            legendgroup: `cluster-${clusterId}`
        };
        traces.push(pointsTrace);
        
        // Add cluster center as star marker with same color
        const center = centers[clusterId];
        if (center) {
            const centerTrace = {
                x: [center.x],
                y: [center.y],
                mode: 'markers',
                type: 'scatter',
                name: `${label} (center)`,
                marker: {
                    color: clusterColor,
                    symbol: 'star',
                    size: 16,
                    opacity: 1.0,
                    line: {
                        color: 'white',
                        width: 2
                    }
                },
                hovertemplate: '<b>Cluster Center</b><br>' +
                              label + '<br>' +
                              '<extra></extra>',
                showlegend: false,
                legendgroup: `cluster-${clusterId}`
            };
            traces.push(centerTrace);
        }
    });
    
    // Layout configuration
    const plotColors = getPlotColors();
    const layout = {
        title: '',  // No title in plot
        xaxis: {
            title: '',  // Remove axis label
            zeroline: false,
            showgrid: false,  // Remove grid
            showticklabels: false,  // Remove tick labels
            ticks: ''  // Remove ticks
        },
        yaxis: {
            title: '',  // Remove axis label
            zeroline: false,
            showgrid: false,  // Remove grid
            showticklabels: false,  // Remove tick labels
            ticks: ''  // Remove ticks
        },
        hovermode: 'closest',
        showlegend: false,  // Hide built-in legend, we'll create custom one
        plot_bgcolor: plotColors.plot_bgcolor,
        paper_bgcolor: plotColors.paper_bgcolor,
        font: plotColors.font,
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50  // Standard bottom margin
        },
        hoverlabel: {
            namelength: -1,
            align: 'left'
        }
    };
    
    // Config
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
        scrollZoom: true
    };
    
    // Clear the loading spinner before creating the plot
    const plotElement = document.getElementById('cluster-plot');
    plotElement.innerHTML = '';
    
    // Create plot
    Plotly.newPlot('cluster-plot', traces, layout, config).then(function() {
        Plotly.relayout('cluster-plot', {
            'xaxis.fixedrange': false,
            'yaxis.fixedrange': false
        });
    });
    
    // Create custom legend in separate container
    createCustomLegend(sortedClusterEntries, labels);
    
    // Add click handler for point selection
    document.getElementById('cluster-plot').on('plotly_click', function(data) {
        const point = data.points[0];
        const customdata = point.customdata;
        showClusterPaperDetails(customdata.id, customdata);
    });
}

/**
 * Create custom legend with hierarchy controls for hierarchical mode
 * @param {Array} clusters - Array of cluster objects from API
 */
/**
 * Create a simple dendrogram visualization
 * @returns {HTMLElement} Dendrogram SVG container
 */
function createDendrogram() {
    const container = document.createElement('div');
    container.className = 'mb-3 pb-3 border-b border-gray-200 dark:border-gray-700';
    
    // Get hierarchy tree info
    if (!clusterData || !clusterData.cluster_hierarchy || !clusterData.cluster_hierarchy.dendrogram) {
        return container;
    }
    
    const dendrogram = clusterData.cluster_hierarchy.dendrogram;
    const tree = clusterData.cluster_hierarchy.tree;
    const icoord = dendrogram.icoord;
    const dcoord = dendrogram.dcoord;
    const n_samples = clusterData.cluster_hierarchy.n_samples || 0;
    
    if (!icoord || !dcoord || icoord.length === 0 || !tree || !tree.nodes) {
        return container;
    }
    
    // Build mapping from merge index to level
    // Merge index i corresponds to node (n_samples + i) in the tree
    const mergeLevels = [];
    for (let i = 0; i < icoord.length; i++) {
        const nodeId = n_samples + i;
        const nodeInfo = tree.nodes[nodeId];
        const level = nodeInfo ? nodeInfo.level : i + 1;
        mergeLevels.push(level);
    }
    
    // Filter merges to only show levels >= 5
    const visibleMerges = [];
    const visibleIcoord = [];
    const visibleDcoord = [];
    
    for (let i = 0; i < icoord.length; i++) {
        if (mergeLevels[i] >= 5) {
            visibleMerges.push(i);
            visibleIcoord.push(icoord[i]);
            visibleDcoord.push(dcoord[i]);
        }
    }
    
    // If no visible merges, show a message
    if (visibleMerges.length === 0) {
        const message = document.createElement('p');
        message.className = 'text-xs text-gray-500 dark:text-gray-400 text-center py-2';
        message.textContent = 'Dendrogram shown for levels ≥ 5';
        container.appendChild(message);
        return container;
    }
    
    // Find min/max for scaling (only visible merges)
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (const coords of visibleIcoord) {
        for (const x of coords) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
        }
    }
    
    for (const coords of visibleDcoord) {
        for (const y of coords) {
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
    }
    
    // Add padding
    const padding = 10;
    const width = 200;
    const height = 100;
    
    // Create SVG dendrogram
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.style.display = "block";
    
    // Scaling functions
    const scaleX = (x) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding);
    const scaleY = (y) => height - padding - ((y - minY) / (maxY - minY)) * (height - 2 * padding);
    
    // Draw each visible merge as a path
    for (let idx = 0; idx < visibleMerges.length; idx++) {
        const mergeIndex = visibleMerges[idx];
        const level = mergeLevels[mergeIndex];
        const xCoords = visibleIcoord[idx];
        const yCoords = visibleDcoord[idx];
        
        // Create path for this merge (U-shaped line)
        const path = document.createElementNS(svgNS, "path");
        const x1 = scaleX(xCoords[0]);
        const y1 = scaleY(yCoords[0]);
        const x2 = scaleX(xCoords[1]);
        const y2 = scaleY(yCoords[1]);
        const x3 = scaleX(xCoords[2]);
        const y3 = scaleY(yCoords[2]);
        const x4 = scaleX(xCoords[3]);
        const y4 = scaleY(yCoords[3]);
        
        // Path: vertical from child1, horizontal across, vertical to child2
        const d = `M ${x1},${y1} L ${x2},${y2} L ${x3},${y3} L ${x4},${y4}`;
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        
        // Highlight current level in purple, others in gray
        const isCurrentLevel = (currentHierarchyLevel === level);
        path.setAttribute("stroke", isCurrentLevel ? "#9333ea" : "#d1d5db");
        path.setAttribute("stroke-width", isCurrentLevel ? "2.5" : "1.5");
        path.setAttribute("opacity", isCurrentLevel ? "1.0" : "0.5");
        
        svg.appendChild(path);
    }
    
    // Add title
    const title = document.createElement('p');
    title.className = 'text-xs text-gray-600 dark:text-gray-400 mt-2 text-center';
    title.textContent = `Dendrogram (Levels ≥ 5)`;
    
    container.appendChild(svg);
    container.appendChild(title);
    
    return container;
}

function createHierarchyLegend(clusters) {
    const legendContainer = document.getElementById('cluster-legend');
    if (!legendContainer) return;
    
    // Clear existing legend
    legendContainer.innerHTML = '';
    
    // Add dendrogram at the top
    const dendrogram = createDendrogram();
    legendContainer.appendChild(dendrogram);
    
    // Create legend header with hierarchy controls
    const header = document.createElement('div');
    header.className = 'mb-3 pb-3 border-b border-gray-200 dark:border-gray-700';
    
    const title = document.createElement('h4');
    title.className = 'text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3';
    title.innerHTML = '🔍 Hierarchical View';
    header.appendChild(title);
    
    // Level navigation controls
    const levelNav = document.createElement('div');
    levelNav.className = 'flex items-center gap-2 mb-2';
    
    const levelUpBtn = document.createElement('button');
    levelUpBtn.className = 'px-2 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed';
    levelUpBtn.textContent = '↑ Up';
    levelUpBtn.disabled = currentHierarchyLevel >= maxHierarchyLevel;
    levelUpBtn.addEventListener('click', navigateHierarchyUp);
    
    const levelDisplay = document.createElement('span');
    levelDisplay.className = 'px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded flex-1 text-center text-gray-800 dark:text-gray-200';
    levelDisplay.textContent = `Level ${currentHierarchyLevel} / ${maxHierarchyLevel}`;
    
    const levelDownBtn = document.createElement('button');
    levelDownBtn.className = 'px-2 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed';
    levelDownBtn.textContent = '↓ Down';
    levelDownBtn.disabled = currentHierarchyLevel <= 0;
    levelDownBtn.addEventListener('click', navigateHierarchyDown);
    
    levelNav.appendChild(levelUpBtn);
    levelNav.appendChild(levelDisplay);
    levelNav.appendChild(levelDownBtn);
    header.appendChild(levelNav);
    
    // Exit button
    const exitBtn = document.createElement('button');
    exitBtn.className = 'w-full px-2 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700 mt-2';
    exitBtn.textContent = 'Exit Hierarchy Mode';
    exitBtn.addEventListener('click', disableHierarchyMode);
    header.appendChild(exitBtn);
    
    // Info text
    const infoText = document.createElement('p');
    infoText.className = 'text-xs text-gray-600 dark:text-gray-400 mt-2';
    infoText.textContent = 'Click on cluster centers (★) to drill down';
    header.appendChild(infoText);
    
    legendContainer.appendChild(header);
    
    // Create legend items container with scrolling
    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'space-y-1 overflow-y-auto pr-2 flex-1';
    itemsContainer.style.minHeight = '0';
    itemsContainer.style.maxHeight = '100%';
    
    clusters.forEach((cluster, idx) => {
        const clusterColor = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        const label = cluster.label || `Cluster ${cluster.cluster_id}`;
        
        // Create legend item
        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors';
        item.style.backgroundColor = 'rgb(249 250 251)';
        
        // Color box
        const colorBox = document.createElement('div');
        colorBox.className = 'w-4 h-4 rounded flex-shrink-0';
        colorBox.style.backgroundColor = clusterColor;
        
        // Label text
        const labelText = document.createElement('span');
        labelText.className = 'text-sm text-gray-700 dark:text-gray-300 flex-1';
        labelText.textContent = `${label} (${cluster.size})`;
        
        item.appendChild(colorBox);
        item.appendChild(labelText);
        
        if (!cluster.is_leaf) {
            const drillIcon = document.createElement('span');
            drillIcon.className = 'text-xs text-gray-500 dark:text-gray-400';
            drillIcon.textContent = '▼';
            item.appendChild(drillIcon);
            
            // Add click handler to drill down
            item.addEventListener('click', () => {
                drillDownToCluster(cluster.cluster_id);
            });
        }
        
        itemsContainer.appendChild(item);
    });
    
    legendContainer.appendChild(itemsContainer);
}

/**
 * Drill down to a specific cluster's children
 * @param {number} clusterId - Node ID to drill down into
 * @async
 */
async function drillDownToCluster(clusterId) {
    if (currentHierarchyLevel <= 0) {
        console.warn('Already at lowest level');
        return;
    }
    
    // Drill down to children of this cluster (same as clicking star marker)
    await loadHierarchyLevel(currentHierarchyLevel - 1, clusterId);
}

/**
 * Add hierarchy mode button if agglomerative clustering with hierarchy is available
 */
/**
 * Format cluster statistics as HTML string for legend title
 * @param {Object} statistics - Cluster statistics object with total_papers, n_clusters, n_noise
 * @param {Object} labels - Cluster labels object
 * @returns {string} Formatted HTML string with cluster statistics
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
 * Create custom legend with multi-select support
 * @param {Array} sortedClusterEntries - Array of [clusterId, points[]] entries
 * @param {Object} labels - Cluster labels object
 */
function createCustomLegend(sortedClusterEntries, labels) {
    const legendContainer = document.getElementById('cluster-legend');
    if (!legendContainer) return;
    
    // Clear existing legend
    legendContainer.innerHTML = '';
    
    // Create legend header with title and action buttons
    const header = document.createElement('div');
    header.className = 'mb-3';
    
    const title = document.createElement('h4');
    title.className = 'text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2';
    
    // Build dynamic title with stats
    const titleHTML = formatClusterStats(clusterData?.statistics, labels);
    title.innerHTML = titleHTML;
    header.appendChild(title);
    
    // Create button container
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'flex gap-2';
    
    // "Select All" button
    const selectAllBtn = document.createElement('button');
    selectAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors';
    selectAllBtn.textContent = 'All';
    selectAllBtn.addEventListener('click', () => {
        selectedClusters.clear();
        sortedClusterEntries.forEach(([clusterId]) => {
            selectedClusters.add(String(clusterId));
        });
        updateClusterVisualization();
    });
    
    // "Clear All" button
    const clearAllBtn = document.createElement('button');
    clearAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors';
    clearAllBtn.textContent = 'None';
    clearAllBtn.addEventListener('click', () => {
        selectedClusters.clear();
        updateClusterVisualization();
    });
    
    buttonContainer.appendChild(selectAllBtn);
    buttonContainer.appendChild(clearAllBtn);
    
    // Add hierarchy mode button if applicable (only for agglomerative clustering with hierarchy)
    if (clusterData && clusterData.cluster_hierarchy && clusterData.cluster_hierarchy.tree) {
        const hierarchyBtn = document.createElement('button');
        hierarchyBtn.className = 'px-2 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors';
        hierarchyBtn.textContent = '⊞ Hierarchy';
        hierarchyBtn.title = 'Enable hierarchical view to explore cluster levels';
        hierarchyBtn.addEventListener('click', enableHierarchyMode);
        buttonContainer.appendChild(hierarchyBtn);
    }
    
    header.appendChild(buttonContainer);
    
    legendContainer.appendChild(header);
    
    // Create legend items container with scrolling
    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'space-y-1 overflow-y-auto pr-2 flex-1';
    itemsContainer.style.minHeight = '0'; // Allow flex child to shrink
    itemsContainer.style.maxHeight = '100%'; // Ensure scrolling works in grid
    
    sortedClusterEntries.forEach(([clusterId, clusterPoints], idx) => {
        const paperCount = clusterPoints.length;
        const clusterColor = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        const label = getClusterLabelWithCount(clusterId, labels, paperCount);
        const isSelected = selectedClusters.has(String(clusterId));
        
        // Create legend item
        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 p-2 rounded cursor-pointer transition-all';
        item.style.opacity = isSelected ? '1' : '0.4';
        item.style.backgroundColor = isSelected ? 'rgb(249 250 251)' : 'transparent';
        item.title = `Click to ${isSelected ? 'deselect' : 'select'} ${label}`;
        item.dataset.clusterId = clusterId;
        
        // Color box
        const colorBox = document.createElement('div');
        colorBox.className = 'w-4 h-4 rounded flex-shrink-0';
        colorBox.style.backgroundColor = clusterColor;
        
        // Label text
        const labelText = document.createElement('span');
        labelText.className = 'text-sm text-gray-700 dark:text-gray-300 flex-1';
        labelText.textContent = label;
        
        item.appendChild(colorBox);
        item.appendChild(labelText);
        
        // Add click handler for multi-select
        item.addEventListener('click', () => {
            const clusterIdStr = String(clusterId);
            if (selectedClusters.has(clusterIdStr)) {
                selectedClusters.delete(clusterIdStr);
            } else {
                selectedClusters.add(clusterIdStr);
            }
            updateClusterVisualization();
        });
        
        // Hover effect
        item.addEventListener('mouseenter', () => {
            if (!selectedClusters.has(String(clusterId))) {
                item.style.backgroundColor = 'rgb(243 244 246)';
            }
        });
        item.addEventListener('mouseleave', () => {
            if (!selectedClusters.has(String(clusterId))) {
                item.style.backgroundColor = 'transparent';
            }
        });
        
        itemsContainer.appendChild(item);
    });
    
    legendContainer.appendChild(itemsContainer);
}

/**
 * Update cluster visualization based on selected clusters
 */
function updateClusterVisualization() {
    if (!clusterData || !clusterData.points) return;
    
    const points = clusterData.points;
    const centers = clusterData.cluster_centers || {};
    const labels = clusterData.cluster_labels || {};
    
    // Group points by cluster
    const clusterGroups = {};
    points.forEach(point => {
        const cluster = point.cluster;
        if (!clusterGroups[cluster]) {
            clusterGroups[cluster] = [];
        }
        clusterGroups[cluster].push(point);
    });
    
    // Sort clusters
    const sortedClusterEntries = sortClustersBySizeDesc(Object.entries(clusterGroups));
    
    // If no clusters selected, show all
    const clustersToShow = selectedClusters.size === 0 
        ? new Set(sortedClusterEntries.map(([id]) => String(id)))
        : selectedClusters;
    
    // Create traces for selected clusters
    const traces = [];
    sortedClusterEntries.forEach(([clusterId, clusterPoints], idx) => {
        const clusterIdStr = String(clusterId);
        if (!clustersToShow.has(clusterIdStr)) return;
        
        const paperCount = clusterPoints.length;
        const clusterColor = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        const label = getClusterLabelWithCount(clusterId, labels, paperCount);
        
        const trace = {
            x: clusterPoints.map(p => p.x),
            y: clusterPoints.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: label,
            text: clusterPoints.map(p => p.title || p.id),
            customdata: clusterPoints.map(p => ({
                id: p.id,
                title: p.title || '',
                year: p.year || '',
                conference: p.conference || '',
                session: p.session || ''
            })),
            marker: {
                color: clusterColor,
                size: 8,
                opacity: 0.7,
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '<b>%{text}</b><br>' +
                          'Cluster: ' + label + '<br>' +
                          'Year: %{customdata.year}<br>' +
                          'Conference: %{customdata.conference}<br>' +
                          '<extra></extra>'
        };
        
        traces.push(trace);
        
        // Add cluster center if available
        const center = centers[clusterId];
        if (center) {
            const centerTrace = {
                x: [center.x],
                y: [center.y],
                mode: 'markers',
                type: 'scatter',
                name: 'Center',
                marker: {
                    color: clusterColor,
                    symbol: 'star',
                    size: 15,
                    opacity: 1.0,
                    line: {
                        color: 'white',
                        width: 2
                    }
                },
                hovertemplate: '<b>Cluster Center</b><br>' +
                              label + '<br>' +
                              '<extra></extra>',
                showlegend: false
            };
            traces.push(centerTrace);
        }
    });
    
    // Update plot with new data
    const plotColors3 = getPlotColors();
    const layout = {
        title: selectedClusters.size > 0 && selectedClusters.size < sortedClusterEntries.length
            ? `Clusters (${selectedClusters.size} selected)`
            : 'All Clusters',
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
        plot_bgcolor: plotColors3.plot_bgcolor,
        paper_bgcolor: plotColors3.paper_bgcolor,
        font: plotColors3.font,
        margin: { l: 50, r: 50, t: 50, b: 50 },
        hoverlabel: {
            namelength: -1,
            align: 'left'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        scrollZoom: true
    };
    
    // Clear and recreate plot
    const plotElement = document.getElementById('cluster-plot');
    plotElement.innerHTML = '';
    
    Plotly.newPlot('cluster-plot', traces, layout, config).then(function() {
        Plotly.relayout('cluster-plot', {
            'xaxis.fixedrange': false,
            'yaxis.fixedrange': false
        });
    });
    
    // Re-add click handler
    document.getElementById('cluster-plot').on('plotly_click', function(data) {
        const point = data.points[0];
        const customdata = point.customdata;
        showClusterPaperDetails(customdata.id, customdata);
    });
    
    // Update legend to reflect selection state
    const legendItems = document.querySelectorAll('#cluster-legend [data-cluster-id]');
    legendItems.forEach(item => {
        const clusterId = item.dataset.clusterId;
        const isSelected = selectedClusters.has(String(clusterId));
        item.style.opacity = isSelected ? '1' : '0.4';
        item.style.backgroundColor = isSelected ? 'rgb(249 250 251)' : 'transparent';
        item.title = `Click to ${isSelected ? 'deselect' : 'select'} ${item.querySelector('span').textContent}`;
    });
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
        if (response.ok) {
            const paper = await response.json();
            
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
        } else {
            console.error('Failed to fetch paper details:', response.status);
        }
    } catch (error) {
        console.error('Error fetching paper details:', error);
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
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'clusters.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

/**
 * Get current cluster data (for external access)
 * @returns {Object|null} Cluster data
 */
export function getClusterData() {
    return clusterData;
}

// Make hierarchy functions globally accessible for onclick handlers
if (typeof window !== 'undefined') {
    window.enableHierarchyMode = enableHierarchyMode;
    window.disableHierarchyMode = disableHierarchyMode;
    window.navigateHierarchyUp = navigateHierarchyUp;
    window.navigateHierarchyDown = navigateHierarchyDown;
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
        
        const response = await fetch(`${API_BASE}/api/clusters/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
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
        
        // Initialize visibility to true (visible by default)
        customClusterVisibility[uniqueId] = true;
        
        // Re-visualize in custom cluster mode
        visualizeClustersWithCustomQueries();
        
        console.log(`Found ${data.count} papers within distance ${distance} for query: "${query}"`);
        
        // Fetch and display topic evolution for this custom topic
        fetchAndDisplayTopicEvolution(query, distance, selectedConference);
        
    } catch (error) {
        console.error('Error searching custom cluster:', error);
        alert(`Failed to search: ${error.message}`);
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
        
        // Calculate center position as mean of cluster points
        const centerX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
        const centerY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
        
        // Add cluster center as star marker
        traces.push({
            x: [centerX],
            y: [centerY],
            mode: 'markers+text',
            type: 'scatter',
            name: `${cluster.query} (center)`,
            text: [cluster.query],
            textposition: 'top center',
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
        if (data.points.length > 0) {
            const point = data.points[0];
            const paperId = point.customdata?.id || point.data.customdata?.[point.pointIndex]?.id;
            if (paperId) {
                showClusterPaperDetails(paperId);
            }
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
    
    if (customQueryClusters.length === 0) {
        legendDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-sm">No custom topics</p>';
        return;
    }
    
    let html = '<div class="space-y-3">';
    html += '<h4 class="text-md font-bold text-gray-700 dark:text-gray-300 mb-3">Custom Topics</h4>';
    
    customQueryClusters.forEach((cluster, idx) => {
        // Skip blue (first color) - start from index 1 to avoid confusion with background
        const colorIndex = (idx + 1) % PLOTLY_COLORS.length;
        const clusterColor = PLOTLY_COLORS[colorIndex];
        const escapedQuery = escapeHtml(cluster.query);
        
        // Check if this cluster is visible (default: true)
        const isVisible = customClusterVisibility[cluster.id] !== false;
        const opacityClass = isVisible ? 'opacity-100' : 'opacity-50';
        
        html += `
            <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700">
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
    
    html += '</div>';
    legendDiv.innerHTML = html;
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
        // Reload normal clusters (also rebuilds the legend via createCustomLegend)
        visualizeClusters();
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

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

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
                        <p class="text-sm">Failed to load topic evolution: ${escapeHtml(error.message)}</p>
                    </div>
                `;
            } else {
                loadingEl.className = 'mt-3 text-sm text-red-500';
                loadingEl.innerHTML = `<i class="fas fa-exclamation-triangle mr-1"></i>Failed to load topic evolution for "${escapeHtml(topic)}": ${escapeHtml(error.message)}`;
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
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
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
