/**
 * Clustering Module
 * 
 * Handles cluster visualization, analysis, and management using Plotly.
 */

import { API_BASE, PLOTLY_COLORS } from './utils/constants.js';
import { showLoading, showErrorInElement } from './utils/ui-utils.js';
import { sortClustersBySizeDesc } from './utils/sort-utils.js';
import { getClusterLabelWithCount } from './utils/cluster-utils.js';
import { formatPaperCard } from './paper-card.js';

// Cluster state
let clusterData = null;
let currentClusterConfig = {
    reduction_method: 'tsne',
    n_components: 2,
    clustering_method: 'agglomerative',  // Default to agglomerative
    n_clusters: null,  // Empty - rely on distance_threshold
    eps: 0.5,
    min_samples: 5,
    distance_threshold: 150,  // Default distance threshold
    linkage: 'ward',
    affinity: 'rbf',
    m: 2.0,
    limit: null,
    use_llm_labels: true  // Use LLM for hierarchy labels by default
};
// Track selected clusters for multi-select
let selectedClusters = new Set();

// Hierarchical clustering state
let hierarchyMode = false;
let currentHierarchyLevel = 0;
let maxHierarchyLevel = 0;
let currentParentId = null;

// Custom query clustering state
let customQueryClusters = [];  // Array of custom query cluster objects
let customClusterMode = false;  // Whether custom cluster mode is active

/**
 * Initialize default cluster count from backend
 * @async
 */
async function initDefaultClusterCount() {
    try {
        const response = await fetch(`${API_BASE}/api/clusters/default-count`);
        if (response.ok) {
            const data = await response.json();
            // Only set n_clusters if not using agglomerative with distance_threshold
            if (currentClusterConfig.n_clusters === null && 
                !(currentClusterConfig.clustering_method === 'agglomerative' && 
                  currentClusterConfig.distance_threshold !== null)) {
                currentClusterConfig.n_clusters = data.n_clusters;
                console.log(`Auto-calculated n_clusters=${data.n_clusters} based on ${data.n_papers} papers`);
            }
        }
    } catch (error) {
        console.warn('Failed to fetch default cluster count, using fallback', error);
        // Only set fallback n_clusters if not using agglomerative with distance_threshold
        if (currentClusterConfig.n_clusters === null && 
            !(currentClusterConfig.clustering_method === 'agglomerative' && 
              currentClusterConfig.distance_threshold !== null)) {
            currentClusterConfig.n_clusters = 5;  // Fallback
        }
    }
}

/**
 * Check if clusters are loaded
 * @returns {boolean} True if clusters are loaded
 */
export function areClustersLoaded() {
    return clusterData !== null;
}

/**
 * Load and visualize clusters
 * @async
 */
export async function loadClusters() {
    try {
        showLoading('cluster-plot', 'Loading clusters...');
        
        // Initialize default cluster count if not set
        await initDefaultClusterCount();
        
        // Try to load cached clusters first
        let response = await fetch(`${API_BASE}/api/clusters/cached`);
        
        if (!response.ok) {
            // If no cache, compute on demand
            console.log('No cached clusters found, computing...');
            response = await fetch(`${API_BASE}/api/clusters/compute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentClusterConfig)
            });
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
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
        
        // Auto-enable hierarchy mode for agglomerative clustering
        if (currentClusterConfig.clustering_method === 'agglomerative' && 
            clusterData.cluster_hierarchy && 
            clusterData.cluster_hierarchy.tree) {
            // Small delay to ensure visualization is rendered first
            setTimeout(() => {
                enableHierarchyMode();
            }, 100);
        }
        
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
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
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
        plot_bgcolor: 'white',  // White background
        paper_bgcolor: 'white',
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
    container.className = 'mb-3 pb-3 border-b border-gray-200';
    
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
        message.className = 'text-xs text-gray-500 text-center py-2';
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
    title.className = 'text-xs text-gray-600 mt-2 text-center';
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
    header.className = 'mb-3 pb-3 border-b border-gray-200';
    
    const title = document.createElement('h4');
    title.className = 'text-sm font-semibold text-gray-700 mb-3';
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
    levelDisplay.className = 'px-2 py-1 text-xs bg-gray-100 border border-gray-300 rounded flex-1 text-center';
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
    infoText.className = 'text-xs text-gray-600 mt-2';
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
        item.className = 'flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-200 transition-colors';
        item.style.backgroundColor = 'rgb(249 250 251)';
        
        // Color box
        const colorBox = document.createElement('div');
        colorBox.className = 'w-4 h-4 rounded flex-shrink-0';
        colorBox.style.backgroundColor = clusterColor;
        
        // Label text
        const labelText = document.createElement('span');
        labelText.className = 'text-sm text-gray-700 flex-1';
        labelText.textContent = `${label} (${cluster.size})`;
        
        item.appendChild(colorBox);
        item.appendChild(labelText);
        
        if (!cluster.is_leaf) {
            const drillIcon = document.createElement('span');
            drillIcon.className = 'text-xs text-gray-500';
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
    title.className = 'text-sm font-semibold text-gray-700 mb-2';
    
    // Build dynamic title with stats
    const titleHTML = formatClusterStats(clusterData?.statistics, labels);
    title.innerHTML = titleHTML;
    header.appendChild(title);
    
    // Create button container
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'flex gap-2';
    
    // "Select All" button
    const selectAllBtn = document.createElement('button');
    selectAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors';
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
    clearAllBtn.className = 'px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors';
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
        labelText.className = 'text-sm text-gray-700 flex-1';
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
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
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
            <div class="bg-white rounded-lg shadow-md p-6">
                <h4 class="text-lg font-semibold text-gray-800">${basicInfo.title}</h4>
                <p class="text-sm text-gray-500 mt-2">Loading full details...</p>
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
 * Open cluster settings modal
 */
export function openClusterSettings() {
    const modalHTML = `
        <div id="cluster-settings-modal" class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div class="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] flex flex-col">
                <div class="bg-purple-600 text-white px-6 py-4 flex justify-between items-center rounded-t-lg flex-shrink-0">
                    <h3 class="text-xl font-semibold">
                        <i class="fas fa-cog mr-2"></i>Clustering Settings
                    </h3>
                    <button onclick="closeClusterSettings()" class="text-white hover:text-gray-200 text-2xl">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6 space-y-4 overflow-y-auto flex-1">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Dimensionality Reduction</label>
                        <select id="cluster-reduction-method" class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                            <option value="pca" ${currentClusterConfig.reduction_method === 'pca' ? 'selected' : ''}>PCA</option>
                            <option value="tsne" ${currentClusterConfig.reduction_method === 'tsne' ? 'selected' : ''}>t-SNE</option>
                            <option value="umap" ${currentClusterConfig.reduction_method === 'umap' ? 'selected' : ''}>UMAP</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Clustering Method</label>
                        <select id="cluster-method" class="w-full px-4 py-2 border border-gray-300 rounded-lg" onchange="toggleClusterParams()">
                            <option value="kmeans" ${currentClusterConfig.clustering_method === 'kmeans' ? 'selected' : ''}>K-Means</option>
                            <option value="dbscan" ${currentClusterConfig.clustering_method === 'dbscan' ? 'selected' : ''}>DBSCAN</option>
                            <option value="agglomerative" ${currentClusterConfig.clustering_method === 'agglomerative' ? 'selected' : ''}>Agglomerative</option>
                            <option value="spectral" ${currentClusterConfig.clustering_method === 'spectral' ? 'selected' : ''}>Spectral</option>
                            <option value="fuzzy_cmeans" ${currentClusterConfig.clustering_method === 'fuzzy_cmeans' ? 'selected' : ''}>Fuzzy C-Means</option>
                        </select>
                    </div>
                    <div id="kmeans-params" class="${currentClusterConfig.clustering_method === 'kmeans' || currentClusterConfig.clustering_method === 'spectral' || currentClusterConfig.clustering_method === 'fuzzy_cmeans' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Number of Clusters</label>
                        <input type="number" id="cluster-n-clusters" value="${currentClusterConfig.n_clusters || ''}" min="2" max="100" 
                               placeholder="Auto (n_papers / 100)"
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                        <p class="text-xs text-gray-500 mt-1">Leave empty for automatic calculation based on paper count</p>
                    </div>
                    <div id="agglomerative-params" class="${currentClusterConfig.clustering_method === 'agglomerative' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Number of Clusters (or Distance Threshold)</label>
                        <input type="number" id="cluster-n-clusters-agg" value="${currentClusterConfig.n_clusters || ''}" min="2" max="100" 
                               placeholder="Leave empty to use distance threshold"
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Distance Threshold (optional)</label>
                        <input type="number" id="cluster-distance-threshold" value="${currentClusterConfig.distance_threshold || ''}" step="0.1" min="0.1" 
                               placeholder="Leave empty to use n_clusters"
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Linkage Method</label>
                        <select id="cluster-linkage" class="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4">
                            <option value="ward" ${currentClusterConfig.linkage === 'ward' ? 'selected' : ''}>Ward</option>
                            <option value="complete" ${currentClusterConfig.linkage === 'complete' ? 'selected' : ''}>Complete</option>
                            <option value="average" ${currentClusterConfig.linkage === 'average' ? 'selected' : ''}>Average</option>
                            <option value="single" ${currentClusterConfig.linkage === 'single' ? 'selected' : ''}>Single</option>
                        </select>
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="checkbox" id="cluster-use-llm-labels" ${currentClusterConfig.use_llm_labels !== false ? 'checked' : ''} class="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500">
                            <span class="text-sm font-medium text-gray-700">Use LLM for Hierarchy Labels</span>
                        </label>
                        <p class="text-xs text-gray-500 mt-1 ml-6">When enabled, uses LLM to generate meaningful labels for parent clusters. When disabled, uses simple label concatenation.</p>
                        <p class="text-xs text-gray-500 mt-1">Specify either n_clusters OR distance_threshold (not both)</p>
                    </div>
                    <div id="dbscan-params" class="${currentClusterConfig.clustering_method === 'dbscan' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Epsilon (eps)</label>
                        <input type="number" id="cluster-eps" value="${currentClusterConfig.eps}" step="0.1" min="0.1" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Min Samples</label>
                        <input type="number" id="cluster-min-samples" value="${currentClusterConfig.min_samples}" min="2" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                    </div>
                    <div id="fuzzy-params" class="${currentClusterConfig.clustering_method === 'fuzzy_cmeans' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Fuzziness Parameter (m)</label>
                        <input type="number" id="cluster-fuzziness" value="${currentClusterConfig.m || 2.0}" step="0.1" min="1.1" max="5.0"
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                        <p class="text-xs text-gray-500 mt-1">Higher values create fuzzier clusters (default: 2.0)</p>
                    </div>
                    <div id="spectral-params" class="${currentClusterConfig.clustering_method === 'spectral' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Affinity</label>
                        <select id="cluster-affinity" class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                            <option value="rbf" ${currentClusterConfig.affinity === 'rbf' ? 'selected' : ''}>RBF</option>
                            <option value="nearest_neighbors" ${currentClusterConfig.affinity === 'nearest_neighbors' ? 'selected' : ''}>Nearest Neighbors</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Max Papers (optional)</label>
                        <input type="number" id="cluster-limit" value="${currentClusterConfig.limit || ''}" placeholder="All papers" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                        <p class="text-xs text-gray-500 mt-1">Limit number of papers for faster computation</p>
                    </div>
                    <div class="border-t border-gray-200 pt-4">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="checkbox" id="cluster-force-recreate" class="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500">
                            <span class="text-sm font-medium text-gray-700">Force Recreate (Ignore Cache)</span>
                        </label>
                        <p class="text-xs text-gray-500 mt-1 ml-6">When enabled, clustering will always be recomputed even if cached results exist</p>
                    </div>
                </div>
                <div class="px-6 py-4 bg-gray-50 flex justify-end gap-3 rounded-b-lg flex-shrink-0">
                    <button onclick="closeClusterSettings()" class="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100">
                        Cancel
                    </button>
                    <button onclick="applyClusterSettings()" class="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                        Apply & Recompute
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

/**
 * Close cluster settings modal
 */
export function closeClusterSettings() {
    const modal = document.getElementById('cluster-settings-modal');
    if (modal) {
        modal.remove();
    }
}

/**
 * Apply cluster settings and recompute
 * @async
 */
export async function applyClusterSettings() {
    // Get settings from modal
    const method = document.getElementById('cluster-method').value;
    const forceRecreate = document.getElementById('cluster-force-recreate').checked;
    
    // Base config
    currentClusterConfig = {
        reduction_method: document.getElementById('cluster-reduction-method').value,
        clustering_method: method,
        limit: parseInt(document.getElementById('cluster-limit').value) || null,
        force: forceRecreate
    };
    
    // Method-specific parameters
    if (method === 'kmeans' || method === 'spectral' || method === 'fuzzy_cmeans') {
        const nClustersValue = document.getElementById('cluster-n-clusters').value;
        currentClusterConfig.n_clusters = nClustersValue ? parseInt(nClustersValue) : null;
    }
    
    if (method === 'agglomerative') {
        const nClustersValue = document.getElementById('cluster-n-clusters-agg').value;
        const distThresholdValue = document.getElementById('cluster-distance-threshold').value;
        const useLLMLabels = document.getElementById('cluster-use-llm-labels').checked;
        
        if (distThresholdValue) {
            currentClusterConfig.distance_threshold = parseFloat(distThresholdValue);
            currentClusterConfig.n_clusters = null;  // Can't specify both
        } else {
            currentClusterConfig.n_clusters = nClustersValue ? parseInt(nClustersValue) : null;
        }
        
        currentClusterConfig.linkage = document.getElementById('cluster-linkage').value;
        currentClusterConfig.use_llm_labels = useLLMLabels;
    }
    
    if (method === 'dbscan') {
        currentClusterConfig.eps = parseFloat(document.getElementById('cluster-eps').value) || 0.5;
        currentClusterConfig.min_samples = parseInt(document.getElementById('cluster-min-samples').value) || 5;
    }
    
    if (method === 'fuzzy_cmeans') {
        currentClusterConfig.m = parseFloat(document.getElementById('cluster-fuzziness').value) || 2.0;
    }
    
    if (method === 'spectral') {
        currentClusterConfig.affinity = document.getElementById('cluster-affinity').value;
    }
    
    closeClusterSettings();
    
    showLoading('cluster-plot', 'Recomputing clusters with new settings...');
    
    try {
        const response = await fetch(`${API_BASE}/api/clusters/compute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentClusterConfig)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        clusterData = await response.json();
        
        if (clusterData.error) {
            showErrorInElement('cluster-plot', clusterData.error);
            return;
        }
        
        visualizeClusters();
        
    } catch (error) {
        console.error('Error recomputing clusters:', error);
        showErrorInElement('cluster-plot', `Failed to recompute clusters: ${error.message}`);
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
 * Toggle cluster parameter visibility
 */
export function toggleClusterParams() {
    const method = document.getElementById('cluster-method').value;
    const kmeansParams = document.getElementById('kmeans-params');
    const dbscanParams = document.getElementById('dbscan-params');
    const agglomerativeParams = document.getElementById('agglomerative-params');
    const fuzzyParams = document.getElementById('fuzzy-params');
    const spectralParams = document.getElementById('spectral-params');
    
    // Hide all parameter sections first
    kmeansParams.classList.add('hidden');
    dbscanParams.classList.add('hidden');
    agglomerativeParams.classList.add('hidden');
    fuzzyParams.classList.add('hidden');
    spectralParams.classList.add('hidden');
    
    // Show relevant parameter section
    if (method === 'kmeans') {
        kmeansParams.classList.remove('hidden');
    } else if (method === 'dbscan') {
        dbscanParams.classList.remove('hidden');
    } else if (method === 'agglomerative') {
        agglomerativeParams.classList.remove('hidden');
    } else if (method === 'fuzzy_cmeans') {
        kmeansParams.classList.remove('hidden');  // Uses n_clusters
        fuzzyParams.classList.remove('hidden');
    } else if (method === 'spectral') {
        kmeansParams.classList.remove('hidden');  // Uses n_clusters
        spectralParams.classList.remove('hidden');
    }
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
 * Pre-calculate clusters in background for caching
 * Called when filters change to warm up the cache
 * @async
 */
export async function precalculateClusters() {
    try {
        // Use current config for pre-calculation
        const config = {
            reduction_method: currentClusterConfig.reduction_method,
            n_components: currentClusterConfig.n_components,
            clustering_method: currentClusterConfig.clustering_method,
            n_clusters: currentClusterConfig.n_clusters
        };
        
        console.log('Starting background clustering pre-calculation...');
        
        const response = await fetch(`${API_BASE}/api/clusters/precalculate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('Background clustering pre-calculation:', data.message);
        } else {
            console.warn('Failed to start background clustering pre-calculation');
        }
    } catch (error) {
        console.warn('Error starting background clustering pre-calculation:', error);
    }
}

/**
 * Search for papers within distance of a custom query
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
        const response = await fetch(`${API_BASE}/api/clusters/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, distance })
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
        
        // Show statistics
        displayCustomQueryStats(customCluster);
        
        // Update visualization to highlight custom cluster
        if (clusterData) {
            updateVisualizationWithCustomCluster(customCluster);
        }
        
        // Update legend to include custom cluster
        updateLegendWithCustomClusters();
        
        console.log(`Found ${data.count} papers within distance ${distance} for query: "${query}"`);
        
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
 * Display statistics for a custom query cluster
 * @param {Object} customCluster - Custom cluster data
 */
function displayCustomQueryStats(customCluster) {
    const statsDiv = document.getElementById('custom-query-stats');
    const statsContent = document.getElementById('custom-query-stats-content');
    
    // Show the stats div
    statsDiv.classList.remove('hidden');
    
    // Escape query text to prevent XSS
    const escapedQuery = escapeHtml(customCluster.query);
    
    // Build statistics HTML
    const html = `
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div class="bg-purple-50 rounded-lg p-4">
                <div class="text-sm text-gray-600 mb-1">Query</div>
                <div class="text-xl font-bold text-purple-700">${escapedQuery}</div>
            </div>
            <div class="bg-blue-50 rounded-lg p-4">
                <div class="text-sm text-gray-600 mb-1">Papers Found</div>
                <div class="text-xl font-bold text-blue-700">${customCluster.count}</div>
            </div>
            <div class="bg-green-50 rounded-lg p-4">
                <div class="text-sm text-gray-600 mb-1">Distance Radius</div>
                <div class="text-xl font-bold text-green-700">${customCluster.distance.toFixed(1)}</div>
            </div>
        </div>
        <div class="text-sm text-gray-600 mt-4">
            <strong>Relevance:</strong> ${customCluster.count} papers found within a Euclidean distance of ${customCluster.distance.toFixed(1)} in the embedding space.
            ${customCluster.count > 0 
                ? `The closest paper is at distance ${customCluster.papers[0].distance.toFixed(2)}.` 
                : 'No papers found within this radius.'}
        </div>
    `;
    
    statsContent.innerHTML = html;
}

/**
 * Update visualization to highlight papers in custom cluster
 * @param {Object} customCluster - Custom cluster data
 */
function updateVisualizationWithCustomCluster(customCluster) {
    if (!clusterData || !clusterData.reduced_embeddings) {
        console.warn('No cluster data available for visualization update');
        return;
    }
    
    // Get the plot div
    const plotDiv = document.getElementById('cluster-plot');
    
    // Find indices of papers in the custom cluster
    const paperIds = new Set(customCluster.papers.map(p => p.openreview_id));
    const customIndices = [];
    const customX = [];
    const customY = [];
    const customText = [];
    
    clusterData.papers.forEach((paper, idx) => {
        if (paperIds.has(paper.openreview_id)) {
            customIndices.push(idx);
            customX.push(clusterData.reduced_embeddings[idx][0]);
            customY.push(clusterData.reduced_embeddings[idx][1]);
            
            // Find the distance for this paper
            const matchingPaper = customCluster.papers.find(p => p.openreview_id === paper.openreview_id);
            const distance = matchingPaper ? matchingPaper.distance.toFixed(2) : 'N/A';
            customText.push(`${paper.title}<br>Distance: ${distance}`);
        }
    });
    
    // Add a new trace for custom cluster papers
    const customTrace = {
        x: customX,
        y: customY,
        mode: 'markers',
        type: 'scatter',
        name: `Custom: ${customCluster.query}`,
        text: customText,
        hoverinfo: 'text',
        marker: {
            size: 12,
            color: 'red',
            symbol: 'diamond',
            line: {
                color: 'darkred',
                width: 2
            }
        },
        customdata: customIndices.map(idx => clusterData.papers[idx].openreview_id),
        showlegend: false  // We'll add it to the legend separately
    };
    
    // Add the trace to the existing plot
    Plotly.addTraces(plotDiv, customTrace);
}

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Update legend to include custom clusters with delete buttons
 */
function updateLegendWithCustomClusters() {
    const legendDiv = document.getElementById('cluster-legend');
    
    // Remove any existing custom clusters section
    const existingCustomSection = legendDiv.querySelector('.custom-queries-section');
    if (existingCustomSection) {
        existingCustomSection.remove();
    }
    
    if (customQueryClusters.length === 0) {
        // No custom clusters, nothing to add
        return;
    }
    
    // Build custom clusters section
    let customClustersHtml = `
        <div class="custom-queries-section border-t-2 border-purple-600 pt-4 mt-4">
            <h4 class="text-md font-bold text-purple-700 mb-3">
                <i class="fas fa-search mr-2"></i>Custom Queries
            </h4>
    `;
    
    customQueryClusters.forEach(cluster => {
        // Escape user-controlled data to prevent XSS
        const escapedQuery = escapeHtml(cluster.query);
        const escapedId = escapeHtml(cluster.id);
        
        customClustersHtml += `
            <div class="flex items-center justify-between mb-3 p-2 bg-purple-50 rounded-lg">
                <div class="flex-1">
                    <div class="font-semibold text-sm text-gray-800">${escapedQuery}</div>
                    <div class="text-xs text-gray-600">${cluster.count} papers (d=${cluster.distance})</div>
                </div>
                <button onclick="deleteCustomCluster('${escapedId}')" 
                    class="ml-2 px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
    });
    
    customClustersHtml += '</div>';
    
    // Append custom clusters section to legend
    legendDiv.insertAdjacentHTML('beforeend', customClustersHtml);
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
    
    // If no more custom clusters, exit custom cluster mode
    if (customQueryClusters.length === 0) {
        customClusterMode = false;
        // Hide stats div
        document.getElementById('custom-query-stats').classList.add('hidden');
        // Reload clusters to remove custom traces
        await loadClusters();
    } else {
        // Re-render the visualization with remaining custom clusters
        await loadClusters();
        customQueryClusters.forEach(cluster => {
            updateVisualizationWithCustomCluster(cluster);
        });
        updateLegendWithCustomClusters();
    }
}

// Make custom cluster functions globally accessible for onclick handlers
if (typeof window !== 'undefined') {
    window.searchCustomCluster = searchCustomCluster;
    window.deleteCustomCluster = deleteCustomCluster;
}
