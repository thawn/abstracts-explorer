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
    clustering_method: 'kmeans',
    n_clusters: null,  // Will be auto-calculated
    eps: 0.5,
    min_samples: 5,
    limit: null
};
// Track selected clusters for multi-select
let selectedClusters = new Set();

/**
 * Initialize default cluster count from backend
 * @async
 */
async function initDefaultClusterCount() {
    try {
        const response = await fetch(`${API_BASE}/api/clusters/default-count`);
        if (response.ok) {
            const data = await response.json();
            if (currentClusterConfig.n_clusters === null) {
                currentClusterConfig.n_clusters = data.n_clusters;
                console.log(`Auto-calculated n_clusters=${data.n_clusters} based on ${data.n_papers} papers`);
            }
        }
    } catch (error) {
        console.warn('Failed to fetch default cluster count, using fallback', error);
        if (currentClusterConfig.n_clusters === null) {
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
        
    } catch (error) {
        console.error('Error loading clusters:', error);
        showErrorInElement('cluster-plot', `Failed to load clusters: ${error.message}`);
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
        title: 'All Clusters',
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
            <div class="bg-white rounded-lg shadow-xl max-w-2xl w-full">
                <div class="bg-purple-600 text-white px-6 py-4 flex justify-between items-center rounded-t-lg">
                    <h3 class="text-xl font-semibold">
                        <i class="fas fa-cog mr-2"></i>Clustering Settings
                    </h3>
                    <button onclick="closeClusterSettings()" class="text-white hover:text-gray-200 text-2xl">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6 space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Dimensionality Reduction</label>
                        <select id="cluster-reduction-method" class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                            <option value="pca" ${currentClusterConfig.reduction_method === 'pca' ? 'selected' : ''}>PCA</option>
                            <option value="tsne" ${currentClusterConfig.reduction_method === 'tsne' ? 'selected' : ''}>t-SNE</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Clustering Method</label>
                        <select id="cluster-method" class="w-full px-4 py-2 border border-gray-300 rounded-lg" onchange="toggleClusterParams()">
                            <option value="kmeans" ${currentClusterConfig.clustering_method === 'kmeans' ? 'selected' : ''}>K-Means</option>
                            <option value="dbscan" ${currentClusterConfig.clustering_method === 'dbscan' ? 'selected' : ''}>DBSCAN</option>
                            <option value="agglomerative" ${currentClusterConfig.clustering_method === 'agglomerative' ? 'selected' : ''}>Agglomerative</option>
                        </select>
                    </div>
                    <div id="kmeans-params" class="${currentClusterConfig.clustering_method === 'kmeans' || currentClusterConfig.clustering_method === 'agglomerative' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Number of Clusters</label>
                        <input type="number" id="cluster-n-clusters" value="${currentClusterConfig.n_clusters || ''}" min="2" max="100" 
                               placeholder="Auto (n_papers / 100)"
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                        <p class="text-xs text-gray-500 mt-1">Leave empty for automatic calculation based on paper count</p>
                    </div>
                    <div id="dbscan-params" class="${currentClusterConfig.clustering_method === 'dbscan' ? '' : 'hidden'}">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Epsilon (eps)</label>
                        <input type="number" id="cluster-eps" value="${currentClusterConfig.eps}" step="0.1" min="0.1" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Min Samples</label>
                        <input type="number" id="cluster-min-samples" value="${currentClusterConfig.min_samples}" min="2" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Max Papers (optional)</label>
                        <input type="number" id="cluster-limit" value="${currentClusterConfig.limit || ''}" placeholder="All papers" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-lg">
                        <p class="text-xs text-gray-500 mt-1">Limit number of papers for faster computation</p>
                    </div>
                </div>
                <div class="px-6 py-4 bg-gray-50 flex justify-end gap-3 rounded-b-lg">
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
    // Get settings from modal
    const nClustersValue = document.getElementById('cluster-n-clusters').value;
    currentClusterConfig = {
        reduction_method: document.getElementById('cluster-reduction-method').value,
        clustering_method: document.getElementById('cluster-method').value,
        n_clusters: nClustersValue ? parseInt(nClustersValue) : null,  // null means auto-calculate
        eps: parseFloat(document.getElementById('cluster-eps').value) || 0.5,
        min_samples: parseInt(document.getElementById('cluster-min-samples').value) || 5,
        limit: parseInt(document.getElementById('cluster-limit').value) || null
    };
    
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
    
    if (method === 'kmeans' || method === 'agglomerative') {
        kmeansParams.classList.remove('hidden');
        dbscanParams.classList.add('hidden');
    } else if (method === 'dbscan') {
        kmeansParams.classList.add('hidden');
        dbscanParams.classList.remove('hidden');
    }
}

/**
 * Get current cluster data (for external access)
 * @returns {Object|null} Cluster data
 */
export function getClusterData() {
    return clusterData;
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
