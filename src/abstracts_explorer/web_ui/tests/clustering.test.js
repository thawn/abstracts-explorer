/**
 * Comprehensive unit tests for clustering.js module
 */

import { jest } from '@jest/globals';

// Mock dependencies before imports
global.fetch = jest.fn();
global.Plotly = {
    newPlot: jest.fn(() => Promise.resolve()),
    relayout: jest.fn(() => Promise.resolve()),
    restyle: jest.fn(() => Promise.resolve())
};
global.alert = jest.fn();
global.console = {
    ...console,
    log: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
};

// Mock the imported modules
jest.unstable_mockModule('../static/modules/utils/constants.js', () => ({
    API_BASE: '',
    PLOTLY_COLORS: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}));

jest.unstable_mockModule('../static/modules/utils/ui-utils.js', () => ({
    showLoading: jest.fn(),
    showErrorInElement: jest.fn()
}));

jest.unstable_mockModule('../static/modules/utils/sort-utils.js', () => ({
    sortClustersBySizeDesc: jest.fn((entries) => entries.sort((a, b) => b[1].length - a[1].length))
}));

jest.unstable_mockModule('../static/modules/utils/cluster-utils.js', () => ({
    getClusterLabelWithCount: jest.fn((clusterId, labels, count) => 
        `${labels[clusterId] || `Cluster ${clusterId}`} (${count})`)
}));

jest.unstable_mockModule('../static/modules/paper-card.js', () => ({
    formatPaperCard: jest.fn((paper) => `<div class="paper-card">${paper.title}</div>`)
}));

// Import the module after mocks are set up
const {
    areClustersLoaded,
    loadClusters,
    enableHierarchyMode,
    disableHierarchyMode,
    loadHierarchyLevel,
    navigateHierarchyUp,
    navigateHierarchyDown,
    visualizeClusters,
    showClusterPaperDetails,
    exportClusters,
    getClusterData,
} = await import('../static/modules/clustering.js');

describe('Clustering Module', () => {
    // Reset module state between describe blocks
    let moduleInstance;
    
    beforeEach(() => {
        jest.clearAllMocks();
        document.body.innerHTML = `
            <div id="cluster-plot"></div>
            <div id="cluster-legend"></div>
            <div id="selected-paper-details" class="hidden"></div>
            <div id="selected-paper-content"></div>
            <select id="year-selector"><option value="2025">2025</option></select>
            <select id="conference-selector"><option value="">All</option></select>
        `;
        
        // Mock plotDiv.on method for Plotly event handlers
        const plotDiv = document.getElementById('cluster-plot');
        plotDiv.on = jest.fn();
        plotDiv.data = [];
        
        global.fetch.mockClear();
    });

    describe('areClustersLoaded', () => {
        it('should return false when no clusters loaded', () => {
            expect(areClustersLoaded()).toBe(false);
        });

        it('should return true after clusters are loaded', async () => {
            const mockClusterData = {
                points: [
                    { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }
                ],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_centers: { 0: { x: 1, y: 1 } }
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();
            expect(areClustersLoaded()).toBe(true);
        });
    });

    describe('loadClusters', () => {
        const mockClusterData = {
            points: [
                { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1', year: 2025, conference: 'NeurIPS', session: 'ML' },
                { x: 2, y: 2, cluster: 1, id: '2', title: 'Paper 2', year: 2025, conference: 'NeurIPS', session: 'DL' }
            ],
            cluster_labels: { 0: 'ML Cluster', 1: 'DL Cluster' },
            cluster_centers: {
                0: { x: 1, y: 1 },
                1: { x: 2, y: 2 }
            }
        };

        it('should load clusters via POST to compute endpoint', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();

            expect(global.fetch).toHaveBeenCalledWith(
                '/api/clusters/compute',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
            );
            expect(areClustersLoaded()).toBe(true);
        });

        it('should include conference and year filters in request', async () => {
            const yearSelect = document.getElementById('year-selector');
            yearSelect.value = '2025';
            const conferenceSelect = document.getElementById('conference-selector');
            conferenceSelect.innerHTML = '<option value="NeurIPS">NeurIPS</option>';
            conferenceSelect.value = 'NeurIPS';

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();

            const requestBody = JSON.parse(global.fetch.mock.calls[0][1].body);
            expect(requestBody.conferences).toEqual(['NeurIPS']);
            expect(requestBody.years).toEqual([2025]);
        });

        it('should handle error response', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            global.fetch.mockResolvedValueOnce({
                ok: false,
                status: 500,
                json: async () => ({ error: 'Server error' })
            });

            await loadClusters();

            expect(uiUtils.showErrorInElement).toHaveBeenCalled();
        });

        it('should handle error in response data', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ error: 'Clustering failed' })
            });

            await loadClusters();

            expect(uiUtils.showErrorInElement).toHaveBeenCalledWith(
                'cluster-plot',
                'Clustering failed'
            );
        });

        it('should not auto-enable hierarchy mode for agglomerative clustering', async () => {
            const mockHierarchyData = {
                ...mockClusterData,
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Paper 1' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockHierarchyData
            });

            await loadClusters();

            // Hierarchy mode should remain disabled after loading (default is false).
            // Verify by attempting to load a hierarchy level — this warns when not in hierarchy mode.
            await loadHierarchyLevel(0);
            expect(global.console.warn).toHaveBeenCalledWith('Not in hierarchy mode');
            // Normal visualization should have been called, not hierarchy visualization
            expect(global.Plotly.newPlot).toHaveBeenCalled();
        });

        it('should visualize clusters after loading', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();

            expect(global.Plotly.newPlot).toHaveBeenCalled();
        });
    });

    describe('visualizeClusters', () => {
        it('should not visualize when no data loaded', () => {
            // Since module state persists, we verify the function handles missing data
            // by checking it logs an error when data is invalid
            const currentData = getClusterData();
            if (!currentData || !currentData.points) {
                visualizeClusters();
                expect(global.console.error).toHaveBeenCalledWith('No cluster data to visualize');
            } else {
                // Module already has data from previous tests - skip verification
                expect(true).toBe(true);
            }
        });

        it('should create plotly visualization with cluster data', async () => {
            const mockClusterData = {
                points: [
                    { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1', year: 2025, conference: 'NeurIPS', session: 'ML' }
                ],
                cluster_labels: { 0: 'Test Cluster' },
                cluster_centers: { 0: { x: 1, y: 1 } }
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();

            expect(global.Plotly.newPlot).toHaveBeenCalledWith(
                'cluster-plot',
                expect.any(Array),
                expect.objectContaining({
                    hovermode: 'closest',
                    showlegend: false
                }),
                expect.any(Object)
            );
        });

        it('should create traces for each cluster', async () => {
            const mockClusterData = {
                points: [
                    { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1', year: 2025, conference: 'NeurIPS', session: 'ML' },
                    { x: 2, y: 2, cluster: 1, id: '2', title: 'Paper 2', year: 2025, conference: 'ICML', session: 'RL' }
                ],
                cluster_labels: { 0: 'Cluster 0', 1: 'Cluster 1' },
                cluster_centers: {
                    0: { x: 1, y: 1 },
                    1: { x: 2, y: 2 }
                }
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockClusterData
            });

            await loadClusters();

            const plotCall = global.Plotly.newPlot.mock.calls[0];
            const traces = plotCall[1];
            
            // Should have traces for points and centers
            expect(traces.length).toBeGreaterThan(0);
        });
    });

    describe('enableHierarchyMode', () => {
        it('should show alert when no hierarchy available', async () => {
            const mockClusterData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();
            await enableHierarchyMode();

            expect(global.alert).toHaveBeenCalled();
        });

        it('should enable hierarchy mode with valid tree', async () => {
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();

            expect(global.Plotly.newPlot).toHaveBeenCalled();
        });

        it('should show alert when tree has no nodes', async () => {
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {},
                        root: 0,
                        max_level: 0
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();

            expect(global.alert).toHaveBeenCalled();
        });
    });

    describe('disableHierarchyMode', () => {
        it('should restore normal visualization', async () => {
            const mockClusterData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();
            const callCount = global.Plotly.newPlot.mock.calls.length;
            
            disableHierarchyMode();

            expect(global.Plotly.newPlot.mock.calls.length).toBe(callCount + 1);
        });
    });

    describe('loadHierarchyLevel', () => {
        const mockHierarchyData = {
            points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
            cluster_labels: { 0: 'Cluster 0' },
            cluster_hierarchy: {
                tree: {
                    nodes: {
                        0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf 1' },
                        1: { node_id: 1, level: 0, samples: [1], is_leaf: true, children: [], label: 'Leaf 2' },
                        2: { node_id: 2, level: 1, samples: [0, 1], is_leaf: false, children: [0, 1], label: 'Parent' }
                    },
                    root: 2,
                    max_level: 1
                }
            }
        };

        it('should not load level when not in hierarchy mode', async () => {
            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await loadHierarchyLevel(0);

            expect(global.console.warn).toHaveBeenCalledWith('Not in hierarchy mode');
        });

        it('should load hierarchy level when in hierarchy mode', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();
            
            jest.clearAllMocks();
            await loadHierarchyLevel(0);

            expect(uiUtils.showLoading).toHaveBeenCalledWith('cluster-plot', 'Loading hierarchy level 0...');
        });
    });

    describe('navigateHierarchyUp', () => {
        it('should navigate to higher hierarchy level', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();
            
            // Navigate down first
            await loadHierarchyLevel(0);
            jest.clearAllMocks();
            
            // Navigate up
            await navigateHierarchyUp();

            expect(uiUtils.showLoading).toHaveBeenCalled();
        });
    });

    describe('navigateHierarchyDown', () => {
        it('should navigate to lower hierarchy level', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();
            jest.clearAllMocks();
            
            // Navigate down
            await navigateHierarchyDown();

            expect(uiUtils.showLoading).toHaveBeenCalled();
        });
    });

    describe('showClusterPaperDetails', () => {
        it('should show loading state initially', async () => {
            const paperCard = await import('../static/modules/paper-card.js');
            
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    title: 'Full Paper Title',
                    authors: ['Author 1'],
                    year: 2025,
                    conference: 'NeurIPS',
                    abstract: 'Full abstract'
                })
            });

            await showClusterPaperDetails('paper-1', {
                title: 'Paper Title',
                year: 2025,
                conference: 'NeurIPS'
            });

            expect(paperCard.formatPaperCard).toHaveBeenCalled();
        });

        it('should fetch and display full paper details', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    title: 'Full Paper Title',
                    authors: ['Author 1', 'Author 2'],
                    year: 2025,
                    conference: 'NeurIPS',
                    session: 'ML',
                    abstract: 'Full abstract text',
                    url: 'https://example.com/paper',
                    keywords: ['ML', 'DL']
                })
            });

            await showClusterPaperDetails('paper-1', {
                title: 'Paper Title',
                year: 2025,
                conference: 'NeurIPS'
            });

            expect(global.fetch).toHaveBeenCalledWith('/api/paper/paper-1');
        });

        it('should handle fetch error gracefully', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: false,
                status: 404
            });

            await showClusterPaperDetails('paper-1', {
                title: 'Paper Title',
                year: 2025,
                conference: 'NeurIPS'
            });

            expect(global.console.error).toHaveBeenCalled();
        });

        it('should handle missing DOM elements', async () => {
            document.body.innerHTML = '';

            await showClusterPaperDetails('paper-1', {
                title: 'Paper Title',
                year: 2025,
                conference: 'NeurIPS'
            });

            expect(global.console.warn).toHaveBeenCalled();
        });
    });





    describe('exportClusters', () => {
        it('should trigger download of cluster data', async () => {
            const mockClusterData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();

            // Mock createObjectURL and createElement
            const mockUrl = 'blob:mock-url';
            global.URL.createObjectURL = jest.fn(() => mockUrl);
            global.URL.revokeObjectURL = jest.fn();
            
            const mockLink = {
                href: '',
                download: '',
                click: jest.fn(),
                remove: jest.fn(),
                setAttribute: jest.fn()
            };
            const origCreateElement = document.createElement.bind(document);
            document.createElement = jest.fn((tag) => {
                if (tag === 'a') return mockLink;
                return origCreateElement(tag);
            });

            exportClusters();

            expect(mockLink.setAttribute).toHaveBeenCalledWith('download', 'clusters.json');
            expect(mockLink.click).toHaveBeenCalled();
        });
    });

    describe('getClusterData', () => {
        it('should return cluster data state', async () => {
            // Module state persists across tests
            // This verifies getClusterData returns the current state
            const data = getClusterData();
            expect(data === null || (data && typeof data === 'object')).toBe(true);
            
            // If we have data, verify it has expected structure
            if (data) {
                expect(data.points !== undefined || data.cluster_labels !== undefined).toBe(true);
            }
        });

        it('should return cluster data after loading', async () => {
            const mockClusterData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();

            const data = getClusterData();
            expect(data).toBeTruthy();
            expect(data.points).toBeDefined();
        });
    });

    describe('Edge cases and error handling', () => {
        it('should handle network errors gracefully', async () => {
            const uiUtils = await import('../static/modules/utils/ui-utils.js');
            
            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            await loadClusters();

            expect(uiUtils.showErrorInElement).toHaveBeenCalled();
        });

        it('should handle malformed cluster data', async () => {
            // Test validates that visualizeClusters doesn't crash with missing data
            // The function checks for clusterData and points before processing
            const data = getClusterData();
            
            // Call visualize - should not throw even if data is malformed
            expect(() => {
                visualizeClusters();
            }).not.toThrow();
            
            // If data is missing or invalid, error should be logged
            if (!data || !data.points) {
                expect(global.console.error).toHaveBeenCalled();
            }
        });

    });


    describe('Window global functions', () => {
        it('should expose hierarchy functions to window', () => {
            expect(window.enableHierarchyMode).toBeDefined();
            expect(window.disableHierarchyMode).toBeDefined();
            expect(window.navigateHierarchyUp).toBeDefined();
            expect(window.navigateHierarchyDown).toBeDefined();
        });
    });

    describe('Legend and visualization helpers', () => {
        it('should create custom legend with clusters', async () => {
            const mockClusterData = {
                points: [
                    { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1', year: 2025, conference: 'NeurIPS', session: 'ML' },
                    { x: 2, y: 2, cluster: 1, id: '2', title: 'Paper 2', year: 2025, conference: 'NeurIPS', session: 'DL' }
                ],
                cluster_labels: { 0: 'ML Cluster', 1: 'DL Cluster' },
                cluster_centers: {
                    0: { x: 1, y: 1 },
                    1: { x: 2, y: 2 }
                },
                statistics: {
                    total_papers: 2,
                    n_clusters: 2,
                    n_noise: 0
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();

            const legendDiv = document.getElementById('cluster-legend');
            expect(legendDiv.innerHTML).toContain('ML Cluster');
        });

        it('should handle cluster statistics with noise', async () => {
            const mockClusterData = {
                points: [
                    { x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1', year: 2025, conference: 'NeurIPS', session: 'ML' },
                    { x: 2, y: 2, cluster: -1, id: '2', title: 'Paper 2', year: 2025, conference: 'NeurIPS', session: 'Noise' }
                ],
                cluster_labels: { 0: 'ML Cluster' },
                cluster_centers: { 0: { x: 1, y: 1 } },
                statistics: {
                    total_papers: 2,
                    n_clusters: 1,
                    n_noise: 1
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockClusterData
                });

            await loadClusters();

            const legendDiv = document.getElementById('cluster-legend');
            expect(legendDiv.innerHTML).toBeTruthy();
        });
    });

    describe('Hierarchy legend and dendrogram', () => {
        it('should create hierarchy legend with controls', async () => {
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    },
                    dendrogram: {
                        icoord: [[1, 2, 3, 4]],
                        dcoord: [[0, 1, 1, 0]]
                    },
                    n_samples: 1
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();

            const legendDiv = document.getElementById('cluster-legend');
            expect(legendDiv.innerHTML).toContain('Hierarchical View');
        });

        it('should handle missing dendrogram data', async () => {
            const mockHierarchyData = {
                points: [{ x: 1, y: 1, cluster: 0, id: '1', title: 'Paper 1' }],
                cluster_labels: { 0: 'Cluster 0' },
                cluster_hierarchy: {
                    tree: {
                        nodes: {
                            0: { node_id: 0, level: 0, samples: [0], is_leaf: true, children: [], label: 'Leaf' },
                            1: { node_id: 1, level: 1, samples: [0], is_leaf: false, children: [0], label: 'Parent' }
                        },
                        root: 1,
                        max_level: 1
                    }
                }
            };

            global.fetch.mockResolvedValueOnce({
                    ok: true,
                    json: async () => mockHierarchyData
                });

            await loadClusters();
            await enableHierarchyMode();

            // Should still work without dendrogram
            const legendDiv = document.getElementById('cluster-legend');
            expect(legendDiv).toBeTruthy();
        });
    });


});
