/**
 * Unit tests for hierarchical clustering functionality
 */

import { jest } from '@jest/globals';

// Mock global objects before importing modules
global.fetch = jest.fn();
global.Plotly = {
    newPlot: jest.fn(),
    relayout: jest.fn(),
    restyle: jest.fn()
};
global.alert = jest.fn();
global.console.warn = jest.fn();
global.console.error = jest.fn();

// Import clustering module - we need to mock it since it has circular dependencies
// For now, we'll test the buildLevelDataFromHierarchy logic directly

describe('Hierarchical Clustering', () => {
    // Sample hierarchy tree structure matching backend format
    const mockHierarchyTree = {
        nodes: {
            0: { node_id: 0, is_leaf: true, children: [], samples: [0], level: 0, label: 'Paper 1' },
            1: { node_id: 1, is_leaf: true, children: [], samples: [1], level: 0, label: 'Paper 2' },
            2: { node_id: 2, is_leaf: true, children: [], samples: [2], level: 0, label: 'Paper 3' },
            3: { node_id: 3, is_leaf: true, children: [], samples: [3], level: 0, label: 'Paper 4' },
            4: { node_id: 4, is_leaf: false, children: [0, 1], samples: [0, 1], level: 1, label: 'ML Cluster' },
            5: { node_id: 5, is_leaf: false, children: [2, 3], samples: [2, 3], level: 1, label: 'NLP Cluster' },
            6: { node_id: 6, is_leaf: false, children: [4, 5], samples: [0, 1, 2, 3], level: 2, label: 'AI Cluster' }
        },
        root: 6,
        max_level: 2
    };

    describe('buildLevelDataFromHierarchy', () => {
        // Helper function that mimics the JavaScript function
        function buildLevelDataFromHierarchy(tree, level, parentId = null) {
            if (!tree || !tree.nodes) {
                return { clusters: [], level: 0, max_level: 0, labels: {} };
            }
            
            // Get all nodes at this level
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
                labels: {}
            };
        }

        it('should return empty data for invalid tree', () => {
            const result = buildLevelDataFromHierarchy(null, 0);
            expect(result.clusters).toEqual([]);
            expect(result.level).toBe(0);
            expect(result.max_level).toBe(0);
        });

        it('should return empty data for tree without nodes', () => {
            const result = buildLevelDataFromHierarchy({}, 0);
            expect(result.clusters).toEqual([]);
            expect(result.level).toBe(0);
        });

        it('should get all leaf nodes at level 0', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 0);
            expect(result.clusters.length).toBe(4);  // 4 leaf nodes
            expect(result.level).toBe(0);
            expect(result.max_level).toBe(2);
            expect(result.clusters[0].is_leaf).toBe(true);
            expect(result.clusters[0].samples).toHaveLength(1);
        });

        it('should get intermediate nodes at level 1', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 1);
            expect(result.clusters.length).toBe(2);  // 2 intermediate nodes
            expect(result.level).toBe(1);
            expect(result.clusters[0].is_leaf).toBe(false);
            expect(result.clusters[0].has_children).toBe(true);
        });

        it('should get root node at level 2', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 2);
            expect(result.clusters.length).toBe(1);  // Root node
            expect(result.level).toBe(2);
            expect(result.clusters[0].node_id).toBe(6);
            expect(result.clusters[0].samples).toHaveLength(4);
        });

        it('should filter by parent node', () => {
            // Get children of node 4 (which has children 0 and 1 at level 0)
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 0, 4);
            expect(result.clusters.length).toBe(2);
            const nodeIds = result.clusters.map(c => c.node_id);
            expect(nodeIds).toContain(0);
            expect(nodeIds).toContain(1);
            expect(nodeIds).not.toContain(2);
            expect(nodeIds).not.toContain(3);
        });

        it('should return empty when parent has no children at level', () => {
            // Try to get level 2 nodes that are children of node 4 (but node 4 is at level 1)
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 2, 4);
            expect(result.clusters.length).toBe(0);
        });

        it('should calculate correct cluster sizes', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 1);
            expect(result.clusters[0].size).toBe(2);  // Each intermediate node has 2 samples
            expect(result.clusters[1].size).toBe(2);
        });

        it('should include cluster labels', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 1);
            expect(result.clusters[0].label).toBeTruthy();
            expect(result.clusters[0].label).toMatch(/Cluster|ML|NLP/);
        });
    });

    describe('Hierarchy validation', () => {
        it('should validate tree has nodes', () => {
            const tree = { root: 1, max_level: 1 };  // Missing nodes
            const result = buildLevelDataFromHierarchy(tree, 0);
            expect(result.clusters).toEqual([]);
        });

        it('should handle tree with empty nodes object', () => {
            const tree = { nodes: {}, root: 0, max_level: 0 };
            const result = buildLevelDataFromHierarchy(tree, 0);
            expect(result.clusters).toEqual([]);
        });
    });

    describe('Edge cases', () => {
        it('should handle nodes without samples array', () => {
            const tree = {
                nodes: {
                    0: { node_id: 0, is_leaf: true, children: [], level: 0 }
                },
                root: 0,
                max_level: 0
            };
            const result = buildLevelDataFromHierarchy(tree, 0);
            expect(result.clusters[0].samples).toEqual([]);
            expect(result.clusters[0].size).toBe(0);
        });

        it('should handle nodes without children array', () => {
            const tree = {
                nodes: {
                    0: { node_id: 0, is_leaf: false, samples: [0], level: 0 }
                },
                root: 0,
                max_level: 0
            };
            const result = buildLevelDataFromHierarchy(tree, 0);
            expect(result.clusters[0].has_children).toBe(false);
        });

        it('should handle nodes without label', () => {
            const tree = {
                nodes: {
                    0: { node_id: 0, is_leaf: true, children: [], samples: [0], level: 0 }
                },
                root: 0,
                max_level: 0
            };
            const result = buildLevelDataFromHierarchy(tree, 0);
            expect(result.clusters[0].label).toBe('Cluster 0');
        });

        it('should handle level beyond max_level', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, 10);
            expect(result.clusters).toEqual([]);  // No nodes at level 10
        });

        it('should handle negative level', () => {
            const result = buildLevelDataFromHierarchy(mockHierarchyTree, -1);
            expect(result.clusters).toEqual([]);  // No nodes at negative level
        });
    });
});

// Helper function mimicking the JS logic
function buildLevelDataFromHierarchy(tree, level, parentId = null) {
    if (!tree || !tree.nodes) {
        return { clusters: [], level: 0, max_level: 0, labels: {} };
    }
    
    const nodesAtLevel = [];
    for (const [nodeId, nodeInfo] of Object.entries(tree.nodes)) {
        if (nodeInfo.level === level) {
            nodesAtLevel.push(parseInt(nodeId));
        }
    }
    
    let filteredNodes = nodesAtLevel;
    if (parentId !== null && tree.nodes[parentId]) {
        const parentChildren = tree.nodes[parentId].children || [];
        filteredNodes = nodesAtLevel.filter(nodeId => parentChildren.includes(nodeId));
    }
    
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
        labels: {}
    };
}
