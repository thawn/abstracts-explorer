# 3D Clustering Visualization Feature

## Overview

This feature adds the ability to switch between 2D and 3D clustering visualizations in the Abstracts Explorer web interface.

## Changes Made

### 1. Frontend Changes (clustering.js)

#### Added n_components Configuration Option
- Added a "Visualization Dimensions" dropdown in the cluster settings modal
- Users can now choose between "2D" and "3D" visualization
- The selection is stored in `currentClusterConfig.n_components`

#### Updated visualizeClusters() Function
- Detects 3D data by checking `clusterData.n_dimensions === 3`
- Switches plot type from `scatter` to `scatter3d` for 3D visualization
- Adds z-coordinates to all data points when in 3D mode
- Configures 3D-specific settings:
  - Uses `scene` layout with xaxis, yaxis, and zaxis configuration
  - Sets camera eye position to (1.5, 1.5, 1.5) for optimal viewing angle
  - Reduces marker size from 8 to 4 for better visibility in 3D
  - Uses diamond markers for cluster centers (instead of stars which don't render well in 3D)
  - Adjusts marker line width for 3D rendering

#### Updated updateClusterVisualization() Function
- Same 3D detection and rendering logic as visualizeClusters()
- Ensures cluster filtering works correctly in both 2D and 3D modes
- Updates plot title to indicate "3D" when in 3D mode

#### Updated applyClusterSettings() Function
- Reads the n_components value from the settings modal
- Passes it to the backend API when recomputing clusters

### 2. Backend Changes

No backend changes were necessary! The backend already fully supported 3D:

- `clustering.py::reduce_dimensions()` already accepts `n_components` parameter (2 or 3)
- `clustering.py::get_clustering_results()` automatically includes z-coordinates when `reduced_embeddings.shape[1] > 2`
- `clustering.py::_calculate_cluster_centers()` automatically includes z-coordinates for 3D cluster centers

### 3. Test Coverage

Added two new test cases to `tests/test_clustering.py`:

1. **test_reduce_dimensions_3d()**
   - Validates that PCA reduction to 3 components produces correct shape
   - Ensures reduced_embeddings has shape (n_samples, 3)

2. **test_get_clustering_results_3d()**
   - Validates that clustering results include z-coordinates for 3D
   - Checks that all points have x, y, z coordinates
   - Verifies cluster centers also have x, y, z coordinates
   - Confirms n_dimensions is correctly set to 3

All 25 tests pass successfully.

## User Experience

### How to Use

1. Navigate to the "Clustering" tab in the web interface
2. Click the settings icon (⚙️) to open cluster settings
3. In the "Visualization Dimensions" dropdown, select either:
   - **2D**: Traditional 2D scatter plot (faster, simpler)
   - **3D**: Interactive 3D scatter plot (more detailed, can reveal additional patterns)
4. Click "Apply & Recompute" to regenerate clusters with the selected dimensionality
5. Interact with the 3D plot:
   - Click and drag to rotate
   - Scroll to zoom
   - Hover over points to see details

### Benefits of 3D Visualization

- **More Information**: 3D uses more dimensions of the reduced embedding space
- **Better Separation**: Clusters that overlap in 2D may be clearly separated in 3D
- **Pattern Discovery**: Reveals spatial relationships not visible in 2D
- **Interactive**: Rotation allows viewing from different angles

### Trade-offs

- 3D plots are slightly more computationally intensive
- May be harder to interpret for some users
- Requires more interaction to explore all angles
- Marker sizes are smaller to avoid visual clutter

## Technical Details

### Plotly Integration

The feature leverages Plotly.js's built-in 3D scatter plot support:

```javascript
// 2D plot
{
    type: 'scatter',
    x: [...],
    y: [...]
}

// 3D plot
{
    type: 'scatter3d',
    x: [...],
    y: [...],
    z: [...]
}
```

### Layout Configuration

3D plots use a different layout structure:

```javascript
// 2D layout
{
    xaxis: { ... },
    yaxis: { ... }
}

// 3D layout
{
    scene: {
        xaxis: { ... },
        yaxis: { ... },
        zaxis: { ... },
        camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
        }
    }
}
```

### Marker Adjustments

| Property | 2D | 3D |
|----------|----|----|
| Size | 8 | 4 |
| Line Width | 0.5 | 0.2 |
| Center Symbol | star | diamond |
| Center Size | 16 | 8 |

## API Changes

### Request to /api/clusters/compute

Added parameter:
```json
{
    "n_components": 3  // Can be 2 or 3
}
```

### Response from /api/clusters/compute

When n_components=3, response includes:
```json
{
    "points": [
        {
            "x": 1.23,
            "y": 4.56,
            "z": 7.89,  // New: only present in 3D
            "cluster": 0,
            "title": "Paper Title"
        }
    ],
    "cluster_centers": {
        "0": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0  // New: only present in 3D
        }
    },
    "n_dimensions": 3  // Indicates 3D data
}
```

## Files Modified

1. `src/abstracts_explorer/web_ui/static/modules/clustering.js` - Main implementation
2. `tests/test_clustering.py` - Added 3D test coverage

## Compatibility

- ✅ Backward compatible: Existing 2D functionality unchanged
- ✅ No database migrations required
- ✅ No API breaking changes
- ✅ Works with all clustering methods (kmeans, dbscan, agglomerative)
- ✅ Works with all reduction methods (PCA, t-SNE)

## Future Enhancements

Possible improvements for future iterations:

1. Add a toggle button to switch between 2D/3D without recomputing
2. Add animation to smoothly transition between views
3. Support 4D+ visualization with color or size encoding
4. Add export functionality for 3D plots
5. Allow customization of camera angle
6. Add VR/AR support for immersive exploration
