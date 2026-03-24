# Visualization Scripts Inventory

This document organizes existing visualization-related scripts in this project.

All visualization scripts are now placed under the `visualization/` subfolder.

## Scope

Current inventory focuses on scripts related to:

- trajectory visualization
- heatmaps
- statistical plots
- ROI/transform visualization tools
- interactive adjustment helpers

## Grouped Scripts

### Video Visualization

- `GenerateTrajectoryVideos.py` - Batch overlay of tracked points/trajectory on source videos.
- `GenerateTrackingVideo.py` - Single-video overlay pipeline (depends on `LocationTracking_Functions.py`).
- `GenerateTrackingVideo_Standalone.py` - Standalone single-video overlay with broader runtime compatibility.

### Static Trajectory Figures

- `GenerateTrajectoryImages.py` - Static trajectory figures on reference frame with ROI overlays.

### Heatmaps

- `GenerateEPMHeatmap_Standard.py` - Single-subject heatmap on standardized EPM coordinates (from transformed trajectory CSV).
- `GenerateGroupMeanHeatmap.py` - Group mean heatmap from multiple transformed trajectories.
- `GenerateTransformedTrajectoryHeatmap.py` - Heatmap generation with regional transform from raw location outputs.

### Statistical Plots / Reports

- `GenerateEPMBarCharts.py` - Group bar charts with statistical tests.
- `GenerateROIStatistics.py` - ROI dwell-time statistics export (table/CSV oriented).
- `GenerateROIEntryStatistics.py` - ROI entry/exit counts export (table/CSV oriented).
- `GenerateGroupStatistics.py` - Group summary report from ROI statistics (console/table oriented).
- `GenerateGroupStatistics_Full.py` - Extended group summary report (console/table oriented).

### Transform / Geometry Visualization

- `RegionalTransformVisualizer.py` - Regional transform core module with visualization/demo entry point.
- `VisualizeCrossMazeTransform.py` - Cross-maze transformation visualization utility.
- `BatchGenerateTransformedTrajectories.py` - Batch generation of transformed trajectories for downstream heatmaps.

### Interactive ROI/Vertex Adjustment Tools

- `AdjustLeftArmVertices.py` - Left-arm vertex visualization/check helper.
- `InteractiveAdjustLeftArm.py` - Click-based left-arm vertex adjustment.
- `DragAdjustLeftArm.py` - Drag-based left-arm vertex adjustment.
- `CheckROISize.py` - ROI size sanity-check utility.

## Overlap / Redundancy Candidates

- `GenerateTrackingVideo.py` and `GenerateTrackingVideo_Standalone.py` (same target output, different dependency strategy).
- `GenerateTrajectoryVideos.py` partially overlaps with tracking-video overlay scripts.
- `GenerateGroupStatistics.py` and `GenerateGroupStatistics_Full.py` (same input domain, different output granularity).
- Left-arm adjustment trio:
  - `AdjustLeftArmVertices.py`
  - `InteractiveAdjustLeftArm.py`
  - `DragAdjustLeftArm.py`

## Suggested Cleanup Plan

1. Keep one primary single-video overlay entry (`GenerateTrackingVideo_Standalone.py` preferred).
2. Keep one batch overlay entry (`GenerateTrajectoryVideos.py`).
3. Merge group statistics scripts into one CLI with format mode.
4. Keep one primary interactive vertex adjuster (prefer drag or click), archive others as legacy.
5. Keep `RegionalTransformVisualizer.py` as transform core and mark alternative transform scripts clearly as legacy/experimental if needed.

