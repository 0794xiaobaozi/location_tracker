"""
Batch Generate Transformed Trajectories for All Videos
=======================================================
This script processes all LocationOutput.csv files and generates
TransformedTrajectory.csv files for group mean heatmap generation.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from RegionalTransformVisualizer import (
    create_ideal_plus_maze,
    define_regions,
    transform_point_regional
)

# Configuration
video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"

# Original polygon vertices (from notebook, left arm adjusted)
original_vertices = [
    (61, 244), (68, 341), (405, 331), (424, 569), 
    (536, 569), (525, 323), (822, 303), (821, 195), 
    (514, 206), (474, 1), (372, 0), (396, 211)
]

# Convert to numpy array
src_vertices = np.array(original_vertices, dtype=np.float32)

# Create ideal Plus Maze
center_size = 119
arm_length_ratio = 5
canvas_size = 1409
dst_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
regions = define_regions(src_vertices)

# Find all LocationOutput.csv files
all_location_csv = glob.glob(os.path.join(video_dir, "*_LocationOutput.csv"))

print("=" * 70)
print("Batch Generate Transformed Trajectories")
print("=" * 70)
print(f"Found {len(all_location_csv)} LocationOutput.csv files\n")

processed = 0
skipped = 0

for i, csv_path in enumerate(all_location_csv, 1):
    basename = os.path.basename(csv_path)
    video_name = basename.replace('_LocationOutput.csv', '')
    transformed_csv = csv_path.replace('_LocationOutput.csv', '_TransformedTrajectory.csv')
    
    # Skip if already exists
    if os.path.exists(transformed_csv):
        print(f"[{i}/{len(all_location_csv)}] Skipping (already exists): {video_name}")
        skipped += 1
        continue
    
    try:
        # Load trajectory data
        df = pd.read_csv(csv_path)
        
        # Extract X, Y coordinates and Frame numbers
        x_coords = df['X'].values
        y_coords = df['Y'].values
        frames = df['Frame'].values if 'Frame' in df.columns else np.arange(len(df))
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        frames = frames[valid_mask]
        
        if len(x_coords) == 0:
            print(f"[{i}/{len(all_location_csv)}] Skipping (no valid points): {video_name}")
            skipped += 1
            continue
        
        # Transform all trajectory points
        transformed_points = []
        transformed_regions = []
        
        for x, y in zip(x_coords, y_coords):
            tx, ty, region = transform_point_regional(x, y, src_vertices, dst_vertices, regions)
            transformed_points.append([tx, ty])
            transformed_regions.append(region)
        
        transformed_points = np.array(transformed_points)
        
        # Create DataFrame with transformed coordinates
        transformed_df = pd.DataFrame({
            'Frame': frames,
            'X_Original': x_coords,
            'Y_Original': y_coords,
            'X_Transformed': transformed_points[:, 0],
            'Y_Transformed': transformed_points[:, 1],
            'Region': transformed_regions
        })
        
        # Add original CSV columns if they exist
        if 'Distance_px' in df.columns:
            transformed_df['Distance_px'] = df['Distance_px'].values[valid_mask]
        if 'Distance_mm' in df.columns:
            transformed_df['Distance_mm'] = df['Distance_mm'].values[valid_mask]
        if 'ROI_location' in df.columns:
            transformed_df['ROI_location'] = df['ROI_location'].values[valid_mask]
        
        # Save transformed trajectory
        transformed_df.to_csv(transformed_csv, index=False)
        print(f"[{i}/{len(all_location_csv)}] Processed: {video_name} ({len(transformed_points)} points)")
        processed += 1
        
    except Exception as e:
        print(f"[{i}/{len(all_location_csv)}] Error processing {video_name}: {e}")
        skipped += 1
        continue

print(f"\n{'='*70}")
print(f"Summary:")
print(f"  Processed: {processed} files")
print(f"  Skipped: {skipped} files")
print(f"  Total: {len(all_location_csv)} files")
print(f"{'='*70}")
