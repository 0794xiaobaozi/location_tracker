#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Count the number of entries into each Functional ROI

For each video, analyzes the trajectory and counts:
- How many times the animal ENTERED each ROI
- Entry = transition from outside ROI to inside ROI

Four ROIs are non-overlapping: Left, Right, Top, Bottom
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import cv2

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def point_in_polygon(x, y, polygon_xs, polygon_ys):
    """Check if point (x, y) is inside polygon
    
    Uses cv2.pointPolygonTest for robust checking
    """
    if np.isnan(x) or np.isnan(y):
        return False
    
    # Create polygon array for OpenCV
    polygon = np.array(list(zip(polygon_xs, polygon_ys)), dtype=np.int32)
    
    # pointPolygonTest returns: >0 inside, =0 on edge, <0 outside
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0


def count_roi_entries(location_csv, roi_data, region_names):
    """Count entries into each ROI for one video
    
    Args:
        location_csv: Path to _LocationOutput.csv
        roi_data: Dictionary with 'xs' and 'ys' lists for each ROI polygon
        region_names: List of ROI names ['Left', 'Right', 'Top', 'Bottom']
    
    Returns:
        Dictionary with entry counts and additional stats
    """
    # Load location data
    df = pd.read_csv(location_csv)
    
    if 'X' not in df.columns or 'Y' not in df.columns:
        return None
    
    x_coords = df['X'].values
    y_coords = df['Y'].values
    
    n_rois = len(region_names)
    n_frames = len(x_coords)
    
    # Track which ROI the animal is in for each frame
    # -1 = not in any ROI, 0-3 = in ROI index
    roi_status = np.full(n_frames, -1, dtype=int)
    
    # Build polygon lists
    roi_polygons = []
    for i in range(n_rois):
        xs = roi_data['xs'][i]
        ys = roi_data['ys'][i]
        roi_polygons.append((xs, ys))
    
    # Determine ROI status for each frame
    for f in range(n_frames):
        x, y = x_coords[f], y_coords[f]
        if np.isnan(x) or np.isnan(y):
            roi_status[f] = -1
            continue
        
        # Check each ROI
        for roi_idx, (xs, ys) in enumerate(roi_polygons):
            if point_in_polygon(x, y, xs, ys):
                roi_status[f] = roi_idx
                break  # Assume non-overlapping ROIs
    
    # Count entries for each ROI
    # Entry = transition from (not in ROI) to (in ROI)
    entry_counts = {name: 0 for name in region_names}
    exit_counts = {name: 0 for name in region_names}
    
    for f in range(1, n_frames):
        prev_roi = roi_status[f - 1]
        curr_roi = roi_status[f]
        
        # Entry: was not in this ROI, now is in this ROI
        if curr_roi >= 0 and prev_roi != curr_roi:
            entry_counts[region_names[curr_roi]] += 1
        
        # Exit: was in this ROI, now is not in this ROI
        if prev_roi >= 0 and curr_roi != prev_roi:
            exit_counts[region_names[prev_roi]] += 1
    
    # Also count if first frame is in a ROI (counts as entry)
    if roi_status[0] >= 0:
        entry_counts[region_names[roi_status[0]]] += 1
    
    # Calculate time spent in each ROI (in frames)
    time_in_roi = {name: 0 for name in region_names}
    for f in range(n_frames):
        if roi_status[f] >= 0:
            time_in_roi[region_names[roi_status[f]]] += 1
    
    # Calculate average bout duration
    bout_durations = {name: [] for name in region_names}
    current_bout_start = None
    current_roi = None
    
    for f in range(n_frames):
        if roi_status[f] >= 0:
            roi_name = region_names[roi_status[f]]
            if current_roi != roi_name:
                # End previous bout
                if current_bout_start is not None and current_roi is not None:
                    bout_durations[current_roi].append(f - current_bout_start)
                # Start new bout
                current_bout_start = f
                current_roi = roi_name
        else:
            # End previous bout
            if current_bout_start is not None and current_roi is not None:
                bout_durations[current_roi].append(f - current_bout_start)
            current_bout_start = None
            current_roi = None
    
    # Close last bout if needed
    if current_bout_start is not None and current_roi is not None:
        bout_durations[current_roi].append(n_frames - current_bout_start)
    
    # Calculate average bout duration
    avg_bout = {}
    for name in region_names:
        if len(bout_durations[name]) > 0:
            avg_bout[name] = np.mean(bout_durations[name])
        else:
            avg_bout[name] = 0
    
    return {
        'entries': entry_counts,
        'exits': exit_counts,
        'time_frames': time_in_roi,
        'avg_bout_frames': avg_bout,
        'n_frames': n_frames
    }


def generate_entry_statistics(directory, fps=25.0):
    """Generate ROI entry statistics for all videos
    
    Args:
        directory: Directory containing BatchSummary.csv and LocationOutput files
        fps: Frames per second for time conversion
    """
    directory = Path(directory)
    summary_file = directory / 'BatchSummary.csv'
    
    if not summary_file.exists():
        print(f"[ERROR] BatchSummary.csv not found in {directory}")
        return None
    
    # Load batch summary to get ROI coordinates
    batch_df = pd.read_csv(summary_file)
    
    if len(batch_df) == 0:
        print(f"[ERROR] No data in BatchSummary.csv")
        return None
    
    # Parse ROI coordinates from first row (assumes all videos use same ROIs)
    roi_str = batch_df.iloc[0]['ROI_coordinates']
    try:
        roi_data = ast.literal_eval(roi_str)
    except:
        print(f"[ERROR] Cannot parse ROI coordinates")
        return None
    
    # Get region names
    region_names = ['Left', 'Right', 'Top', 'Bottom']
    n_rois = min(len(roi_data['xs']), len(region_names))
    region_names = region_names[:n_rois]
    
    print("=" * 100)
    print("ROI ENTRY STATISTICS")
    print("=" * 100)
    print(f"ROI Regions: {', '.join(region_names)}")
    print(f"FPS: {fps}")
    print("=" * 100)
    print()
    
    # Process each video
    all_stats = []
    
    for idx, row in batch_df.iterrows():
        video_name = row['File']
        csv_file = directory / video_name.replace('.mp4', '_LocationOutput.csv')
        
        if not csv_file.exists():
            print(f"[SKIP] {video_name}: No LocationOutput.csv")
            continue
        
        stats = count_roi_entries(csv_file, roi_data, region_names)
        
        if stats is None:
            print(f"[SKIP] {video_name}: Cannot process")
            continue
        
        video_stats = {'Video': video_name.replace('.mp4', '')}
        
        # Entry counts
        for name in region_names:
            video_stats[f'{name}_entries'] = stats['entries'][name]
        
        # Total entries
        video_stats['Total_entries'] = sum(stats['entries'].values())
        
        # Average bout duration (in seconds)
        for name in region_names:
            video_stats[f'{name}_avg_bout_sec'] = stats['avg_bout_frames'][name] / fps
        
        all_stats.append(video_stats)
    
    if not all_stats:
        print("[ERROR] No videos processed")
        return None
    
    stats_df = pd.DataFrame(all_stats)
    
    # Print entry count table
    print("ENTRY COUNTS (number of times entered each ROI)")
    print("-" * 100)
    
    header = f"{'Video':<12}"
    for name in region_names:
        header += f" | {name:>8}"
    header += f" | {'Total':>8}"
    print(header)
    print("-" * 100)
    
    for idx, row in stats_df.iterrows():
        line = f"{row['Video']:<12}"
        for name in region_names:
            line += f" | {row[f'{name}_entries']:>8}"
        line += f" | {row['Total_entries']:>8}"
        print(line)
    
    print("-" * 100)
    print()
    
    # Print average bout duration table
    print("AVERAGE BOUT DURATION (seconds per visit)")
    print("-" * 100)
    
    header = f"{'Video':<12}"
    for name in region_names:
        header += f" | {name:>10}"
    print(header)
    print("-" * 100)
    
    for idx, row in stats_df.iterrows():
        line = f"{row['Video']:<12}"
        for name in region_names:
            val = row[f'{name}_avg_bout_sec']
            if val > 0:
                line += f" | {val:>9.1f}s"
            else:
                line += f" | {'-':>10}"
        print(line)
    
    print("-" * 100)
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 100)
    
    for name in region_names:
        col = f'{name}_entries'
        mean_val = stats_df[col].mean()
        std_val = stats_df[col].std()
        min_val = stats_df[col].min()
        max_val = stats_df[col].max()
        
        bout_col = f'{name}_avg_bout_sec'
        mean_bout = stats_df[bout_col].mean()
        
        print(f"{name:>8}: {mean_val:>5.1f} +/- {std_val:>4.1f} entries  "
              f"[Range: {min_val:>3} - {max_val:>3}]  "
              f"Avg bout: {mean_bout:>5.1f}s")
    
    # Total entries
    mean_total = stats_df['Total_entries'].mean()
    std_total = stats_df['Total_entries'].std()
    print()
    print(f"{'TOTAL':>8}: {mean_total:>5.1f} +/- {std_total:>4.1f} entries per video")
    
    print("-" * 100)
    print()
    
    # Save CSV
    output_csv = directory / 'ROI_Entry_Statistics.csv'
    stats_df.to_csv(output_csv, index=False)
    print(f"[SAVED] {output_csv}")
    
    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    
    return stats_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Count ROI entries')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing BatchSummary.csv and LocationOutput files')
    parser.add_argument('--fps', type=float, default=25.0,
                       help='Video FPS (default: 25)')
    
    args = parser.parse_args()
    
    generate_entry_statistics(args.directory, args.fps)

