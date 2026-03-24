#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate detailed statistics for Functional ROI time spent

Reads BatchSummary.csv and generates:
1. Formatted console output
2. Excel-friendly CSV with time in seconds
3. Summary statistics (mean, std, etc.)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def generate_roi_statistics(directory, fps=25.0, total_frames=7500):
    """Generate comprehensive ROI statistics
    
    Args:
        directory: Directory containing BatchSummary.csv
        fps: Frames per second (default: 25)
        total_frames: Total frames per video (default: 7500)
    """
    directory = Path(directory)
    summary_file = directory / 'BatchSummary.csv'
    
    if not summary_file.exists():
        print(f"[ERROR] BatchSummary.csv not found in {directory}")
        return None
    
    # Load data
    df = pd.read_csv(summary_file)
    
    # Video duration in seconds
    video_duration = total_frames / fps
    
    # ROI columns
    roi_columns = ['Left', 'Right', 'Top', 'Bottom']
    
    # Check which ROI columns exist
    existing_rois = [col for col in roi_columns if col in df.columns]
    
    if not existing_rois:
        print(f"[ERROR] No ROI columns found in BatchSummary.csv")
        return None
    
    print("=" * 90)
    print("FUNCTIONAL ROI TIME STATISTICS")
    print("=" * 90)
    print(f"Video Duration: {video_duration:.1f} seconds ({total_frames} frames @ {fps} fps)")
    print(f"Total Videos: {len(df)}")
    print(f"ROI Regions: {', '.join(existing_rois)}")
    print("=" * 90)
    print()
    
    # Create detailed statistics dataframe
    stats_data = []
    
    for idx, row in df.iterrows():
        video_name = row['File']
        
        # Calculate time in seconds for each ROI
        roi_times = {}
        roi_percentages = {}
        
        for roi in existing_rois:
            if roi in row and pd.notna(row[roi]):
                proportion = row[roi]
                time_seconds = proportion * video_duration
                roi_times[f'{roi}_sec'] = time_seconds
                roi_percentages[f'{roi}_pct'] = proportion * 100
            else:
                roi_times[f'{roi}_sec'] = np.nan
                roi_percentages[f'{roi}_pct'] = np.nan
        
        # Get distance if available
        distance_mm = row.get('Distance_mm', np.nan)
        
        stats_data.append({
            'Video': video_name.replace('.mp4', ''),
            **roi_times,
            **roi_percentages,
            'Distance_mm': distance_mm,
            'Total_ROI_sec': sum(v for v in roi_times.values() if pd.notna(v)),
            'Total_ROI_pct': sum(v for v in roi_percentages.values() if pd.notna(v))
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Print detailed table
    print("DETAILED ROI TIME (seconds)")
    print("-" * 90)
    
    # Header
    header = f"{'Video':<12}"
    for roi in existing_rois:
        header += f" | {roi:>8}"
    header += f" | {'Total':>8} | {'Distance':>10}"
    print(header)
    print("-" * 90)
    
    # Data rows
    for idx, row in stats_df.iterrows():
        line = f"{row['Video']:<12}"
        for roi in existing_rois:
            val = row[f'{roi}_sec']
            if pd.notna(val):
                line += f" | {val:>7.1f}s"
            else:
                line += f" | {'N/A':>8}"
        total_roi = row['Total_ROI_sec']
        distance = row['Distance_mm']
        line += f" | {total_roi:>7.1f}s | {distance:>8.0f}mm"
        print(line)
    
    print("-" * 90)
    print()
    
    # Print percentage table
    print("DETAILED ROI TIME (percentage)")
    print("-" * 90)
    
    # Header
    header = f"{'Video':<12}"
    for roi in existing_rois:
        header += f" | {roi:>8}"
    header += f" | {'Total':>8} | {'Outside':>8}"
    print(header)
    print("-" * 90)
    
    # Data rows
    for idx, row in stats_df.iterrows():
        line = f"{row['Video']:<12}"
        for roi in existing_rois:
            val = row[f'{roi}_pct']
            if pd.notna(val):
                line += f" | {val:>7.1f}%"
            else:
                line += f" | {'N/A':>8}"
        total_pct = row['Total_ROI_pct']
        outside_pct = 100 - total_pct if pd.notna(total_pct) else np.nan
        line += f" | {total_pct:>7.1f}% | {outside_pct:>7.1f}%"
        print(line)
    
    print("-" * 90)
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 90)
    
    summary_stats = []
    for roi in existing_rois:
        sec_col = f'{roi}_sec'
        pct_col = f'{roi}_pct'
        
        mean_sec = stats_df[sec_col].mean()
        std_sec = stats_df[sec_col].std()
        mean_pct = stats_df[pct_col].mean()
        std_pct = stats_df[pct_col].std()
        
        summary_stats.append({
            'ROI': roi,
            'Mean_sec': mean_sec,
            'Std_sec': std_sec,
            'Mean_pct': mean_pct,
            'Std_pct': std_pct,
            'Min_sec': stats_df[sec_col].min(),
            'Max_sec': stats_df[sec_col].max()
        })
        
        print(f"{roi:>8}: {mean_sec:>6.1f}s +/- {std_sec:>5.1f}s  "
              f"({mean_pct:>5.1f}% +/- {std_pct:>4.1f}%)  "
              f"[Range: {stats_df[sec_col].min():>5.1f}s - {stats_df[sec_col].max():>5.1f}s]")
    
    # Distance statistics
    mean_dist = stats_df['Distance_mm'].mean()
    std_dist = stats_df['Distance_mm'].std()
    print()
    print(f"{'Distance':>8}: {mean_dist:>6.0f}mm +/- {std_dist:>5.0f}mm  "
          f"[Range: {stats_df['Distance_mm'].min():>5.0f}mm - {stats_df['Distance_mm'].max():>5.0f}mm]")
    
    print("-" * 90)
    print()
    
    # Save detailed CSV
    output_csv = directory / 'ROI_Statistics_Detailed.csv'
    
    # Create export dataframe
    export_df = stats_df.copy()
    export_df = export_df.round(2)
    export_df.to_csv(output_csv, index=False)
    print(f"[SAVED] Detailed statistics: {output_csv}")
    
    # Create summary CSV
    summary_csv = directory / 'ROI_Statistics_Summary.csv'
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVED] Summary statistics: {summary_csv}")
    
    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE!")
    print("=" * 90)
    
    return stats_df, summary_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ROI time statistics')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing BatchSummary.csv')
    parser.add_argument('--fps', type=float, default=25.0,
                       help='Video FPS (default: 25)')
    parser.add_argument('--frames', type=int, default=7500,
                       help='Total frames per video (default: 7500)')
    
    args = parser.parse_args()
    
    generate_roi_statistics(args.directory, args.fps, args.frames)

