#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate group-wise statistics for pp3r1 vs Control groups

pp3r1 group: Videos starting with "3p" (3pr*, 3pl*)
Control group: All other videos (cl*, cn*, cq*, cr*)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def classify_group(video_name):
    """Classify video into group based on name"""
    if video_name.startswith('3p'):
        return 'pp3r1'
    else:
        return 'Control'


def generate_group_statistics(directory):
    """Generate group-wise statistics"""
    directory = Path(directory)
    
    # Load both CSV files
    entry_file = directory / 'ROI_Entry_Statistics.csv'
    time_file = directory / 'ROI_Statistics_Detailed.csv'
    
    if not entry_file.exists() or not time_file.exists():
        print("[ERROR] Statistics files not found")
        return None
    
    entry_df = pd.read_csv(entry_file)
    time_df = pd.read_csv(time_file)
    
    # Merge dataframes
    df = pd.merge(entry_df, time_df, on='Video')
    
    # Add group column
    df['Group'] = df['Video'].apply(classify_group)
    
    # Separate groups
    pp3r1_df = df[df['Group'] == 'pp3r1'].copy()
    control_df = df[df['Group'] == 'Control'].copy()
    
    print("=" * 120)
    print("GROUP STATISTICS: pp3r1 vs Control")
    print("=" * 120)
    print(f"pp3r1 group: {len(pp3r1_df)} videos")
    print(f"Control group: {len(control_df)} videos")
    print("=" * 120)
    print()
    
    # ===== TABLE 1: pp3r1 Group =====
    print("=" * 120)
    print("TABLE 1: pp3r1 GROUP (3p* videos)")
    print("=" * 120)
    
    # Individual data
    print("\n--- Individual Data ---")
    print(f"{'Video':<10} | {'Left':>8} | {'Right':>8} | {'Top':>8} | {'Bottom':>8} | {'Total':>8} | {'Distance':>10}")
    print("-" * 90)
    print("TIME SPENT (seconds):")
    for _, row in pp3r1_df.iterrows():
        print(f"{row['Video']:<10} | {row['Left_sec']:>7.1f}s | {row['Right_sec']:>7.1f}s | {row['Top_sec']:>7.1f}s | {row['Bottom_sec']:>7.1f}s | {row['Total_ROI_sec']:>7.1f}s | {row['Distance_mm']:>8.0f}mm")
    
    print()
    print("ENTRY COUNTS:")
    print(f"{'Video':<10} | {'Left':>8} | {'Right':>8} | {'Top':>8} | {'Bottom':>8} | {'Total':>8}")
    print("-" * 70)
    for _, row in pp3r1_df.iterrows():
        print(f"{row['Video']:<10} | {row['Left_entries']:>8} | {row['Right_entries']:>8} | {row['Top_entries']:>8} | {row['Bottom_entries']:>8} | {row['Total_entries']:>8}")
    
    # Summary statistics for pp3r1
    print()
    print("--- pp3r1 Group Summary (Mean +/- SEM) ---")
    
    rois = ['Left', 'Right', 'Top', 'Bottom']
    
    # Time summary
    print("\nTIME SPENT:")
    print(f"{'ROI':<10} | {'Mean':>10} | {'SEM':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 60)
    for roi in rois:
        col = f'{roi}_sec'
        mean_val = pp3r1_df[col].mean()
        sem_val = pp3r1_df[col].std() / np.sqrt(len(pp3r1_df))
        min_val = pp3r1_df[col].min()
        max_val = pp3r1_df[col].max()
        print(f"{roi:<10} | {mean_val:>9.1f}s | {sem_val:>9.1f}s | {min_val:>9.1f}s | {max_val:>9.1f}s")
    
    # Total and Distance
    print(f"{'Total ROI':<10} | {pp3r1_df['Total_ROI_sec'].mean():>9.1f}s | {pp3r1_df['Total_ROI_sec'].std()/np.sqrt(len(pp3r1_df)):>9.1f}s | {pp3r1_df['Total_ROI_sec'].min():>9.1f}s | {pp3r1_df['Total_ROI_sec'].max():>9.1f}s")
    print(f"{'Distance':<10} | {pp3r1_df['Distance_mm'].mean():>8.0f}mm | {pp3r1_df['Distance_mm'].std()/np.sqrt(len(pp3r1_df)):>8.0f}mm | {pp3r1_df['Distance_mm'].min():>8.0f}mm | {pp3r1_df['Distance_mm'].max():>8.0f}mm")
    
    # Entry summary
    print("\nENTRY COUNTS:")
    print(f"{'ROI':<10} | {'Mean':>10} | {'SEM':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 60)
    for roi in rois:
        col = f'{roi}_entries'
        mean_val = pp3r1_df[col].mean()
        sem_val = pp3r1_df[col].std() / np.sqrt(len(pp3r1_df))
        min_val = pp3r1_df[col].min()
        max_val = pp3r1_df[col].max()
        print(f"{roi:<10} | {mean_val:>10.1f} | {sem_val:>10.1f} | {min_val:>10} | {max_val:>10}")
    print(f"{'Total':<10} | {pp3r1_df['Total_entries'].mean():>10.1f} | {pp3r1_df['Total_entries'].std()/np.sqrt(len(pp3r1_df)):>10.1f} | {pp3r1_df['Total_entries'].min():>10} | {pp3r1_df['Total_entries'].max():>10}")
    
    # ===== TABLE 2: Control Group =====
    print()
    print("=" * 120)
    print("TABLE 2: CONTROL GROUP (cl*, cn*, cq*, cr* videos)")
    print("=" * 120)
    
    # Individual data
    print("\n--- Individual Data ---")
    print(f"{'Video':<10} | {'Left':>8} | {'Right':>8} | {'Top':>8} | {'Bottom':>8} | {'Total':>8} | {'Distance':>10}")
    print("-" * 90)
    print("TIME SPENT (seconds):")
    for _, row in control_df.iterrows():
        print(f"{row['Video']:<10} | {row['Left_sec']:>7.1f}s | {row['Right_sec']:>7.1f}s | {row['Top_sec']:>7.1f}s | {row['Bottom_sec']:>7.1f}s | {row['Total_ROI_sec']:>7.1f}s | {row['Distance_mm']:>8.0f}mm")
    
    print()
    print("ENTRY COUNTS:")
    print(f"{'Video':<10} | {'Left':>8} | {'Right':>8} | {'Top':>8} | {'Bottom':>8} | {'Total':>8}")
    print("-" * 70)
    for _, row in control_df.iterrows():
        print(f"{row['Video']:<10} | {row['Left_entries']:>8} | {row['Right_entries']:>8} | {row['Top_entries']:>8} | {row['Bottom_entries']:>8} | {row['Total_entries']:>8}")
    
    # Summary statistics for Control
    print()
    print("--- Control Group Summary (Mean +/- SEM) ---")
    
    # Time summary
    print("\nTIME SPENT:")
    print(f"{'ROI':<10} | {'Mean':>10} | {'SEM':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 60)
    for roi in rois:
        col = f'{roi}_sec'
        mean_val = control_df[col].mean()
        sem_val = control_df[col].std() / np.sqrt(len(control_df))
        min_val = control_df[col].min()
        max_val = control_df[col].max()
        print(f"{roi:<10} | {mean_val:>9.1f}s | {sem_val:>9.1f}s | {min_val:>9.1f}s | {max_val:>9.1f}s")
    
    # Total and Distance
    print(f"{'Total ROI':<10} | {control_df['Total_ROI_sec'].mean():>9.1f}s | {control_df['Total_ROI_sec'].std()/np.sqrt(len(control_df)):>9.1f}s | {control_df['Total_ROI_sec'].min():>9.1f}s | {control_df['Total_ROI_sec'].max():>9.1f}s")
    print(f"{'Distance':<10} | {control_df['Distance_mm'].mean():>8.0f}mm | {control_df['Distance_mm'].std()/np.sqrt(len(control_df)):>8.0f}mm | {control_df['Distance_mm'].min():>8.0f}mm | {control_df['Distance_mm'].max():>8.0f}mm")
    
    # Entry summary
    print("\nENTRY COUNTS:")
    print(f"{'ROI':<10} | {'Mean':>10} | {'SEM':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 60)
    for roi in rois:
        col = f'{roi}_entries'
        mean_val = control_df[col].mean()
        sem_val = control_df[col].std() / np.sqrt(len(control_df))
        min_val = control_df[col].min()
        max_val = control_df[col].max()
        print(f"{roi:<10} | {mean_val:>10.1f} | {sem_val:>10.1f} | {min_val:>10} | {max_val:>10}")
    print(f"{'Total':<10} | {control_df['Total_entries'].mean():>10.1f} | {control_df['Total_entries'].std()/np.sqrt(len(control_df)):>10.1f} | {control_df['Total_entries'].min():>10} | {control_df['Total_entries'].max():>10}")
    
    # ===== GROUP COMPARISON =====
    print()
    print("=" * 120)
    print("GROUP COMPARISON: pp3r1 vs Control")
    print("=" * 120)
    
    print("\nTIME SPENT (Mean +/- SEM):")
    print(f"{'ROI':<10} | {'pp3r1':>18} | {'Control':>18} | {'Diff':>12}")
    print("-" * 70)
    for roi in rois:
        col = f'{roi}_sec'
        pp3r1_mean = pp3r1_df[col].mean()
        pp3r1_sem = pp3r1_df[col].std() / np.sqrt(len(pp3r1_df))
        ctrl_mean = control_df[col].mean()
        ctrl_sem = control_df[col].std() / np.sqrt(len(control_df))
        diff = pp3r1_mean - ctrl_mean
        diff_sign = '+' if diff > 0 else ''
        print(f"{roi:<10} | {pp3r1_mean:>6.1f} +/- {pp3r1_sem:>5.1f}s | {ctrl_mean:>6.1f} +/- {ctrl_sem:>5.1f}s | {diff_sign}{diff:>6.1f}s")
    
    # Distance comparison
    pp3r1_dist = pp3r1_df['Distance_mm'].mean()
    pp3r1_dist_sem = pp3r1_df['Distance_mm'].std() / np.sqrt(len(pp3r1_df))
    ctrl_dist = control_df['Distance_mm'].mean()
    ctrl_dist_sem = control_df['Distance_mm'].std() / np.sqrt(len(control_df))
    diff_dist = pp3r1_dist - ctrl_dist
    diff_sign = '+' if diff_dist > 0 else ''
    print(f"{'Distance':<10} | {pp3r1_dist:>5.0f} +/- {pp3r1_dist_sem:>4.0f}mm | {ctrl_dist:>5.0f} +/- {ctrl_dist_sem:>4.0f}mm | {diff_sign}{diff_dist:>5.0f}mm")
    
    print("\nENTRY COUNTS (Mean +/- SEM):")
    print(f"{'ROI':<10} | {'pp3r1':>18} | {'Control':>18} | {'Diff':>12}")
    print("-" * 70)
    for roi in rois:
        col = f'{roi}_entries'
        pp3r1_mean = pp3r1_df[col].mean()
        pp3r1_sem = pp3r1_df[col].std() / np.sqrt(len(pp3r1_df))
        ctrl_mean = control_df[col].mean()
        ctrl_sem = control_df[col].std() / np.sqrt(len(control_df))
        diff = pp3r1_mean - ctrl_mean
        diff_sign = '+' if diff > 0 else ''
        print(f"{roi:<10} | {pp3r1_mean:>6.1f} +/- {pp3r1_sem:>5.1f}  | {ctrl_mean:>6.1f} +/- {ctrl_sem:>5.1f}  | {diff_sign}{diff:>6.1f}")
    
    pp3r1_total = pp3r1_df['Total_entries'].mean()
    pp3r1_total_sem = pp3r1_df['Total_entries'].std() / np.sqrt(len(pp3r1_df))
    ctrl_total = control_df['Total_entries'].mean()
    ctrl_total_sem = control_df['Total_entries'].std() / np.sqrt(len(control_df))
    diff_total = pp3r1_total - ctrl_total
    diff_sign = '+' if diff_total > 0 else ''
    print(f"{'Total':<10} | {pp3r1_total:>6.1f} +/- {pp3r1_total_sem:>5.1f}  | {ctrl_total:>6.1f} +/- {ctrl_total_sem:>5.1f}  | {diff_sign}{diff_total:>6.1f}")
    
    print()
    print("=" * 120)
    
    # Save to Excel-friendly CSV files
    # pp3r1 group
    pp3r1_export = pp3r1_df[['Video', 'Left_sec', 'Right_sec', 'Top_sec', 'Bottom_sec', 
                             'Total_ROI_sec', 'Distance_mm',
                             'Left_entries', 'Right_entries', 'Top_entries', 'Bottom_entries',
                             'Total_entries']].copy()
    pp3r1_export.to_csv(directory / 'Group_pp3r1_Statistics.csv', index=False)
    print(f"[SAVED] Group_pp3r1_Statistics.csv")
    
    # Control group
    control_export = control_df[['Video', 'Left_sec', 'Right_sec', 'Top_sec', 'Bottom_sec',
                                  'Total_ROI_sec', 'Distance_mm',
                                  'Left_entries', 'Right_entries', 'Top_entries', 'Bottom_entries',
                                  'Total_entries']].copy()
    control_export.to_csv(directory / 'Group_Control_Statistics.csv', index=False)
    print(f"[SAVED] Group_Control_Statistics.csv")
    
    # Summary comparison
    summary_data = []
    for roi in rois:
        summary_data.append({
            'Metric': f'{roi}_time_sec',
            'pp3r1_mean': pp3r1_df[f'{roi}_sec'].mean(),
            'pp3r1_sem': pp3r1_df[f'{roi}_sec'].std() / np.sqrt(len(pp3r1_df)),
            'pp3r1_n': len(pp3r1_df),
            'Control_mean': control_df[f'{roi}_sec'].mean(),
            'Control_sem': control_df[f'{roi}_sec'].std() / np.sqrt(len(control_df)),
            'Control_n': len(control_df),
        })
        summary_data.append({
            'Metric': f'{roi}_entries',
            'pp3r1_mean': pp3r1_df[f'{roi}_entries'].mean(),
            'pp3r1_sem': pp3r1_df[f'{roi}_entries'].std() / np.sqrt(len(pp3r1_df)),
            'pp3r1_n': len(pp3r1_df),
            'Control_mean': control_df[f'{roi}_entries'].mean(),
            'Control_sem': control_df[f'{roi}_entries'].std() / np.sqrt(len(control_df)),
            'Control_n': len(control_df),
        })
    
    summary_data.append({
        'Metric': 'Total_entries',
        'pp3r1_mean': pp3r1_df['Total_entries'].mean(),
        'pp3r1_sem': pp3r1_df['Total_entries'].std() / np.sqrt(len(pp3r1_df)),
        'pp3r1_n': len(pp3r1_df),
        'Control_mean': control_df['Total_entries'].mean(),
        'Control_sem': control_df['Total_entries'].std() / np.sqrt(len(control_df)),
        'Control_n': len(control_df),
    })
    summary_data.append({
        'Metric': 'Distance_mm',
        'pp3r1_mean': pp3r1_df['Distance_mm'].mean(),
        'pp3r1_sem': pp3r1_df['Distance_mm'].std() / np.sqrt(len(pp3r1_df)),
        'pp3r1_n': len(pp3r1_df),
        'Control_mean': control_df['Distance_mm'].mean(),
        'Control_sem': control_df['Distance_mm'].std() / np.sqrt(len(control_df)),
        'Control_n': len(control_df),
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(directory / 'Group_Comparison_Summary.csv', index=False)
    print(f"[SAVED] Group_Comparison_Summary.csv")
    
    print()
    print("=" * 120)
    print("ANALYSIS COMPLETE!")
    print("=" * 120)
    
    return pp3r1_df, control_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate group statistics')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing statistics files')
    
    args = parser.parse_args()
    
    generate_group_statistics(args.directory)

