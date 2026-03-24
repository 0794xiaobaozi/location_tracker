#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate complete group statistics with both TIME and ENTRY COUNTS for each ROI
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


def generate_full_statistics(directory):
    """Generate complete group statistics with time and entries"""
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
    
    rois = ['Left', 'Right', 'Top', 'Bottom']
    
    print("=" * 150)
    print("COMPLETE GROUP STATISTICS: pp3r1 vs Control")
    print("=" * 150)
    print(f"pp3r1 group: {len(pp3r1_df)} animals")
    print(f"Control group: {len(control_df)} animals")
    print("=" * 150)
    
    # ===== TABLE 1: pp3r1 Group - Complete Data =====
    print()
    print("=" * 150)
    print("TABLE 1: pp3r1 GROUP (n={})".format(len(pp3r1_df)))
    print("=" * 150)
    
    # Header with all columns
    header = f"{'Animal':<8}"
    for roi in rois:
        header += f" | {roi+'_T':>7} | {roi+'_N':>5}"
    header += f" | {'Dist':>8}"
    print(header)
    print("-" * 130)
    
    # Data rows
    for _, row in pp3r1_df.iterrows():
        line = f"{row['Video']:<8}"
        for roi in rois:
            time_val = row[f'{roi}_sec']
            entry_val = row[f'{roi}_entries']
            line += f" | {time_val:>6.1f}s | {entry_val:>5}"
        line += f" | {row['Distance_mm']:>7.0f}mm"
        print(line)
    
    # Summary row
    print("-" * 130)
    line = f"{'Mean':<8}"
    for roi in rois:
        mean_time = pp3r1_df[f'{roi}_sec'].mean()
        mean_entry = pp3r1_df[f'{roi}_entries'].mean()
        line += f" | {mean_time:>6.1f}s | {mean_entry:>5.1f}"
    line += f" | {pp3r1_df['Distance_mm'].mean():>7.0f}mm"
    print(line)
    
    line = f"{'SEM':<8}"
    n = len(pp3r1_df)
    for roi in rois:
        sem_time = pp3r1_df[f'{roi}_sec'].std() / np.sqrt(n)
        sem_entry = pp3r1_df[f'{roi}_entries'].std() / np.sqrt(n)
        line += f" | {sem_time:>6.1f}s | {sem_entry:>5.1f}"
    line += f" | {pp3r1_df['Distance_mm'].std()/np.sqrt(n):>7.0f}mm"
    print(line)
    print("-" * 130)
    
    # ===== TABLE 2: Control Group - Complete Data =====
    print()
    print("=" * 150)
    print("TABLE 2: CONTROL GROUP (n={})".format(len(control_df)))
    print("=" * 150)
    
    # Header
    header = f"{'Animal':<8}"
    for roi in rois:
        header += f" | {roi+'_T':>7} | {roi+'_N':>5}"
    header += f" | {'Dist':>8}"
    print(header)
    print("-" * 130)
    
    # Data rows
    for _, row in control_df.iterrows():
        line = f"{row['Video']:<8}"
        for roi in rois:
            time_val = row[f'{roi}_sec']
            entry_val = row[f'{roi}_entries']
            line += f" | {time_val:>6.1f}s | {entry_val:>5}"
        line += f" | {row['Distance_mm']:>7.0f}mm"
        print(line)
    
    # Summary row
    print("-" * 130)
    line = f"{'Mean':<8}"
    for roi in rois:
        mean_time = control_df[f'{roi}_sec'].mean()
        mean_entry = control_df[f'{roi}_entries'].mean()
        line += f" | {mean_time:>6.1f}s | {mean_entry:>5.1f}"
    line += f" | {control_df['Distance_mm'].mean():>7.0f}mm"
    print(line)
    
    line = f"{'SEM':<8}"
    n = len(control_df)
    for roi in rois:
        sem_time = control_df[f'{roi}_sec'].std() / np.sqrt(n)
        sem_entry = control_df[f'{roi}_entries'].std() / np.sqrt(n)
        line += f" | {sem_time:>6.1f}s | {sem_entry:>5.1f}"
    line += f" | {control_df['Distance_mm'].std()/np.sqrt(n):>7.0f}mm"
    print(line)
    print("-" * 130)
    
    # ===== Save to Excel-friendly CSV =====
    print()
    print("=" * 150)
    print("SAVING CSV FILES...")
    print("=" * 150)
    
    # pp3r1 full table
    pp3r1_export = pp3r1_df[['Video']].copy()
    for roi in rois:
        pp3r1_export[f'{roi}_Time_sec'] = pp3r1_df[f'{roi}_sec']
        pp3r1_export[f'{roi}_Entries'] = pp3r1_df[f'{roi}_entries']
    pp3r1_export['Distance_mm'] = pp3r1_df['Distance_mm']
    
    # Add summary rows
    summary_mean = {'Video': 'Mean'}
    summary_sem = {'Video': 'SEM'}
    n = len(pp3r1_df)
    for roi in rois:
        summary_mean[f'{roi}_Time_sec'] = pp3r1_df[f'{roi}_sec'].mean()
        summary_mean[f'{roi}_Entries'] = pp3r1_df[f'{roi}_entries'].mean()
        summary_sem[f'{roi}_Time_sec'] = pp3r1_df[f'{roi}_sec'].std() / np.sqrt(n)
        summary_sem[f'{roi}_Entries'] = pp3r1_df[f'{roi}_entries'].std() / np.sqrt(n)
    summary_mean['Distance_mm'] = pp3r1_df['Distance_mm'].mean()
    summary_sem['Distance_mm'] = pp3r1_df['Distance_mm'].std() / np.sqrt(n)
    
    pp3r1_export = pd.concat([pp3r1_export, pd.DataFrame([summary_mean, summary_sem])], ignore_index=True)
    pp3r1_export.to_csv(directory / 'Table_pp3r1_Full.csv', index=False)
    print(f"[SAVED] Table_pp3r1_Full.csv")
    
    # Control full table
    control_export = control_df[['Video']].copy()
    for roi in rois:
        control_export[f'{roi}_Time_sec'] = control_df[f'{roi}_sec']
        control_export[f'{roi}_Entries'] = control_df[f'{roi}_entries']
    control_export['Distance_mm'] = control_df['Distance_mm']
    
    # Add summary rows
    summary_mean = {'Video': 'Mean'}
    summary_sem = {'Video': 'SEM'}
    n = len(control_df)
    for roi in rois:
        summary_mean[f'{roi}_Time_sec'] = control_df[f'{roi}_sec'].mean()
        summary_mean[f'{roi}_Entries'] = control_df[f'{roi}_entries'].mean()
        summary_sem[f'{roi}_Time_sec'] = control_df[f'{roi}_sec'].std() / np.sqrt(n)
        summary_sem[f'{roi}_Entries'] = control_df[f'{roi}_entries'].std() / np.sqrt(n)
    summary_mean['Distance_mm'] = control_df['Distance_mm'].mean()
    summary_sem['Distance_mm'] = control_df['Distance_mm'].std() / np.sqrt(n)
    
    control_export = pd.concat([control_export, pd.DataFrame([summary_mean, summary_sem])], ignore_index=True)
    control_export.to_csv(directory / 'Table_Control_Full.csv', index=False)
    print(f"[SAVED] Table_Control_Full.csv")
    
    print()
    print("=" * 150)
    print("COMPLETE!")
    print("=" * 150)
    
    return pp3r1_df, control_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate full group statistics')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing statistics files')
    
    args = parser.parse_args()
    
    generate_full_statistics(args.directory)


