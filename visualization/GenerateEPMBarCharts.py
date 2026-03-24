#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate EPM Bar Charts - Complete Style Replication
=====================================================
Creates a 2x3 grid of bar charts comparing Control vs pp3r1 groups
with exact styling matching the reference image.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
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


def calculate_open_arms_metrics(directory):
    """Calculate all open arms metrics from statistics files"""
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
    
    # Calculate open arms metrics
    df['OpenArms_Time_sec'] = df['Left_sec'] + df['Right_sec']
    df['OpenArms_Entries'] = df['Left_entries'] + df['Right_entries']
    df['OpenArms_Time_pct'] = (df['OpenArms_Time_sec'] / df['Total_ROI_sec']) * 100
    df['OpenArms_Entries_pct'] = (df['OpenArms_Entries'] / df['Total_entries']) * 100
    
    return df


def perform_statistical_test(control_data, pp3r1_data):
    """Perform t-test and return p-value and significance level"""
    t_stat, p_value = stats.ttest_ind(control_data, pp3r1_data)
    
    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'
    
    return p_value, sig_level


def create_bar_chart(ax, control_mean, control_sem, pp3r1_mean, pp3r1_sem,
                     ylabel, title, p_value, sig_level, y_max=None):
    """Create a single bar chart with exact styling matching reference image"""
    
    # Data for plotting
    groups = ['Control', 'pp3r1']
    # Adjust x positions to create better spacing between bars
    bar_width = 0.65 * (2/3) * (2/3) * (3/4) * (3/4) / 1.5 / 2   # Make bars thinner: divide by 1.5, then by 2 (1/2)
    # Calculate spacing so that gap between bar edges = bar_width/2
    # gap = (x_pos[1] - bar_width/2) - (x_pos[0] + bar_width/2) = bar_spacing - bar_width
    # We want: gap = bar_width/2, so: bar_spacing - bar_width = bar_width/2
    # Therefore: bar_spacing = bar_width * 1.5
    bar_spacing = bar_width * 1.5  # Spacing between bar centers so edge gap = bar_width/2
    x_pos = [-bar_spacing/2, bar_spacing/2]  # Center bars with spacing
    means = [control_mean, pp3r1_mean]
    sems = [control_sem, pp3r1_sem]
    # Colors extracted from reference image (image.png) using ExtractColorsFromImage.py
    # Control: Blue, pp3r1: Orange (exact colors from reference image)
    colors = ['#5e7eb8', '#e09061']  # Extracted from reference image.png
    
    # Create bars with exact styling matching reference
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                   color=colors, width=bar_width, edgecolor='none',
                   error_kw={'elinewidth': 2.0, 'capthick': 2.0, 'capsize': 5},
                   zorder=3)
    
    # Set y-axis limits: max value should be 3/2 (1.5x) of the highest bar (including error bar)
    if y_max is None:
        # Find the highest point including error bars
        max_bar_value = max([control_mean + control_sem, pp3r1_mean + pp3r1_sem])
        y_max = max_bar_value * 1.5  # 3/2 of the highest value
    else:
        # If y_max is provided, still calculate based on actual data but use provided as minimum
        max_bar_value = max([control_mean + control_sem, pp3r1_mean + pp3r1_sem])
        calculated_max = max_bar_value * 1.5
        y_max = max(y_max, calculated_max)  # Use the larger of the two
    ax.set_ylim(0, y_max)
    
    # Add significance bracket if significant
    if p_value < 0.05:
        # Calculate bracket position - above the higher bar with error bar
        control_top = control_mean + control_sem
        pp3r1_top = pp3r1_mean + pp3r1_sem
        max_val = max([control_top, pp3r1_top])
        bracket_spacing = y_max * 0.05
        
        # Bracket height above max value
        bracket_y = max_val + bracket_spacing
        bracket_height = bracket_spacing * 0.4
        
        # Draw horizontal bracket connecting the two bars
        # Use actual x positions for bracket
        control_x = x_pos[0]
        pp3r1_x = x_pos[1]
        # Left vertical line (from Control bar top to bracket)
        ax.plot([control_x, control_x], [control_top, bracket_y], 
                'k-', linewidth=1.2, clip_on=False)
        # Horizontal line connecting both bars
        ax.plot([control_x, pp3r1_x], [bracket_y, bracket_y], 
                'k-', linewidth=1.2, clip_on=False)
        # Right vertical line (from pp3r1 bar top to bracket)
        ax.plot([pp3r1_x, pp3r1_x], [pp3r1_top, bracket_y], 
                'k-', linewidth=1.2, clip_on=False)
        
        # Update text position to center between bars
        text_x = (control_x + pp3r1_x) / 2
        
        # Add significance text above bracket
        ax.text(text_x, bracket_y + bracket_height*0.7, sig_level, 
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                family='Arial')
    
    # Styling to match reference image exactly
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', family='Arial')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, family='Arial')
    # Set x-axis ticks and labels at bar positions
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11, fontweight='bold', family='Arial')
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    
    # Grid styling - subtle horizontal lines
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7, zorder=0, color='gray')
    ax.set_axisbelow(True)
    
    # Spine styling - clean borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Set y-axis ticks to be clean and evenly spaced
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))


def generate_epm_bar_charts(directory, output_path=None):
    """Generate the complete 2x3 grid of bar charts"""
    
    print("=" * 80)
    print("Generating EPM Bar Charts")
    print("=" * 80)
    
    # Calculate metrics
    df = calculate_open_arms_metrics(directory)
    if df is None:
        return
    
    # Separate groups
    control_df = df[df['Group'] == 'Control'].copy()
    pp3r1_df = df[df['Group'] == 'pp3r1'].copy()
    
    print(f"Control group: {len(control_df)} animals")
    print(f"pp3r1 group: {len(pp3r1_df)} animals")
    
    # Calculate means and SEMs for all metrics
    metrics = {
        'Time in Open Arms (s)': {
            'control_col': 'OpenArms_Time_sec',
            'pp3r1_col': 'OpenArms_Time_sec',
            'ylabel': 'Time in Open Arms (s)',
            'y_max': 200
        },
        'Entries into Open Arms': {
            'control_col': 'OpenArms_Entries',
            'pp3r1_col': 'OpenArms_Entries',
            'ylabel': 'Entries into Open Arms',
            'y_max': 30
        },
        '% Time in Open Arms': {
            'control_col': 'OpenArms_Time_pct',
            'pp3r1_col': 'OpenArms_Time_pct',
            'ylabel': '% Time in Open Arms',
            'y_max': 60
        },
        '% Entries into Open Arms': {
            'control_col': 'OpenArms_Entries_pct',
            'pp3r1_col': 'OpenArms_Entries_pct',
            'ylabel': '% Entries into Open Arms',
            'y_max': 60
        },
        'Total Distance (mm)': {
            'control_col': 'Distance_mm',
            'pp3r1_col': 'Distance_mm',
            'ylabel': 'Total Distance (mm)',
            'y_max': 20000
        },
        'Total Entries': {
            'control_col': 'Total_entries',
            'pp3r1_col': 'Total_entries',
            'ylabel': 'Total Entries',
            'y_max': 50
        }
    }
    
    # Create figure with 2x3 subplots - no title to match reference
    # Set figure style to match reference
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='white')
    # Remove title to match reference image style
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Metric order matching the image layout
    metric_order = [
        'Time in Open Arms (s)',
        'Entries into Open Arms',
        '% Time in Open Arms',
        '% Entries into Open Arms',
        'Total Distance (mm)',
        'Total Entries'
    ]
    
    # Generate each subplot
    for idx, metric_name in enumerate(metric_order):
        ax = axes_flat[idx]
        metric_info = metrics[metric_name]
        
        # Get data
        control_data = control_df[metric_info['control_col']].values
        pp3r1_data = pp3r1_df[metric_info['pp3r1_col']].values
        
        # Calculate means and SEMs
        control_mean = np.mean(control_data)
        control_sem = stats.sem(control_data)
        pp3r1_mean = np.mean(pp3r1_data)
        pp3r1_sem = stats.sem(pp3r1_data)
        
        # Perform statistical test
        p_value, sig_level = perform_statistical_test(control_data, pp3r1_data)
        
        print(f"\n{metric_name}:")
        print(f"  Control: {control_mean:.2f} ± {control_sem:.2f}")
        print(f"  pp3r1: {pp3r1_mean:.2f} ± {pp3r1_sem:.2f}")
        print(f"  p-value: {p_value:.4f} ({sig_level})")
        
        # Create bar chart (y_max=None to auto-calculate as 1.5x of highest bar)
        create_bar_chart(ax, control_mean, control_sem, pp3r1_mean, pp3r1_sem,
                        metric_info['ylabel'], metric_name, p_value, sig_level,
                        y_max=None)  # Auto-calculate as 3/2 of highest bar value
    
    # Adjust layout - spacing to match reference
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Save figure
    if output_path is None:
        output_path = Path(directory) / 'EPM_BarCharts.png'
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[SAVED] Bar charts saved to: {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] Bar charts saved to: {pdf_path}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EPM bar charts')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing statistics files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    
    args = parser.parse_args()
    
    generate_epm_bar_charts(args.directory, args.output)
