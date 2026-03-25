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
import yaml

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def classify_group(video_name):
    """Classify video into group based on name"""
    if video_name.startswith('3p'):
        return 'pp3r1'
    else:
        return 'Control'


def normalize_video_id(video_name):
    """Normalize video identifiers across .mp4 / _LocationOutput naming variants."""
    v = str(video_name)
    if v.endswith("_LocationOutput"):
        v = v[: -len("_LocationOutput")]
    if v.lower().endswith(".mp4"):
        v = v[:-4]
    return v


def load_group_map_from_yaml(group_config_path):
    """Load external grouping YAML.

    Supported YAML structure:
    groups:
      Control: [1-1, 1-2]
      pp3r1: [3p-1, 3p-2]
    """
    with open(group_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Group YAML root must be a mapping/object.")

    groups = cfg.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise KeyError("Group YAML must contain a non-empty 'groups' mapping.")

    group_map = {}
    group_order = []
    for group_name, videos in groups.items():
        if not isinstance(videos, list):
            raise ValueError(f"Group '{group_name}' must map to a list of video IDs.")
        group_name = str(group_name)
        group_order.append(group_name)
        for item in videos:
            vid = normalize_video_id(item)
            if vid in group_map and group_map[vid] != group_name:
                raise ValueError(
                    f"Video '{vid}' is assigned to multiple groups: "
                    f"'{group_map[vid]}' and '{group_name}'."
                )
            group_map[vid] = group_name

    if len(group_order) != 2:
        raise ValueError(
            f"EPM bar charts require exactly 2 groups; got {len(group_order)} in group config."
        )
    return group_map, group_order


def _parse_open_arms_text(open_arms_text):
    names = [x.strip() for x in str(open_arms_text).split(",") if x.strip()]
    if not names:
        raise ValueError("Open arms list is empty. Use names like 'Top,Bottom'.")
    return names


def calculate_open_arms_metrics(
    directory,
    group_map=None,
    group_order=None,
    open_arm_names=None,
    closed_arm_names=None,
):
    """Calculate all open arms metrics from statistics files"""
    directory = Path(directory)
    
    # Load both CSV files
    entry_file = directory / 'ROI_Entry_Statistics.csv'
    time_file = directory / 'ROI_Statistics_Detailed.csv'
    
    if not entry_file.exists() or not time_file.exists():
        missing = []
        if not entry_file.exists():
            missing.append(entry_file.name)
        if not time_file.exists():
            missing.append(time_file.name)
        print(f"[ERROR] Statistics files not found in {directory}")
        print(f"[ERROR] Missing: {', '.join(missing)}")
        return None
    
    entry_df = pd.read_csv(entry_file)
    time_df = pd.read_csv(time_file)
    
    # Merge dataframes
    df = pd.merge(entry_df, time_df, on='Video')
    
    # Add group column
    if group_map is not None:
        def _assign_from_group_yaml(video_name):
            key = normalize_video_id(video_name)
            if key not in group_map:
                raise KeyError(
                    f"Video '{video_name}' not found in group config. "
                    "Please add it to the grouping YAML."
                )
            return group_map[key]

        df['Group'] = df['Video'].apply(_assign_from_group_yaml)
    else:
        # Backward-compatible fallback
        df['Group'] = df['Video'].apply(classify_group)
        if group_order is None:
            group_order = ['Control', 'pp3r1']
    
    # Infer available ROI names from *_sec and *_entries columns.
    sec_roi_names = {c[:-4] for c in df.columns if c.endswith("_sec")}
    entry_roi_names = {c[:-8] for c in df.columns if c.endswith("_entries")}
    available_roi_names = sorted(sec_roi_names.intersection(entry_roi_names))
    if not available_roi_names:
        raise ValueError("Cannot infer ROI names from statistics files.")

    if open_arm_names is None or closed_arm_names is None:
        raise ValueError("Both open_arm_names and closed_arm_names must be provided.")

    missing = [n for n in open_arm_names if n not in available_roi_names]
    if missing:
        raise KeyError(
            f"Open arms contain unknown ROI names: {missing}. "
            f"Available ROI names: {available_roi_names}"
        )
    missing = [n for n in closed_arm_names if n not in available_roi_names]
    if missing:
        raise KeyError(
            f"Closed arms contain unknown ROI names: {missing}. "
            f"Available ROI names: {available_roi_names}"
        )
    overlap = sorted(set(open_arm_names).intersection(set(closed_arm_names)))
    if overlap:
        raise ValueError(f"Open/closed arms overlap is not allowed: {overlap}")

    # Calculate open/closed arms metrics based on configured ROI names.
    df['OpenArms_Time_sec'] = sum(df[f"{name}_sec"] for name in open_arm_names)
    df['OpenArms_Entries'] = sum(df[f"{name}_entries"] for name in open_arm_names)
    df['OpenArms_Time_pct'] = (df['OpenArms_Time_sec'] / df['Total_ROI_sec']) * 100
    df['OpenArms_Entries_pct'] = (df['OpenArms_Entries'] / df['Total_entries']) * 100

    if closed_arm_names:
        df['ClosedArms_Time_sec'] = sum(df[f"{name}_sec"] for name in closed_arm_names)
        df['ClosedArms_Entries'] = sum(df[f"{name}_entries"] for name in closed_arm_names)
    else:
        df['ClosedArms_Time_sec'] = 0.0
        df['ClosedArms_Entries'] = 0.0
    df['ClosedArms_Time_pct'] = (df['ClosedArms_Time_sec'] / df['Total_ROI_sec']) * 100
    df['ClosedArms_Entries_pct'] = (df['ClosedArms_Entries'] / df['Total_entries']) * 100
    
    return df, group_order, open_arm_names, closed_arm_names


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


def create_bar_chart(
    ax,
    control_mean,
    control_sem,
    pp3r1_mean,
    pp3r1_sem,
    control_data,
    pp3r1_data,
    ylabel,
    title,
    p_value,
    sig_level,
    y_max=None,
):
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

    # Overlay individual data points for each bar
    # Use deterministic jitter for reproducible plotting
    rng = np.random.default_rng(42)
    point_kwargs = dict(s=20, c="black", alpha=0.8, linewidths=0, zorder=4)
    # Make jitter width one-third of previous total width
    # Previous clip range was +/- 0.38*bar_width (total 0.76*bar_width).
    # New total width = (0.76/3)*bar_width, i.e. +/- 0.126666...*bar_width.
    jitter_scale = bar_width * 0.073

    def _plot_points(center_x, values):
        vals = np.asarray(values, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return
        jitter = rng.normal(loc=0.0, scale=jitter_scale, size=vals.size)
        jitter = np.clip(jitter, -bar_width * 0.1267, bar_width * 0.1267)
        xs = center_x + jitter
        ax.scatter(xs, vals, **point_kwargs)

    _plot_points(x_pos[0], control_data)
    _plot_points(x_pos[1], pp3r1_data)
    
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


def generate_epm_bar_charts(
    directory,
    output_path=None,
    group_map=None,
    group_order=None,
    open_arm_names=None,
    closed_arm_names=None,
):
    """Generate the complete 2x3 grid of bar charts"""
    
    print("=" * 80)
    print("Generating EPM Bar Charts")
    print("=" * 80)
    
    # Calculate metrics
    result = calculate_open_arms_metrics(
        directory,
        group_map=group_map,
        group_order=group_order,
        open_arm_names=open_arm_names,
        closed_arm_names=closed_arm_names,
    )
    if result is None:
        return
    df, group_order, open_arm_names, closed_arm_names = result
    
    # Separate groups
    group_a, group_b = group_order[0], group_order[1]
    control_df = df[df['Group'] == group_a].copy()
    pp3r1_df = df[df['Group'] == group_b].copy()
    
    print(f"{group_a} group: {len(control_df)} animals")
    print(f"{group_b} group: {len(pp3r1_df)} animals")
    print(f"Open arms ROI(s): {', '.join(open_arm_names)}")
    print(f"Closed arms ROI(s): {', '.join(closed_arm_names)}")
    
    # Calculate means and SEMs for all metrics
    metrics = {
        'Time in Open Arms (s)': {
            'control_col': 'OpenArms_Time_sec',
            'pp3r1_col': 'OpenArms_Time_sec',
            'ylabel': 'Time in Open Arms (s)',
            'y_max': 200
        },
        'Time in Closed Arms (s)': {
            'control_col': 'ClosedArms_Time_sec',
            'pp3r1_col': 'ClosedArms_Time_sec',
            'ylabel': 'Time in Closed Arms (s)',
            'y_max': 200
        },
        'Entries into Open Arms': {
            'control_col': 'OpenArms_Entries',
            'pp3r1_col': 'OpenArms_Entries',
            'ylabel': 'Entries into Open Arms',
            'y_max': 30
        },
        'Entries into Closed Arms': {
            'control_col': 'ClosedArms_Entries',
            'pp3r1_col': 'ClosedArms_Entries',
            'ylabel': 'Entries into Closed Arms',
            'y_max': 30
        },
        '% Time in Open Arms': {
            'control_col': 'OpenArms_Time_pct',
            'pp3r1_col': 'OpenArms_Time_pct',
            'ylabel': '% Time in Open Arms',
            'y_max': 60
        },
        '% Time in Closed Arms': {
            'control_col': 'ClosedArms_Time_pct',
            'pp3r1_col': 'ClosedArms_Time_pct',
            'ylabel': '% Time in Closed Arms',
            'y_max': 60
        },
        '% Entries into Open Arms': {
            'control_col': 'OpenArms_Entries_pct',
            'pp3r1_col': 'OpenArms_Entries_pct',
            'ylabel': '% Entries into Open Arms',
            'y_max': 60
        },
        '% Entries into Closed Arms': {
            'control_col': 'ClosedArms_Entries_pct',
            'pp3r1_col': 'ClosedArms_Entries_pct',
            'ylabel': '% Entries into Closed Arms',
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
    
    # Create figure with 2x5 subplots - no title to match reference
    # Set figure style to match reference
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), facecolor='white')
    # Remove title to match reference image style
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Metric order matching the image layout
    metric_order = [
        'Time in Open Arms (s)',
        'Time in Closed Arms (s)',
        'Entries into Open Arms',
        'Entries into Closed Arms',
        '% Time in Open Arms',
        '% Time in Closed Arms',
        '% Entries into Open Arms',
        '% Entries into Closed Arms',
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
        print(f"  {group_a}: {control_mean:.2f} ± {control_sem:.2f}")
        print(f"  {group_b}: {pp3r1_mean:.2f} ± {pp3r1_sem:.2f}")
        print(f"  p-value: {p_value:.4f} ({sig_level})")
        
        # Create bar chart (y_max=None to auto-calculate as 1.5x of highest bar)
        create_bar_chart(
            ax,
            control_mean,
            control_sem,
            pp3r1_mean,
            pp3r1_sem,
            control_data,
            pp3r1_data,
            metric_info['ylabel'],
            metric_name,
            p_value,
            sig_level,
            y_max=None,
        )  # Auto-calculate as 3/2 of highest bar value
        ax.set_xticklabels([group_a, group_b], fontsize=11, fontweight='bold', family='Arial')
    
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


def load_video_dir_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")

    project = cfg.get("project", {}) or {}
    video_dir = project.get("video_dir")
    if not video_dir:
        raise KeyError("Missing required key: project.video_dir")
    return str(Path(video_dir))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EPM bar charts')
    parser.add_argument('--directory', type=str, default=None,
                       help='Directory containing statistics files')
    parser.add_argument('--config', type=str, default=None,
                       help='Project YAML config path (auto-loads project.video_dir)')
    parser.add_argument('--group-config', type=str, default=None,
                       help='Path to grouping YAML with a top-level "groups" mapping')
    parser.add_argument('--open-arms', type=str, default=None,
                       help='Comma-separated ROI names treated as open arms (e.g., "Top,Bottom")')
    parser.add_argument('--closed-arms', type=str, default=None,
                       help='Comma-separated ROI names treated as closed arms (e.g., "Left,Right")')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    
    args = parser.parse_args()
    directory = args.directory

    if args.config:
        try:
            directory = load_video_dir_from_config(args.config)
            print(f"[INFO] Loaded directory from config: {directory}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            sys.exit(1)

    if not directory:
        print("[ERROR] You must provide --directory or --config")
        sys.exit(1)

    group_map = None
    group_order = None
    if args.group_config:
        try:
            group_map, group_order = load_group_map_from_yaml(args.group_config)
            print(f"[INFO] Loaded group config: {args.group_config}")
            print(f"[INFO] Groups: {group_order[0]} vs {group_order[1]}")
        except Exception as e:
            print(f"[ERROR] Failed to load group config: {e}")
            sys.exit(1)

    if not args.open_arms or not args.closed_arms:
        print("[ERROR] You must provide both --open-arms and --closed-arms")
        sys.exit(1)
    try:
        open_arm_names = _parse_open_arms_text(args.open_arms)
    except Exception as e:
        print(f"[ERROR] Invalid --open-arms: {e}")
        sys.exit(1)
    try:
        closed_arm_names = _parse_open_arms_text(args.closed_arms)
    except Exception as e:
        print(f"[ERROR] Invalid --closed-arms: {e}")
        sys.exit(1)

    generate_epm_bar_charts(
        directory=directory,
        output_path=args.output,
        group_map=group_map,
        group_order=group_order,
        open_arm_names=open_arm_names,
        closed_arm_names=closed_arm_names,
    )

