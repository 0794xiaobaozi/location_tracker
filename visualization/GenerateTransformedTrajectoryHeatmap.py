"""
Generate Trajectory Heatmap with Regional Transformation
==========================================================

This script reads trajectory data from CSV files and generates a heatmap
on the standardized (ideal) Plus Maze using regional coordinate transformation.

Each trajectory point is transformed using the appropriate regional transformation
based on which region (center, left_arm, right_arm, top_arm, bottom_arm) it belongs to.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter
from RegionalTransformVisualizer import (
    create_ideal_plus_maze,
    define_regions,
    compute_regional_transform,
    determine_region,
    transform_point_regional
)


def create_blue_green_yellow_white_colormap():
    """
    Create custom colormap: Dark Blue -> Light Blue -> Green -> Yellow -> White
    """
    colors = [
        (0.0, (0.0, 0.0, 0.5)),      # Dark blue
        (0.25, (0.0, 0.5, 0.8)),     # Light blue
        (0.5, (0.0, 0.8, 0.4)),      # Green
        (0.75, (0.9, 0.9, 0.0)),     # Yellow
        (1.0, (1.0, 1.0, 0.9))       # White/Light yellow
    ]
    cmap = LinearSegmentedColormap.from_list('blue_green_yellow_white', colors, N=256)
    return cmap


def create_heatmap_from_trajectory(csv_path, original_vertices, crop_params,
                                   center_size=119, arm_length_ratio=5,
                                   canvas_size=1409, num_bins=80, sigma=1.2, fps=30,
                                   colormap='viridis', skip_seconds=15,
                                   output_path=None, show_plot=False):
    """
    Generate heatmap from trajectory CSV using regional transformation.
    
    Heatmap generation follows standard protocol:
    1. Spatial Binning: Each pixel is a recording unit
    2. Dwell Time Accumulation: Each frame adds 1/fps seconds to the grid
    3. Gaussian Smoothing: Apply 2D Gaussian kernel for smooth density
    4. Color Mapping: Blue (cold/short) to Red (hot/long) gradient
    
    Parameters:
    -----------
    csv_path : str
        Path to LocationOutput.csv file
    original_vertices : list
        12 original vertices of the Plus Maze
    crop_params : dict
        Crop parameters
    center_size : int
        Center square size in pixels
    arm_length_ratio : float
        Arm length ratio
    canvas_size : int
        Size of output canvas
    num_bins : int
        Number of bins for 2D histogram (default: 80, meaning 80×80 grid)
    sigma : float
        Gaussian blur sigma for heatmap smoothing
    fps : float
        Video frame rate (frames per second)
    colormap : str
        Matplotlib colormap name ('viridis', 'plasma', etc.)
    skip_seconds : float
        Number of seconds to skip from the beginning (default: 15)
    output_path : str
        Output image path
    show_plot : bool
        Whether to display plot
    """
    
    # Load trajectory data
    print(f"Loading trajectory data from: {csv_path}")
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
    
    print(f"  Total points: {len(df)}")
    print(f"  Valid points: {len(x_coords)}")
    print(f"  Frame rate: {fps} fps")
    
    # Skip first N seconds
    skip_frames = int(skip_seconds * fps)
    if len(frames) > skip_frames:
        x_coords = x_coords[skip_frames:]
        y_coords = y_coords[skip_frames:]
        frames = frames[skip_frames:]
        print(f"  Skipped first {skip_seconds}s ({skip_frames} frames)")
        print(f"  Remaining points: {len(x_coords)}")
    else:
        print(f"  Warning: Total frames ({len(frames)}) <= skip frames ({skip_frames}), using all data")
    
    if len(x_coords) == 0:
        raise ValueError("No valid trajectory points found!")
    
    # Convert vertices to numpy array
    src_vertices = np.array(original_vertices, dtype=np.float32)
    
    # Create ideal Plus Maze
    ideal_arm_length = int(center_size * arm_length_ratio)
    dst_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
    
    # Define regions
    regions = define_regions(src_vertices)
    
    # Transform all trajectory points
    print("\nTransforming trajectory points...")
    transformed_points = []
    transformed_regions = []
    region_counts = {'center': 0, 'left_arm': 0, 'right_arm': 0, 
                    'top_arm': 0, 'bottom_arm': 0}
    
    for x, y in zip(x_coords, y_coords):
        tx, ty, region = transform_point_regional(x, y, src_vertices, dst_vertices, regions)
        transformed_points.append([tx, ty])
        transformed_regions.append(region)
        region_counts[region] += 1
    
    transformed_points = np.array(transformed_points)
    
    print("  Region distribution:")
    for region, count in region_counts.items():
        print(f"    {region}: {count} points ({count/len(transformed_points)*100:.1f}%)")
    
    # ============ Generate Standard EPM Heatmap ============
    print("\nGenerating Standard EPM Heatmap...")
    
    # Get transformed coordinates
    tx_coords = transformed_points[:, 0]
    ty_coords = transformed_points[:, 1]
    
    # Create ideal maze vertices for mask
    dst_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
    
    # 3. Create mask for maze interior
    print(f"  Creating maze mask...")
    ideal_vertices = dst_vertices.astype(np.float32)
    maze_path = Path(ideal_vertices)
    y_grid, x_grid = np.mgrid[0:canvas_size, 0:canvas_size]
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask_inside = maze_path.contains_points(points).reshape(canvas_size, canvas_size)
    print(f"  Maze interior pixels: {np.sum(mask_inside)}")
    
    # 4. Generate heatmap data (2D histogram with num_bins x num_bins grid)
    print(f"  Generating 2D histogram ({num_bins}x{num_bins} bins)...")
    heatmap, xedges, yedges = np.histogram2d(
        tx_coords, ty_coords, 
        bins=num_bins, 
        range=[[0, canvas_size], [0, canvas_size]]
    )
    print(f"  Max count per bin: {heatmap.max():.0f}")
    
    # 5. Gaussian smoothing
    print(f"  Applying Gaussian smoothing (sigma={sigma})...")
    heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
    
    # 6. Rescale num_bins x num_bins heatmap to canvas_size x canvas_size
    print(f"  Rescaling to {canvas_size}x{canvas_size}...")
    heatmap_rescaled = cv2.resize(
        heatmap_smooth, 
        (canvas_size, canvas_size), 
        interpolation=cv2.INTER_LINEAR
    ).T
    
    # 7. Apply mask and base color logic
    heatmap_final = np.zeros((canvas_size, canvas_size))
    # Fill maze interior with smoothed heatmap values
    heatmap_final[mask_inside] = heatmap_rescaled[mask_inside]
    # Mask outside regions (will show as black background)
    heatmap_masked_outside = np.ma.masked_where(~mask_inside, heatmap_final)
    
    # 8. Plotting and styling (Standard EPM style)
    print(f"\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')  # Black background
    
    # Use colormap, vmin=0 ensures unvisited areas show base color (dark purple for viridis)
    im = ax.imshow(
        heatmap_masked_outside, 
        cmap=colormap, 
        extent=[0, canvas_size, canvas_size, 0], 
        interpolation='bilinear', 
        alpha=0.9, 
        vmin=0
    )
    
    # Overlay maze white outline
    poly = Polygon(ideal_vertices, fill=False, edgecolor='white', 
                   linewidth=2, alpha=0.8)
    ax.add_patch(poly)
    
    ax.set_title(
        f'EPM Standardized Heatmap ({num_bins}x{num_bins} Grid)\n{colormap.capitalize()} - {arm_length_ratio}:1 Ratio', 
        fontsize=15, color='white', fontweight='bold'
    )
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Stay Duration (Density)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    
    if output_path:
        # Save PNG format
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        print(f"\nSaved heatmap to: {output_path}")
        
        # Save PDF format (vector format, supports transparency, better for publications)
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='black', edgecolor='none', transparent=False)
        print(f"Saved PDF format to: {pdf_path}")
        
        # Also save EPS format (vector format, but no transparency support)
        eps_path = output_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        print(f"Saved EPS format to: {eps_path} (no transparency support)")
    
    # Save transformed trajectory to CSV
    if output_path:
        csv_output_path = output_path.replace('_EPM_Heatmap_Standard.png', '_TransformedTrajectory.csv')
        
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
        
        transformed_df.to_csv(csv_output_path, index=False)
        print(f"Saved transformed trajectory to: {csv_output_path}")
        print(f"  Total points: {len(transformed_df)}")
        print(f"  Columns: {list(transformed_df.columns)}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, transformed_points


# ============ Main execution ============
if __name__ == "__main__":
    # Configuration
    video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    csv_file = "3pl2_LocationOutput.csv"
    csv_path = os.path.join(video_dir, csv_file)
    
    # Original polygon vertices (from notebook, left arm adjusted)
    original_vertices = [
        (61, 244), (68, 341), (405, 331), (424, 569), 
        (536, 569), (525, 323), (822, 303), (821, 195), 
        (514, 206), (474, 1), (372, 0), (396, 211)
    ]
    
    # Crop parameters
    crop_params = {
        'x0': 128,
        'x1': 954,
        'y0': 0,
        'y1': 604
    }
    
    # Plus Maze dimensions
    center_size = 119  # pixels = 90mm
    arm_length_ratio = 5  # arm = 5 * center = 450mm
    
    # Output path (Standard EPM heatmap)
    output_file = csv_file.replace('_LocationOutput.csv', '_EPM_Heatmap_Standard.png')
    output_path = os.path.join(video_dir, output_file)
    
    print("=" * 70)
    print("Generate Transformed Trajectory Heatmap")
    print("=" * 70)
    print(f"Input CSV: {csv_path}")
    print(f"Center size: {center_size} px = 90mm")
    print(f"Arm length ratio: {arm_length_ratio}x (arm = {center_size * arm_length_ratio} px = 450mm)")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Get FPS from video (or use default)
    video_path = os.path.join(video_dir, csv_file.replace('_LocationOutput.csv', '.mp4'))
    fps = 30.0  # Default
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        print(f"Detected video FPS: {fps:.2f}")
    
    # Generate heatmap
    fig, transformed_points = create_heatmap_from_trajectory(
        csv_path=csv_path,
        original_vertices=original_vertices,
        crop_params=crop_params,
        center_size=center_size,
        arm_length_ratio=arm_length_ratio,
        canvas_size=1409,
        num_bins=80,      # 80x80 grid for 2D histogram
        sigma=1.2,        # Gaussian smoothing
        fps=fps,          # Frame rate (for CSV metadata)
        colormap='viridis',  # Viridis colormap
        skip_seconds=15,  # Skip first 15 seconds
        output_path=output_path,
        show_plot=False
    )
    
    print("\nDone!")
    print(f"\nTransformed {len(transformed_points)} trajectory points")
    print(f"Heatmap saved to: {output_path}")
