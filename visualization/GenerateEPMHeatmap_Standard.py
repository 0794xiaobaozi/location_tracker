"""
Generate Standardized EPM Heatmap
==================================
Professional-grade heatmap generation following the algorithm logic provided.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path
import cv2
import os

def generate_epm_heatmap(csv_path, output_path=None, canvas_size=1409, 
                         center_side=119, arm_ratio=5, num_bins=80, 
                         sigma=1.2, colormap='viridis', show_plot=False):
    """
    Generate standardized EPM heatmap with professional styling.
    
    Parameters:
    -----------
    csv_path : str
        Path to TransformedTrajectory.csv file
    output_path : str
        Output image path
    canvas_size : int
        Size of canvas (square)
    center_side : int
        Center square side length in pixels
    arm_ratio : float
        Arm length ratio (5:1 for standard EPM)
    num_bins : int
        Number of bins for 2D histogram (80x80 grid)
    sigma : float
        Gaussian smoothing sigma
    colormap : str
        Matplotlib colormap name ('viridis', 'plasma', etc.)
    show_plot : bool
        Whether to display plot
    """
    
    # 1. Load transformed trajectory data
    print(f"Loading trajectory data from: {csv_path}")
    df = pd.read_csv(csv_path)
    x_coords = df['X_Transformed'].values
    y_coords = df['Y_Transformed'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    print(f"  Total points: {len(df)}")
    print(f"  Valid points: {len(x_coords)}")
    
    # 2. Define maze geometry parameters (5:1 ratio)
    arm_length = center_side * arm_ratio
    mid = canvas_size / 2
    hw = center_side / 2
    al = arm_length
    
    # Define ideal 12 vertices (consistent with previous script)
    ideal_vertices = np.array([
        [mid - al - hw, mid - hw],      # 0: 左臂外侧上
        [mid - al - hw, mid + hw],      # 1: 左臂外侧下
        [mid - hw, mid + hw],           # 2: 中心左下
        [mid - hw, mid + al + hw],      # 3: 下臂左侧
        [mid + hw, mid + al + hw],      # 4: 下臂右侧
        [mid + hw, mid + hw],           # 5: 中心右下
        [mid + al + hw, mid + hw],      # 6: 右臂外侧下
        [mid + al + hw, mid - hw],      # 7: 右臂外侧上
        [mid + hw, mid - hw],           # 8: 中心右上
        [mid + hw, mid - al - hw],      # 9: 上臂右侧
        [mid - hw, mid - al - hw],      # 10: 上臂左侧
        [mid - hw, mid - hw]           # 11: 中心左上
    ])
    
    # 3. Create mask for maze interior
    print(f"\nCreating maze mask...")
    maze_path = Path(ideal_vertices)
    y_grid, x_grid = np.mgrid[0:canvas_size, 0:canvas_size]
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask_inside = maze_path.contains_points(points).reshape(canvas_size, canvas_size)
    print(f"  Maze interior pixels: {np.sum(mask_inside)}")
    
    # 4. Generate heatmap data (2D histogram with num_bins x num_bins grid)
    print(f"\nGenerating 2D histogram ({num_bins}x{num_bins} bins)...")
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords, 
        bins=num_bins, 
        range=[[0, canvas_size], [0, canvas_size]]
    )
    print(f"  Max count per bin: {heatmap.max():.0f}")
    
    # 5. Gaussian smoothing
    print(f"  Applying Gaussian smoothing (sigma={sigma})...")
    heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
    
    # 6. Rescale 80x80 heatmap to 1409x1409 canvas size
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
    
    # 8. Plotting and styling
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
    from matplotlib.patches import Polygon
    poly = Polygon(ideal_vertices, fill=False, edgecolor='white', 
                   linewidth=2, alpha=0.8)
    ax.add_patch(poly)
    
    ax.set_title(
        f'EPM Standardized Heatmap ({num_bins}x{num_bins} Grid)\n{colormap.capitalize()} - {arm_ratio}:1 Ratio', 
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
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, heatmap_final


# ============ Main execution ============
if __name__ == "__main__":
    # Configuration
    video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    csv_file = "3pl1_TransformedTrajectory.csv"
    csv_path = os.path.join(video_dir, csv_file)
    
    # Output path
    output_file = csv_file.replace('_TransformedTrajectory.csv', '_EPM_Heatmap_Standard.png')
    output_path = os.path.join(video_dir, output_file)
    
    print("=" * 70)
    print("Generate Standardized EPM Heatmap")
    print("=" * 70)
    print(f"Input CSV: {csv_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Generate heatmap
    fig, heatmap = generate_epm_heatmap(
        csv_path=csv_path,
        output_path=output_path,
        canvas_size=1409,
        center_side=119,
        arm_ratio=5,
        num_bins=80,      # 80x80 grid
        sigma=1.2,        # Gaussian smoothing
        colormap='viridis',  # Viridis colormap
        show_plot=False
    )
    
    print("\nDone!")
