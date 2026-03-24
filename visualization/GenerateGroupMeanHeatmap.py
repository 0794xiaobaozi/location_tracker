"""
Generate Group Mean Heatmap for EPM
====================================
This script generates group-averaged heatmaps by:
1. Loading all transformed trajectory CSVs for each group
2. Accumulating 2D histograms from all individuals
3. Taking the mean across individuals
4. Applying Gaussian smoothing
5. Rendering with mask and colormap
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter
from RegionalTransformVisualizer import create_ideal_plus_maze


def generate_group_mean_heatmap(csv_files, group_name, output_path,
                                canvas_size=1409, center_size=119, 
                                arm_length_ratio=5, num_bins=80, 
                                sigma=1.2, colormap='viridis', 
                                skip_seconds=15, fps=25, show_plot=False):
    """
    Generate group-averaged heatmap from multiple CSV files.
    
    Parameters:
    -----------
    csv_files : list
        List of paths to TransformedTrajectory.csv files
    group_name : str
        Name of the group (for title)
    output_path : str
        Output image path
    canvas_size : int
        Size of canvas
    center_size : int
        Center square size
    arm_length_ratio : float
        Arm length ratio
    num_bins : int
        Number of bins for 2D histogram
    sigma : float
        Gaussian smoothing sigma
    colormap : str
        Matplotlib colormap name
    show_plot : bool
        Whether to display plot
    """
    
    print(f"\n{'='*70}")
    print(f"Generating Group Mean Heatmap: {group_name}")
    print(f"{'='*70}")
    print(f"Number of individuals: {len(csv_files)}")
    
    # Step 1: Create unified grid template (accumulation pool)
    group_accumulation = np.zeros((num_bins, num_bins), dtype=np.float64)
    valid_files = []
    
    # Step 2: Loop through each CSV file
    print(f"\nProcessing individual files...")
    for i, csv_file in enumerate(csv_files, 1):
        if not os.path.exists(csv_file):
            print(f"  [{i}/{len(csv_files)}] Skipping (not found): {csv_file}")
            continue
        
        try:
            # Read transformed coordinates
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'X_Transformed' not in df.columns or 'Y_Transformed' not in df.columns:
                print(f"  [{i}/{len(csv_files)}] Skipping (missing columns): {csv_file}")
                continue
            
            x_coords = df['X_Transformed'].values
            y_coords = df['Y_Transformed'].values
            frames = df['Frame'].values if 'Frame' in df.columns else np.arange(len(df))
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            frames = frames[valid_mask]
            
            if len(x_coords) == 0:
                print(f"  [{i}/{len(csv_files)}] Skipping (no valid points): {csv_file}")
                continue
            
            # Skip first N seconds
            skip_frames = int(skip_seconds * fps)
            if len(frames) > skip_frames:
                x_coords = x_coords[skip_frames:]
                y_coords = y_coords[skip_frames:]
                frames = frames[skip_frames:]
                print(f"  [{i}/{len(csv_files)}] Skipped first {skip_seconds}s ({skip_frames} frames), "
                      f"remaining: {len(x_coords)} points")
            else:
                print(f"  [{i}/{len(csv_files)}] Warning: Total frames ({len(frames)}) <= skip frames ({skip_frames})")
            
            # Generate 2D histogram for this individual (UNSMOOTHED)
            h, xedges, yedges = np.histogram2d(
                x_coords, y_coords,
                bins=num_bins,
                range=[[0, canvas_size], [0, canvas_size]]
            )
            
            # Accumulate (add to group total)
            group_accumulation += h
            valid_files.append(csv_file)
            
            print(f"  [{i}/{len(csv_files)}] Processed: {os.path.basename(csv_file)} "
                  f"({len(x_coords)} points, max count: {h.max():.0f})")
        
        except Exception as e:
            print(f"  [{i}/{len(csv_files)}] Error processing {csv_file}: {e}")
            continue
    
    if len(valid_files) == 0:
        raise ValueError(f"No valid CSV files found for group {group_name}")
    
    # Step 3: Take mean across individuals
    print(f"\nCalculating group mean...")
    num_valid = len(valid_files)
    group_mean = group_accumulation / num_valid
    
    print(f"  Total accumulation max: {group_accumulation.max():.0f}")
    print(f"  Group mean max: {group_mean.max():.2f}")
    print(f"  Valid individuals: {num_valid}")
    
    # Step 4: Apply Gaussian smoothing to the group mean matrix
    print(f"  Applying Gaussian smoothing (sigma={sigma})...")
    heatmap_smooth = gaussian_filter(group_mean, sigma=sigma)
    
    # Step 5: Rescale to canvas size
    print(f"  Rescaling to {canvas_size}x{canvas_size}...")
    heatmap_rescaled = cv2.resize(
        heatmap_smooth,
        (canvas_size, canvas_size),
        interpolation=cv2.INTER_LINEAR
    ).T
    
    # Create ideal maze vertices for mask
    ideal_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
    
    # Create mask for maze interior
    print(f"  Creating maze mask...")
    maze_path = Path(ideal_vertices.astype(np.float32))
    y_grid, x_grid = np.mgrid[0:canvas_size, 0:canvas_size]
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask_inside = maze_path.contains_points(points).reshape(canvas_size, canvas_size)
    print(f"  Maze interior pixels: {np.sum(mask_inside)}")
    
    # Apply mask and base color logic
    heatmap_final = np.zeros((canvas_size, canvas_size))
    heatmap_final[mask_inside] = heatmap_rescaled[mask_inside]
    heatmap_masked_outside = np.ma.masked_where(~mask_inside, heatmap_final)
    
    # Step 6: Plotting and styling
    print(f"\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('none')  # Transparent background
    
    # Use colormap, vmin=0 ensures unvisited areas show base color
    im = ax.imshow(
        heatmap_masked_outside,
        cmap=colormap,
        extent=[0, canvas_size, canvas_size, 0],
        interpolation='bilinear',
        alpha=0.9,
        vmin=0
    )
    
    # Overlay maze black outline (changed from white)
    poly = Polygon(ideal_vertices, fill=False, edgecolor='black',
                   linewidth=2, alpha=1.0)
    ax.add_patch(poly)
    
    ax.set_title(
        f'Group Mean Heatmap: {group_name}\n({num_bins}x{num_bins} Grid, n={num_valid})\n{colormap.capitalize()} - {arm_length_ratio}:1 Ratio',
        fontsize=15, color='black', fontweight='bold'  # Changed to black
    )
    ax.axis('off')
    
    # Add colorbar with black text
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Stay Duration (Density)', color='black', fontsize=12)  # Changed to black
    cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')  # Changed to black
    
    fig.patch.set_facecolor('none')  # Transparent background
    plt.tight_layout()
    
    if output_path:
        # Save PNG format (with transparent background)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='none', edgecolor='none', transparent=True)
        print(f"\nSaved group mean heatmap to: {output_path}")
        
        # Save SVG format (vector format, supports transparency, best for publications)
        svg_path = output_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='none', edgecolor='none', transparent=True)
        print(f"Saved SVG format to: {svg_path} (transparent background)")
        
        # Save PDF format (vector format, supports transparency)
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='none', edgecolor='none', transparent=True)
        print(f"Saved PDF format to: {pdf_path} (transparent background)")
        
        # Also save EPS format (vector format, but no transparency support)
        # Note: EPS doesn't support transparency, so we use white background for compatibility
        eps_path = output_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved EPS format to: {eps_path} (white background, no transparency support)")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, group_mean, num_valid


# ============ Main execution ============
if __name__ == "__main__":
    # Configuration
    video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    
    # Define groups based on file naming pattern
    # pp3r1-KO group: files starting with "3p" (3pl1, 3pl2, 3pl3, 3pl4, 3pl5, 3pr1, 3pr2, 3pr3, 3pr4, 3prh)
    # CTR group: other files (cn, cq, cl, cr series)
    
    # Find all TransformedTrajectory.csv files
    all_csv_files = glob.glob(os.path.join(video_dir, "*_TransformedTrajectory.csv"))
    
    # Separate into groups
    pp3r1_ko_files = [f for f in all_csv_files if os.path.basename(f).startswith('3p')]
    ctr_files = [f for f in all_csv_files if not os.path.basename(f).startswith('3p')]
    
    print("=" * 70)
    print("Group Mean Heatmap Generator")
    print("=" * 70)
    print(f"\nFound CSV files:")
    print(f"  pp3r1-KO group: {len(pp3r1_ko_files)} files")
    print(f"  CTR group: {len(ctr_files)} files")
    
    # Get FPS from first video (assuming all videos have same FPS)
    first_video = glob.glob(os.path.join(video_dir, "*.mp4"))[0] if glob.glob(os.path.join(video_dir, "*.mp4")) else None
    fps = 25.0  # Default
    if first_video:
        import cv2
        cap = cv2.VideoCapture(first_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
    print(f"Video FPS: {fps:.2f}")
    print(f"Skipping first 15 seconds ({int(15 * fps)} frames)")
    
    # Generate heatmap for pp3r1-KO group
    if len(pp3r1_ko_files) > 0:
        pp3r1_output = os.path.join(video_dir, "Group_pp3r1_KO_MeanHeatmap.png")
        fig1, mean1, n1 = generate_group_mean_heatmap(
            csv_files=pp3r1_ko_files,
            group_name="pp3r1-KO",
            output_path=pp3r1_output,
            canvas_size=1409,
            center_size=119,
            arm_length_ratio=5,
            num_bins=80,
            sigma=1.2,
            colormap='viridis',
            skip_seconds=15,
            fps=fps,
            show_plot=False
        )
        print(f"\n[OK] pp3r1-KO group: n={n1} individuals")
    
    # Generate heatmap for CTR group
    if len(ctr_files) > 0:
        ctr_output = os.path.join(video_dir, "Group_CTR_MeanHeatmap.png")
        fig2, mean2, n2 = generate_group_mean_heatmap(
            csv_files=ctr_files,
            group_name="CTR (Control)",
            output_path=ctr_output,
            canvas_size=1409,
            center_size=119,
            arm_length_ratio=5,
            num_bins=80,
            sigma=1.2,
            colormap='viridis',
            skip_seconds=15,
            fps=fps,
            show_plot=False
        )
        print(f"\n[OK] CTR group: n={n2} individuals")
    
    print("\n" + "=" * 70)
    print("Done! All group mean heatmaps generated.")
    print("=" * 70)
