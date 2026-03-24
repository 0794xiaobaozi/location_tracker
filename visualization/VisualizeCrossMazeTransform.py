"""
Cross Maze Perspective Transformation Visualizer
=================================================

This script visualizes the perspective transformation of a cross/plus maze.
It shows:
1. Original distorted cross maze from the video with colored markers
2. Rectified/ideal cross maze with corresponding colored markers

The 12 vertices of the cross maze are mapped to create a proper cross shape.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os


def create_ideal_cross(center_size=100, arm_length_ratio=5, center=(300, 300)):
    """
    Create ideal Plus Maze vertices with proper proportions.
    
    The Plus Maze has 12 vertices arranged as follows:
    
              10 ---- 9
                |    |
                |    |  (arm_length = center_size * arm_length_ratio)
                |    |
     0 ----11   |    |   8 ---- 7
     |      |   |    |   |      |
     |      |   |    |   |      |
     1 ---- 2   |    |   5 ---- 6
                |    |
                |    |
              3 ---- 4
    
    Parameters:
    -----------
    center_size : int
        Size of the center square (width = height = center_size)
        This corresponds to the arm width as well
    arm_length_ratio : float
        Ratio of arm length to center size (arm_length = center_size * arm_length_ratio)
    center : tuple
        Center position of the cross (x, y)
    
    Returns:
    --------
    list of tuples : 12 vertices in the same order as the original polygon
    """
    cx, cy = center
    hw = center_size // 2  # half width (half of center square side)
    arm_length = int(center_size * arm_length_ratio)  # arm length from center edge
    
    # Define vertices in the same order as the original polygon
    # Original order: 0,1,2,3,4,5,6,7,8,9,10,11
    vertices = [
        (cx - arm_length - hw, cy - hw),      # 0: Left arm, outer-top
        (cx - arm_length - hw, cy + hw),      # 1: Left arm, outer-bottom
        (cx - hw, cy + hw),                   # 2: Center-left, bottom (junction)
        (cx - hw, cy + arm_length + hw),      # 3: Bottom arm, left
        (cx + hw, cy + arm_length + hw),      # 4: Bottom arm, right
        (cx + hw, cy + hw),                   # 5: Center-right, bottom (junction)
        (cx + arm_length + hw, cy + hw),      # 6: Right arm, outer-bottom
        (cx + arm_length + hw, cy - hw),      # 7: Right arm, outer-top
        (cx + hw, cy - hw),                   # 8: Center-right, top (junction)
        (cx + hw, cy - arm_length - hw),      # 9: Top arm, right
        (cx - hw, cy - arm_length - hw),      # 10: Top arm, left
        (cx - hw, cy - hw),                   # 11: Center-left, top (junction)
    ]
    
    return vertices


def visualize_cross_maze_transform(video_path, original_vertices, output_path=None,
                                   arm_width=100, arm_length=150, show_plot=True):
    """
    Visualize the perspective transformation of a cross maze.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    original_vertices : list
        List of 12 (x, y) tuples defining the original distorted cross
    output_path : str, optional
        Path to save the output image
    arm_width : int
        Width of arms in the ideal cross
    arm_length : int
        Length of arms in the ideal cross
    show_plot : bool
        Whether to display the plot
    
    Returns:
    --------
    tuple : (original_frame_with_markers, transformed_frame)
    """
    
    # Load first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create ideal cross vertices
    h, w = frame.shape[:2]
    ideal_center = (w // 2, h // 2)
    ideal_vertices = create_ideal_cross(arm_width, arm_length, ideal_center)
    
    # Define colors for each vertex (using a colormap)
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ============ Left subplot: Original distorted cross ============
    ax1 = axes[0]
    ax1.imshow(frame_rgb)
    
    # Draw polygon
    poly_pts = np.array(original_vertices)
    polygon = MplPolygon(poly_pts, fill=True, facecolor='cyan', 
                         edgecolor='yellow', alpha=0.3, linewidth=2)
    ax1.add_patch(polygon)
    
    # Draw vertices with colors and numbers
    for i, (x, y) in enumerate(original_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=200, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    ax1.set_title('Original Distorted Cross Maze\n(Camera Perspective)', fontsize=14)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.axis('on')
    
    # ============ Right subplot: Ideal/rectified cross ============
    ax2 = axes[1]
    
    # Create a blank image for the ideal cross
    ideal_img = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw the ideal cross polygon
    ideal_pts = np.array(ideal_vertices, dtype=np.int32)
    cv2.fillPoly(ideal_img, [ideal_pts], (200, 255, 255))  # Cyan fill
    cv2.polylines(ideal_img, [ideal_pts], True, (0, 200, 200), 3)  # Yellow outline
    
    ax2.imshow(ideal_img)
    
    # Draw vertices with colors and numbers
    for i, (x, y) in enumerate(ideal_vertices):
        ax2.scatter(x, y, c=[colors[i]], s=200, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    ax2.set_title('Ideal Cross Maze\n(Rectified/Undistorted)', fontsize=14)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.axis('on')
    
    # Add legend
    legend_text = "Vertex Colors:\n"
    arm_labels = {
        0: "Left arm (outer-top)",
        1: "Left arm (outer-bottom)", 
        2: "Left-Center (bottom)",
        3: "Bottom arm (left)",
        4: "Bottom arm (right)",
        5: "Right-Center (bottom)",
        6: "Right arm (outer-bottom)",
        7: "Right arm (outer-top)",
        8: "Right-Center (top)",
        9: "Top arm (right)",
        10: "Top arm (left)",
        11: "Left-Center (top)"
    }
    
    fig.suptitle('Cross Maze Perspective Transformation Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def visualize_with_transformed_frame(video_path, original_vertices, output_path=None,
                                     arm_width=100, arm_length=150, show_plot=True):
    """
    Visualize with actual perspective transformation of the video frame.
    This warps the actual video to show what the rectified view would look like.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    original_vertices : list
        List of 12 (x, y) tuples defining the original distorted cross
    output_path : str, optional
        Path to save the output image
    arm_width : int
        Width of arms in the ideal cross
    arm_length : int
        Length of arms in the ideal cross
    show_plot : bool
        Whether to display the plot
    """
    
    # Load first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Create ideal cross vertices
    ideal_center = (w // 2, h // 2)
    ideal_vertices = create_ideal_cross(arm_width, arm_length, ideal_center)
    
    # For perspective transformation, we need to use 4 corresponding points
    # Let's use the 4 corner points of the cross center region
    # Points 2, 5, 8, 11 form the inner square of the cross
    
    # Original center points (distorted)
    src_center = np.array([
        original_vertices[11],  # top-left of center
        original_vertices[8],   # top-right of center  
        original_vertices[5],   # bottom-right of center
        original_vertices[2],   # bottom-left of center
    ], dtype=np.float32)
    
    # Ideal center points
    dst_center = np.array([
        ideal_vertices[11],
        ideal_vertices[8],
        ideal_vertices[5],
        ideal_vertices[2],
    ], dtype=np.float32)
    
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_center, dst_center)
    
    # Warp the frame
    warped_frame = cv2.warpPerspective(frame_rgb, M, (w, h), 
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(240, 240, 240))
    
    # Transform all original vertices to see where they end up
    orig_pts = np.array(original_vertices, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(orig_pts, M).reshape(-1, 2)
    
    # Define colors
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ============ Left subplot: Original ============
    ax1 = axes[0]
    ax1.imshow(frame_rgb)
    
    # Draw polygon
    poly_pts = np.array(original_vertices)
    polygon = MplPolygon(poly_pts, fill=True, facecolor='cyan', 
                         edgecolor='yellow', alpha=0.3, linewidth=2)
    ax1.add_patch(polygon)
    
    # Draw vertices
    for i, (x, y) in enumerate(original_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=200, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    ax1.set_title('Original Frame with Distorted Cross\n(Camera Perspective)', fontsize=14)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # ============ Right subplot: Warped frame ============
    ax2 = axes[1]
    ax2.imshow(warped_frame)
    
    # Draw transformed polygon
    trans_poly = MplPolygon(transformed_pts, fill=True, facecolor='cyan', 
                            edgecolor='yellow', alpha=0.3, linewidth=2)
    ax2.add_patch(trans_poly)
    
    # Draw transformed vertices
    for i, (x, y) in enumerate(transformed_pts):
        ax2.scatter(x, y, c=[colors[i]], s=200, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    ax2.set_title('Perspective-Corrected Frame\n(Rectified Cross Maze)', fontsize=14)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    fig.suptitle('Cross Maze Perspective Transformation\n(Same colored numbers = corresponding points)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig, M, transformed_pts


def visualize_comparison_grid(video_path, original_vertices, output_path=None,
                              arm_width=100, arm_length=150, show_plot=True):
    """
    Create a comprehensive 2x2 grid visualization showing:
    1. Original frame with polygon
    2. Ideal cross schematic
    3. Warped frame (perspective corrected)
    4. Overlay comparison
    """
    
    # Load frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Create ideal cross
    ideal_center = (w // 2, h // 2)
    ideal_vertices = create_ideal_cross(arm_width, arm_length, ideal_center)
    
    # Perspective transform
    src_center = np.array([
        original_vertices[11], original_vertices[8],
        original_vertices[5], original_vertices[2],
    ], dtype=np.float32)
    
    dst_center = np.array([
        ideal_vertices[11], ideal_vertices[8],
        ideal_vertices[5], ideal_vertices[2],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_center, dst_center)
    warped_frame = cv2.warpPerspective(frame_rgb, M, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(240, 240, 240))
    
    orig_pts = np.array(original_vertices, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(orig_pts, M).reshape(-1, 2)
    
    # Colors
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ============ Top-Left: Original frame ============
    ax1 = axes[0, 0]
    ax1.imshow(frame_rgb)
    poly = MplPolygon(np.array(original_vertices), fill=True, 
                      facecolor='cyan', edgecolor='yellow', alpha=0.3, linewidth=2)
    ax1.add_patch(poly)
    for i, (x, y) in enumerate(original_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    ax1.set_title('① Original Distorted Cross\n(Due to Camera Perspective)', fontsize=12)
    ax1.axis('on')
    
    # ============ Top-Right: Ideal cross schematic ============
    ax2 = axes[0, 1]
    ideal_img = np.ones((h, w, 3), dtype=np.uint8) * 245
    ideal_pts_np = np.array(ideal_vertices, dtype=np.int32)
    cv2.fillPoly(ideal_img, [ideal_pts_np], (200, 255, 255))
    cv2.polylines(ideal_img, [ideal_pts_np], True, (0, 180, 180), 3)
    ax2.imshow(ideal_img)
    for i, (x, y) in enumerate(ideal_vertices):
        ax2.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    ax2.set_title('② Ideal Cross Maze Shape\n(Target Rectified Shape)', fontsize=12)
    ax2.axis('on')
    
    # ============ Bottom-Left: Warped frame ============
    ax3 = axes[1, 0]
    ax3.imshow(warped_frame)
    trans_poly = MplPolygon(transformed_pts, fill=True,
                            facecolor='cyan', edgecolor='yellow', alpha=0.3, linewidth=2)
    ax3.add_patch(trans_poly)
    for i, (x, y) in enumerate(transformed_pts):
        ax3.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax3.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    ax3.set_title('③ Perspective-Corrected Frame\n(Actual Video Warped)', fontsize=12)
    ax3.axis('on')
    
    # ============ Bottom-Right: Legend and info ============
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Add vertex legend
    ax4.text(0.5, 9.5, 'Vertex Legend:', fontsize=14, fontweight='bold', 
             transform=ax4.transAxes, verticalalignment='top')
    
    labels = [
        "0: Left arm (outer-top)",
        "1: Left arm (outer-bottom)",
        "2: Center-left (bottom junction)",
        "3: Bottom arm (left edge)",
        "4: Bottom arm (right edge)",
        "5: Center-right (bottom junction)",
        "6: Right arm (outer-bottom)",
        "7: Right arm (outer-top)",
        "8: Center-right (top junction)",
        "9: Top arm (right edge)",
        "10: Top arm (left edge)",
        "11: Center-left (top junction)"
    ]
    
    for i, label in enumerate(labels):
        y_pos = 8.5 - i * 0.6
        ax4.scatter([0.8], [y_pos], c=[colors[i]], s=100, edgecolors='black', 
                   linewidths=1, transform=ax4.transData)
        ax4.text(1.5, y_pos, label, fontsize=10, verticalalignment='center',
                transform=ax4.transData)
    
    # Add info text
    ax4.text(0.5, 1.5, 
             'Note: Same colored/numbered points\ncorrespond between views.\n'
             'The transformation preserves\nthe cross maze structure.',
             fontsize=11, style='italic', transform=ax4.transData,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Cross Maze Perspective Transformation Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved comprehensive visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def visualize_with_crop(video_path, original_vertices, crop_params, output_path=None,
                        center_size=100, arm_length_ratio=5, show_plot=True):
    """
    Visualize with crop applied first, then show the polygon.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    original_vertices : list
        List of 12 (x, y) tuples defining the cross in CROPPED coordinates
    crop_params : dict
        Crop parameters: {'x0': int, 'x1': int, 'y0': int, 'y1': int}
    center_size : int
        Size of the center square in pixels
    arm_length_ratio : float
        Ratio of arm length to center size
    """
    
    # Load frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    
    # Apply crop
    x0, x1 = crop_params['x0'], crop_params['x1']
    y0, y1 = crop_params['y0'], crop_params['y1']
    
    # Clamp to actual frame size
    x1 = min(x1, orig_w)
    y1 = min(y1, orig_h)
    
    cropped_frame = frame_rgb[y0:y1, x0:x1]
    h, w = cropped_frame.shape[:2]
    
    print(f"Original frame: {orig_w} x {orig_h}")
    print(f"Crop: ({x0}, {y0}) -> ({x1}, {y1})")
    print(f"Cropped frame: {w} x {h}")
    
    # Calculate original maze dimensions from vertices
    p11 = np.array(original_vertices[11])  # center-left-top
    p8 = np.array(original_vertices[8])    # center-right-top
    p5 = np.array(original_vertices[5])    # center-right-bottom
    p2 = np.array(original_vertices[2])    # center-left-bottom
    
    # Calculate original center dimensions
    original_center_top = np.linalg.norm(p8 - p11)      # top edge of center
    original_center_bottom = np.linalg.norm(p5 - p2)    # bottom edge of center
    original_center_left = np.linalg.norm(p11 - p2)     # left edge of center
    original_center_right = np.linalg.norm(p8 - p5)     # right edge of center
    
    print(f"Original center dimensions:")
    print(f"  Top edge (11-8): {original_center_top:.2f} px")
    print(f"  Bottom edge (2-5): {original_center_bottom:.2f} px")
    print(f"  Left edge (2-11): {original_center_left:.2f} px")
    print(f"  Right edge (5-8): {original_center_right:.2f} px")
    
    # Calculate arm lengths from original
    p0 = np.array(original_vertices[0])   # left arm outer-top
    p6 = np.array(original_vertices[6])   # right arm outer-bottom
    p9 = np.array(original_vertices[9])   # top arm right
    p3 = np.array(original_vertices[3])   # bottom arm left
    
    # Use the center of the original maze for reference
    original_center_x = (p11[0] + p8[0] + p5[0] + p2[0]) / 4
    original_center_y = (p11[1] + p8[1] + p5[1] + p2[1]) / 4
    
    # For ideal cross: keep the same overall size but make it symmetric
    # Use average center size as the ideal center size
    avg_center_size = (original_center_top + original_center_bottom + 
                       original_center_left + original_center_right) / 4
    
    # Calculate actual arm lengths
    left_arm = np.linalg.norm(p0 - p11)
    right_arm = np.linalg.norm(p6 - p5)
    top_arm = np.linalg.norm(p9 - p8)
    bottom_arm = np.linalg.norm(p3 - p2)
    avg_arm_length = (left_arm + right_arm + top_arm + bottom_arm) / 4
    
    actual_ratio = avg_arm_length / avg_center_size
    print(f"Original arm lengths: L={left_arm:.0f}, R={right_arm:.0f}, T={top_arm:.0f}, B={bottom_arm:.0f}")
    print(f"Average center size: {avg_center_size:.2f} px")
    print(f"Average arm length: {avg_arm_length:.2f} px")
    print(f"Actual arm/center ratio: {actual_ratio:.2f}x")
    
    # KEY INSIGHT: The perspective transform should:
    # 1. Keep the cross in the SAME bounding box as original
    # 2. Just make the shape REGULAR (symmetric, perpendicular arms)
    # 3. The canvas size stays the same, content gets "stretched/corrected"
    
    # Calculate original bounding box
    all_x = [v[0] for v in original_vertices]
    all_y = [v[1] for v in original_vertices]
    bbox_left, bbox_right = min(all_x), max(all_x)
    bbox_top, bbox_bottom = min(all_y), max(all_y)
    bbox_width = bbox_right - bbox_left
    bbox_height = bbox_bottom - bbox_top
    bbox_center_x = (bbox_left + bbox_right) / 2
    bbox_center_y = (bbox_top + bbox_bottom) / 2
    
    print(f"\nOriginal bounding box:")
    print(f"  X: {bbox_left:.0f} to {bbox_right:.0f} (width: {bbox_width:.0f})")
    print(f"  Y: {bbox_top:.0f} to {bbox_bottom:.0f} (height: {bbox_height:.0f})")
    print(f"  Center: ({bbox_center_x:.0f}, {bbox_center_y:.0f})")
    
    # Determine ratio to use
    if arm_length_ratio is None:
        arm_length_ratio = actual_ratio
        print(f"\n>> Using ACTUAL ratio from video: {arm_length_ratio:.2f}x")
    else:
        print(f"\n>> Using SPECIFIED ratio: {arm_length_ratio}x")
    
    # CORRECT APPROACH: Keep the center size similar to original,
    # and stretch/compress arms to achieve the desired ratio
    # This may require a LARGER canvas
    
    ideal_center_size = int(avg_center_size)  # Keep center size ~same as original
    ideal_arm_length = int(ideal_center_size * arm_length_ratio)
    ideal_total_size = ideal_center_size + 2 * ideal_arm_length
    
    print(f"\nIdeal Plus Maze (TRUE {arm_length_ratio}x ratio):")
    print(f"  Center size: {ideal_center_size} px (same as original)")
    print(f"  Arm length: {ideal_arm_length} px (= center × {arm_length_ratio})")
    print(f"  Total size: {ideal_total_size} px")
    
    # Check if we need to expand canvas
    need_expand = ideal_total_size > min(w, h)
    if need_expand:
        print(f"\n[!] Ideal cross ({ideal_total_size}px) > canvas ({min(w,h)}px)")
        print(f"    Will EXPAND canvas to fit the corrected image!")
        # New canvas size with margin
        new_size = ideal_total_size + 100
        expanded_w = max(w, new_size)
        expanded_h = max(h, new_size)
        print(f"    New canvas size: {expanded_w} x {expanded_h}")
    else:
        expanded_w, expanded_h = w, h
        print(f"\n[OK] Ideal cross fits in current canvas")
    
    # ========== 优化：使用12点全匹配 (Homography) ==========
    # 按照标准Plus Maze构建理想12顶点
    mid_x, mid_y = expanded_w / 2, expanded_h / 2
    hw = ideal_center_size / 2  # 半宽
    al = ideal_arm_length       # 臂长
    
    # 按照原始12个点的顺序构建理想坐标
    # 0:左臂外上, 1:左臂外下, 2:中心左下, 3:下臂左, 4:下臂右, 
    # 5:中心右下, 6:右臂外下, 7:右臂外上, 8:中心右上, 9:上臂右, 10:上臂左, 11:中心左上
    ideal_vertices = np.array([
        [mid_x - al - hw, mid_y - hw],  # 0: 左臂外侧上
        [mid_x - al - hw, mid_y + hw],  # 1: 左臂外侧下
        [mid_x - hw, mid_y + hw],       # 2: 中心左下
        [mid_x - hw, mid_y + al + hw],  # 3: 下臂左侧
        [mid_x + hw, mid_y + al + hw],  # 4: 下臂右侧
        [mid_x + hw, mid_y + hw],       # 5: 中心右下
        [mid_x + al + hw, mid_y + hw],  # 6: 右臂外侧下
        [mid_x + al + hw, mid_y - hw],  # 7: 右臂外侧上
        [mid_x + hw, mid_y - hw],       # 8: 中心右上
        [mid_x + hw, mid_y - al - hw],  # 9: 上臂右侧
        [mid_x - hw, mid_y - al - hw],  # 10: 上臂左侧
        [mid_x - hw, mid_y - hw],       # 11: 中心左上
    ], dtype=np.float32)
    
    ideal_center_pos = (int(mid_x), int(mid_y))
    
    print(f"\n[12-Point Homography] Using ALL 12 points for optimal transformation")
    
    # 使用所有12个点计算最优变换矩阵 (RANSAC鲁棒性更强)
    src_pts = np.array(original_vertices, dtype=np.float32)
    dst_pts = ideal_vertices
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    inliers = np.sum(mask) if mask is not None else 12
    print(f"    Inliers: {inliers}/12 points matched")
    
    # Use expanded canvas size for warping
    warp_w, warp_h = expanded_w, expanded_h
    warped_frame = cv2.warpPerspective(cropped_frame, M, (warp_w, warp_h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(240, 240, 240))
    
    # Transform all 12 original vertices
    orig_pts = np.array(original_vertices, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(orig_pts, M).reshape(-1, 2)
    
    # Colors
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ============ Top-Left: Cropped frame with original polygon ============
    ax1 = axes[0, 0]
    ax1.imshow(cropped_frame)
    poly = MplPolygon(np.array(original_vertices), fill=True, 
                      facecolor='cyan', edgecolor='yellow', alpha=0.3, linewidth=2)
    ax1.add_patch(poly)
    for i, (x, y) in enumerate(original_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    ax1.set_title(f'① Original Distorted Plus Maze\nSize: {w} x {h} px, Ratio: ~{actual_ratio:.1f}x', fontsize=12)
    ax1.axis('on')
    
    # ============ Top-Right: Ideal cross schematic ============
    ax2 = axes[0, 1]
    # Use expanded size for ideal image
    ideal_img = np.ones((expanded_h, expanded_w, 3), dtype=np.uint8) * 245
    ideal_pts_np = np.array(ideal_vertices, dtype=np.int32)
    cv2.fillPoly(ideal_img, [ideal_pts_np], (200, 255, 255))
    cv2.polylines(ideal_img, [ideal_pts_np], True, (0, 180, 180), 3)
    
    # Draw center square outline for reference
    cx, cy = ideal_center_pos
    hw = ideal_center_size // 2
    cv2.rectangle(ideal_img, (cx - hw, cy - hw), (cx + hw, cy + hw), (255, 100, 100), 2)
    
    ax2.imshow(ideal_img)
    for i, (x, y) in enumerate(ideal_vertices):
        ax2.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    
    # Add dimension labels
    ax2.text(cx, cy, f'Center\n{ideal_center_size}px', fontsize=8, ha='center', va='center',
             color='darkred', fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    expanded_note = f" (Canvas: {expanded_w}x{expanded_h})" if need_expand else ""
    ax2.set_title(f'② Ideal Plus Maze (Ratio {arm_length_ratio}:1)\nCenter={ideal_center_size}px, Arm={ideal_arm_length}px{expanded_note}', fontsize=11)
    ax2.axis('on')
    
    # ============ Bottom-Left: Warped frame ============
    ax3 = axes[1, 0]
    ax3.imshow(warped_frame)
    trans_poly = MplPolygon(transformed_pts, fill=True,
                            facecolor='cyan', edgecolor='yellow', alpha=0.3, linewidth=2)
    ax3.add_patch(trans_poly)
    for i, (x, y) in enumerate(transformed_pts):
        ax3.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', linewidths=2, zorder=10)
        ax3.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    warp_note = f" (Expanded to {warp_w}x{warp_h})" if need_expand else ""
    ax3.set_title(f'③ Perspective-Corrected Frame{warp_note}\n(Original stretched to {arm_length_ratio}:1 ratio)', fontsize=12)
    ax3.axis('on')
    
    # ============ Bottom-Right: Legend ============
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    ax4.text(0.5, 9.5, 'Vertex Legend:', fontsize=14, fontweight='bold', 
             transform=ax4.transAxes, verticalalignment='top')
    
    labels = [
        "0: Left arm (outer-top)",
        "1: Left arm (outer-bottom)",
        "2: Center-left (bottom)",
        "3: Bottom arm (left)",
        "4: Bottom arm (right)",
        "5: Center-right (bottom)",
        "6: Right arm (outer-bottom)",
        "7: Right arm (outer-top)",
        "8: Center-right (top)",
        "9: Top arm (right)",
        "10: Top arm (left)",
        "11: Center-left (top)"
    ]
    
    for i, label in enumerate(labels):
        y_pos = 8.5 - i * 0.6
        ax4.scatter([0.8], [y_pos], c=[colors[i]], s=100, edgecolors='black', 
                   linewidths=1, transform=ax4.transData)
        ax4.text(1.5, y_pos, label, fontsize=10, verticalalignment='center',
                transform=ax4.transData)
    
    ax4.text(0.5, 1.0, 
             'Note: Same colored/numbered points\ncorrespond between views.',
             fontsize=11, style='italic', transform=ax4.transData,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Cross Maze Perspective Transformation Analysis\n(With Crop Applied)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


# ============ Main execution ============
if __name__ == "__main__":
    # Your original polygon vertices (distorted cross maze)
    # These are in CROPPED coordinate system
    original_vertices = [
        (10, 248), (16, 347), (405, 331), (424, 569), 
        (536, 569), (525, 323), (822, 303), (821, 195), 
        (514, 206), (474, 1), (372, 0), (396, 211)
    ]
    
    # Crop parameters from notebook
    crop_params = {
        'x0': 128,
        'x1': 954,
        'y0': 0,
        'y1': 604
    }
    
    # Video path (original video, not time-cropped)
    video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    video_path = os.path.join(video_dir, "3pl1.mp4")
    
    # Output path
    output_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    output_path = os.path.join(output_dir, "CrossMaze_Transformation_Visualization.png")
    
    # Plus Maze dimensions (based on real measurements)
    # Original center width (point 11 to 8) = ~118 pixels = 90mm
    # 
    # NOTE: The ACTUAL arm/center ratio in your video is about 2.38x
    # If you want 5x ratio, the ideal cross will be much larger than original
    # 
    # Options:
    #   - use_actual_ratio=True: Keep original proportions, just rectify perspective
    #   - use_actual_ratio=False: Force 5x ratio (ideal will be scaled to fit)
    
    use_actual_ratio = False  # Set to False to use 5x ratio
    
    center_size_pixels = 118  # pixels, represents 90mm
    arm_length_ratio = 5 if not use_actual_ratio else None  # Will be calculated from original
    
    print("=" * 60)
    print("Plus Maze Perspective Transformation Visualizer")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Original vertices: {len(original_vertices)} points")
    print(f"Crop params: x0={crop_params['x0']}, x1={crop_params['x1']}, y0={crop_params['y0']}, y1={crop_params['y1']}")
    print(f"\nPlus Maze dimensions:")
    print(f"  Center square: {center_size_pixels} pixels = 90mm")
    if arm_length_ratio is not None:
        print(f"  Arm length: {center_size_pixels * arm_length_ratio} pixels = {90 * arm_length_ratio}mm")
    else:
        print(f"  Arm length: Will use actual ratio from video")
    
    # Run visualization with crop applied
    print("\nGenerating visualization with crop applied...")
    print(f"Mode: {'Use actual ratio from video' if use_actual_ratio else f'Force {arm_length_ratio}x ratio'}")
    fig = visualize_with_crop(
        video_path=video_path,
        original_vertices=original_vertices,
        crop_params=crop_params,
        output_path=output_path,
        center_size=center_size_pixels,
        arm_length_ratio=arm_length_ratio,  # None means use actual ratio
        show_plot=True
    )
    
    print("\nDone!")
