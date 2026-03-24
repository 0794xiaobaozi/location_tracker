"""
Regional Coordinate Transformation for Plus Maze
=================================================

This script implements piecewise/regional coordinate transformation for Plus Maze.
Instead of using a single global homography, it divides the maze into 5 regions
(center, left arm, right arm, top arm, bottom arm) and applies local transformations
to points within each region.

This approach is more robust when the original maze has non-uniform arm lengths
or complex perspective distortions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import distance
import os


def point_in_polygon(x, y, polygon_vertices):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Parameters:
    -----------
    x, y : float
        Point coordinates
    polygon_vertices : np.array
        Array of (x, y) vertices defining the polygon
    
    Returns:
    --------
    bool : True if point is inside polygon
    """
    n = len(polygon_vertices)
    inside = False
    
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def define_regions(vertices):
    """
    Define 5 regions of the Plus Maze based on 12 vertices.
    
    Regions:
    - Center: vertices 2, 5, 8, 11
    - Left arm: vertices 0, 1, 2, 11
    - Right arm: vertices 5, 6, 7, 8
    - Top arm: vertices 8, 9, 10, 11
    - Bottom arm: vertices 2, 3, 4, 5
    
    Parameters:
    -----------
    vertices : list of tuples
        12 vertices of the Plus Maze
    
    Returns:
    --------
    dict : Dictionary with region names as keys and vertex indices as values
    """
    regions = {
        'center': [2, 5, 8, 11],      # Center square
        'left_arm': [0, 1, 2, 11],    # Left arm
        'right_arm': [5, 6, 7, 8],    # Right arm
        'top_arm': [8, 9, 10, 11],    # Top arm
        'bottom_arm': [2, 3, 4, 5],   # Bottom arm
    }
    return regions


def create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size=1409):
    """
    Create ideal Plus Maze with 12 vertices.
    
    Parameters:
    -----------
    center_size : int
        Size of center square (pixels)
    arm_length_ratio : float
        Ratio of arm length to center size
    canvas_size : int
        Size of canvas (square)
    
    Returns:
    --------
    np.array : 12 ideal vertices
    """
    mid = canvas_size / 2
    hw = center_size / 2
    al = center_size * arm_length_ratio
    
    ideal_vertices = np.array([
        [mid - al - hw, mid - hw],  # 0: 左臂外侧上
        [mid - al - hw, mid + hw],  # 1: 左臂外侧下
        [mid - hw, mid + hw],       # 2: 中心左下
        [mid - hw, mid + al + hw],  # 3: 下臂左侧
        [mid + hw, mid + al + hw],  # 4: 下臂右侧
        [mid + hw, mid + hw],       # 5: 中心右下
        [mid + al + hw, mid + hw],  # 6: 右臂外侧下
        [mid + al + hw, mid - hw],  # 7: 右臂外侧上
        [mid + hw, mid - hw],       # 8: 中心右上
        [mid + hw, mid - al - hw],  # 9: 上臂右侧
        [mid - hw, mid - al - hw],  # 10: 上臂左侧
        [mid - hw, mid - hw],       # 11: 中心左上
    ], dtype=np.float32)
    
    return ideal_vertices


def compute_regional_transform(src_vertices, dst_vertices, region_indices):
    """
    Compute transformation matrix for a specific region.
    
    Parameters:
    -----------
    src_vertices : np.array
        Original 12 vertices
    dst_vertices : np.array
        Ideal 12 vertices
    region_indices : list
        Indices of vertices defining the region
    
    Returns:
    --------
    np.array : 3x3 transformation matrix
    """
    src_region = src_vertices[region_indices]
    dst_region = dst_vertices[region_indices]
    
    # Use findHomography with lower threshold for better fit
    # Method 0 = all points, no RANSAC (since we have exact correspondences)
    M, mask = cv2.findHomography(src_region, dst_region, method=0)
    
    return M


def determine_region(x, y, src_vertices):
    """
    Determine which region a point belongs to using distance-based method.
    
    Parameters:
    -----------
    x, y : float
        Point coordinates
    src_vertices : np.array
        12 vertices of the Plus Maze
    
    Returns:
    --------
    str : Region name
    """
    point = np.array([x, y])
    
    # Calculate center of the maze
    center_vertices = src_vertices[[2, 5, 8, 11]]
    center = np.mean(center_vertices, axis=0)
    
    # Calculate distances to center
    dist_to_center = np.linalg.norm(point - center)
    
    # Calculate center size (average of center edges)
    center_size = np.mean([
        np.linalg.norm(src_vertices[11] - src_vertices[8]),  # top
        np.linalg.norm(src_vertices[5] - src_vertices[2]),  # bottom
        np.linalg.norm(src_vertices[11] - src_vertices[2]), # left
        np.linalg.norm(src_vertices[8] - src_vertices[5]),  # right
    ])
    
    # If point is very close to center, it's in center region
    if dist_to_center < center_size * 0.7:
        return 'center'
    
    # Calculate direction from center to point
    direction = point - center
    angle = np.arctan2(direction[1], direction[0])  # -π to π
    
    # Convert to 0-2π range
    if angle < 0:
        angle += 2 * np.pi
    
    # Determine which arm based on angle
    # Left arm: 135° to 225° (3π/4 to 5π/4)
    # Top arm: 225° to 315° (5π/4 to 7π/4) or -135° to -45°
    # Right arm: -45° to 45° (315° to 45° or 7π/4 to π/4)
    # Bottom arm: 45° to 135° (π/4 to 3π/4)
    
    if 3 * np.pi / 4 <= angle <= 5 * np.pi / 4:
        return 'left_arm'
    elif 5 * np.pi / 4 <= angle <= 7 * np.pi / 4:
        return 'top_arm'
    elif angle <= np.pi / 4 or angle >= 7 * np.pi / 4:
        return 'right_arm'
    else:  # π/4 to 3π/4
        return 'bottom_arm'


def transform_point_regional(x, y, src_vertices, dst_vertices, regions):
    """
    Transform a point using regional transformations.
    
    Parameters:
    -----------
    x, y : float
        Point coordinates in original space
    src_vertices : np.array
        Original 12 vertices
    dst_vertices : np.array
        Ideal 12 vertices
    regions : dict
        Region definitions
    
    Returns:
    --------
    tuple : (transformed_x, transformed_y, region_name)
    """
    # Determine which region the point belongs to
    region_name = determine_region(x, y, src_vertices)
    
    # Get region vertex indices
    indices = regions[region_name]
    
    # Compute transformation for this region
    M = compute_regional_transform(src_vertices, dst_vertices, indices)
    
    # Transform the point
    pt = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, M)[0][0]
    
    return transformed[0], transformed[1], region_name


def generate_test_trajectory(vertices, num_points=200):
    """
    Generate a test trajectory that follows the Plus Maze shape.
    
    Parameters:
    -----------
    vertices : np.array
        12 vertices of the Plus Maze
    num_points : int
        Number of trajectory points
    
    Returns:
    --------
    np.array : Array of (x, y) trajectory points
    """
    trajectory = []
    
    # Create a path that goes through all arms
    # Path: center -> left arm -> center -> top arm -> center -> right arm -> center -> bottom arm
    
    # Center
    center = np.mean(vertices[[2, 5, 8, 11]], axis=0)
    
    # Left arm (from center to outer)
    left_outer = np.mean(vertices[[0, 1]], axis=0)
    for t in np.linspace(0, 1, num_points // 8):
        pt = center + t * (left_outer - center)
        trajectory.append(pt)
    
    # Back to center
    for t in np.linspace(1, 0, num_points // 8):
        pt = center + t * (left_outer - center)
        trajectory.append(pt)
    
    # Top arm
    top_outer = np.mean(vertices[[9, 10]], axis=0)
    for t in np.linspace(0, 1, num_points // 8):
        pt = center + t * (top_outer - center)
        trajectory.append(pt)
    
    # Back to center
    for t in np.linspace(1, 0, num_points // 8):
        pt = center + t * (top_outer - center)
        trajectory.append(pt)
    
    # Right arm
    right_outer = np.mean(vertices[[6, 7]], axis=0)
    for t in np.linspace(0, 1, num_points // 8):
        pt = center + t * (right_outer - center)
        trajectory.append(pt)
    
    # Back to center
    for t in np.linspace(1, 0, num_points // 8):
        pt = center + t * (right_outer - center)
        trajectory.append(pt)
    
    # Bottom arm
    bottom_outer = np.mean(vertices[[3, 4]], axis=0)
    for t in np.linspace(0, 1, num_points // 8):
        pt = center + t * (bottom_outer - center)
        trajectory.append(pt)
    
    # Fill remaining points in center
    while len(trajectory) < num_points:
        noise = np.random.normal(0, 10, 2)
        trajectory.append(center + noise)
    
    return np.array(trajectory[:num_points])


def visualize_individual_arms_transform(video_path, original_vertices, crop_params,
                                       center_size=119, arm_length_ratio=5,
                                       output_path=None, show_plot=True):
    """
    Visualize transformation for each arm separately.
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
    x1 = min(x1, orig_w)
    y1 = min(y1, orig_h)
    
    cropped_frame = frame_rgb[y0:y1, x0:x1]
    h, w = cropped_frame.shape[:2]
    
    # Convert vertices to numpy array
    src_vertices = np.array(original_vertices, dtype=np.float32)
    
    # Create ideal Plus Maze
    canvas_size = 1409
    ideal_arm_length = int(center_size * arm_length_ratio)
    dst_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
    
    # Define each arm separately
    arms = {
        'left_arm': {
            'indices': [0, 1, 2, 11],
            'color': 'blue',
            'name': 'Left Arm'
        },
        'right_arm': {
            'indices': [5, 6, 7, 8],
            'color': 'red',
            'name': 'Right Arm'
        },
        'top_arm': {
            'indices': [8, 9, 10, 11],
            'color': 'green',
            'name': 'Top Arm'
        },
        'bottom_arm': {
            'indices': [2, 3, 4, 5],
            'color': 'purple',
            'name': 'Bottom Arm'
        },
        'center': {
            'indices': [2, 5, 8, 11],
            'color': 'yellow',
            'name': 'Center'
        }
    }
    
    # Generate test points for each arm
    def generate_arm_points(arm_indices):
        """Generate points along an arm"""
        arm_vertices = src_vertices[arm_indices]
        points = []
        
        # Get outer and inner points (using relative indices within the 4-vertex arm)
        if arm_indices == [0, 1, 2, 11]:  # left arm: [0,1] outer, [2,3] inner (relative: [2,3])
            outer = np.mean(arm_vertices[[0, 1]], axis=0)  # relative indices 0,1
            inner = np.mean(arm_vertices[[2, 3]], axis=0)  # relative indices 2,3
        elif arm_indices == [5, 6, 7, 8]:  # right arm: [2,3] outer, [0,3] inner
            outer = np.mean(arm_vertices[[2, 3]], axis=0)  # relative indices 2,3
            inner = np.mean(arm_vertices[[0, 3]], axis=0)  # relative indices 0,3
        elif arm_indices == [8, 9, 10, 11]:  # top arm: [1,2] outer, [0,3] inner
            outer = np.mean(arm_vertices[[1, 2]], axis=0)  # relative indices 1,2
            inner = np.mean(arm_vertices[[0, 3]], axis=0)  # relative indices 0,3
        elif arm_indices == [2, 3, 4, 5]:  # bottom arm: [1,2] outer, [0,3] inner
            outer = np.mean(arm_vertices[[1, 2]], axis=0)  # relative indices 1,2
            inner = np.mean(arm_vertices[[0, 3]], axis=0)  # relative indices 0,3
        else:  # center
            inner = np.mean(arm_vertices, axis=0)
            outer = inner + np.array([20, 20])
        
        # Generate points along the arm
        for t in np.linspace(0, 1, 50):
            pt = inner + t * (outer - inner)
            points.append(pt)
        
        return np.array(points)
    
    # Create visualization: Use separate rows for each arm to avoid overlap
    # Row 1: Original, Ideal, Combined result
    # Row 2-5: Each arm gets its own row (original left, transformed right)
    # Row 6: Center region (make it larger)
    # Make each subplot larger while maintaining spacing
    fig = plt.figure(figsize=(48, 42))
    gs = fig.add_gridspec(6, 3, hspace=1.3, wspace=1.6, 
                          left=0.08, right=0.92, top=0.88, bottom=0.10,
                          height_ratios=[1.2, 1, 1, 1, 1, 2.6],  # Top row and last row (2x larger) larger
                          width_ratios=[1, 1, 1])  # Equal width columns
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # ============ Top-Left: Original ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cropped_frame)
    poly = MplPolygon(src_vertices, fill=True, facecolor='cyan', 
                     edgecolor='yellow', alpha=0.2, linewidth=2)
    ax1.add_patch(poly)
    
    for i, (x, y) in enumerate(src_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    
    ax1.set_title(f'① Original Plus Maze\nSize: {w} x {h} px', fontsize=16, fontweight='bold', pad=25)
    ax1.axis('on')
    
    # ============ Top-Middle: Ideal ============
    ax2 = fig.add_subplot(gs[0, 1])
    ideal_img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 245
    ideal_pts_np = dst_vertices.astype(np.int32)
    cv2.fillPoly(ideal_img, [ideal_pts_np], (200, 255, 255))
    cv2.polylines(ideal_img, [ideal_pts_np], True, (0, 180, 180), 3)
    ax2.imshow(ideal_img)
    
    for i, (x, y) in enumerate(dst_vertices):
        ax2.scatter(x, y, c=[colors[i]], s=180, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.8))
    
    ax2.set_title(f'② Ideal Plus Maze\nCenter={center_size}px, Arm={ideal_arm_length}px', 
                  fontsize=16, fontweight='bold', pad=25)
    ax2.axis('on')
    
    # ============ Top-Right: All arms combined ============
    ax_combined = fig.add_subplot(gs[0, 2])
    ax_combined.imshow(ideal_img)
    
    # Process each arm and draw on combined plot
    all_transformed_trajectories = {}
    for arm_name, arm_info in arms.items():
        if arm_name == 'center':
            continue
        
        arm_indices = arm_info['indices']
        arm_color = arm_info['color']
        
        # Compute transformation
        src_arm = src_vertices[arm_indices]
        dst_arm = dst_vertices[arm_indices]
        M = cv2.findHomography(src_arm, dst_arm, method=0)[0]
        
        # Generate and transform test points
        test_points = generate_arm_points(arm_indices)
        transformed_points = []
        for pt in test_points:
            pt_homogeneous = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt_homogeneous, M)[0][0]
            transformed_points.append(transformed)
        transformed_points = np.array(transformed_points)
        
        # Draw on combined plot
        ax_combined.plot(transformed_points[:, 0], transformed_points[:, 1], 
                        color=arm_color, linewidth=3, alpha=0.7, 
                        label=arm_info['name'])
        all_transformed_trajectories[arm_name] = transformed_points
    
    ax_combined.set_title('③ All Arms Combined', 
                         fontsize=16, fontweight='bold', pad=25)
    ax_combined.axis('on')
    ax_combined.legend(loc='upper right', fontsize=12)
    
    # ============ Bottom: Each arm side-by-side ============
    plot_idx = 0
    for arm_name, arm_info in arms.items():
        if arm_name == 'center':
            continue
        if plot_idx >= 4:
            break
        
        row = 1 + (plot_idx // 2)
        col = plot_idx % 2
        
        # Each arm gets its own row to avoid title overlap
        row = 1 + plot_idx  # Row 1, 2, 3, 4 for 4 arms
        col_left = 0
        col_right = 1
        
        # Left: Original arm
        ax_orig = fig.add_subplot(gs[row, col_left])
        ax_orig.imshow(cropped_frame)
        
        # Highlight this arm in original
        arm_indices = arm_info['indices']
        arm_color = arm_info['color']
        arm_display_name = arm_info['name']
        
        # Draw the arm polygon
        arm_poly = MplPolygon(src_vertices[arm_indices], fill=True, 
                             facecolor=arm_color, edgecolor='black', 
                             alpha=0.4, linewidth=2)
        ax_orig.add_patch(arm_poly)
        
        # Highlight vertices of this arm
        for idx in arm_indices:
            ax_orig.scatter(src_vertices[idx][0], src_vertices[idx][1], 
                          c=[colors[idx]], s=240, edgecolors='black', 
                          linewidths=2.5, zorder=10, marker='s')
            ax_orig.annotate(str(idx), (src_vertices[idx][0], src_vertices[idx][1]), 
                           fontsize=12, fontweight='bold', ha='center', va='center', 
                           color='white',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[idx], alpha=0.9))
        
        # Draw test trajectory for this arm
        test_points = generate_arm_points(arm_indices)
        ax_orig.plot(test_points[:, 0], test_points[:, 1], 
                    color='white', linewidth=3, alpha=0.8, linestyle='--')
        ax_orig.scatter(test_points[::5, 0], test_points[::5, 1], 
                       c='white', s=50, alpha=0.9, edgecolors='black', linewidths=0.8)
        
        # Title with more space - each arm in its own row so no overlap
        short_name = arm_display_name.replace(' Arm', '')
        ax_orig.set_title(f'{short_name} - Original', 
                         fontsize=15, fontweight='bold', color=arm_color, pad=45)
        ax_orig.axis('on')
        
        # Right: Transformed arm
        ax_trans = fig.add_subplot(gs[row, col_right])
        ax_trans.imshow(ideal_img)
        
        # Compute transformation
        src_arm = src_vertices[arm_indices]
        dst_arm = dst_vertices[arm_indices]
        M = cv2.findHomography(src_arm, dst_arm, method=0)[0]
        
        # Transform test points
        transformed_points = []
        for pt in test_points:
            pt_homogeneous = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt_homogeneous, M)[0][0]
            transformed_points.append(transformed)
        transformed_points = np.array(transformed_points)
        
        # Draw transformed trajectory
        ax_trans.plot(transformed_points[:, 0], transformed_points[:, 1], 
                     color=arm_color, linewidth=3.5, alpha=0.8, label='Transformed')
        ax_trans.scatter(transformed_points[::5, 0], transformed_points[::5, 1], 
                        c=arm_color, s=60, alpha=0.9, edgecolors='black', linewidths=1.5)
        
        # Draw transformed vertices
        for idx in arm_indices:
            pt_homogeneous = np.array([[[src_vertices[idx][0], src_vertices[idx][1]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt_homogeneous, M)[0][0]
            ax_trans.scatter(transformed[0], transformed[1], c=[colors[idx]], s=220, 
                           edgecolors='black', linewidths=2.5, zorder=10, marker='s')
            ax_trans.annotate(str(idx), (transformed[0], transformed[1]), 
                            fontsize=11, fontweight='bold', ha='center', va='center', 
                            color='white',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[idx], alpha=0.9))
            
            # Draw ideal position for comparison
            ideal_pt = dst_vertices[idx]
            ax_trans.scatter(ideal_pt[0], ideal_pt[1], c=[colors[idx]], s=180, 
                           edgecolors='white', linewidths=3, zorder=9, marker='x')
        
        # Title with more space
        ax_trans.set_title(f'{short_name} - Transformed', 
                          fontsize=15, fontweight='bold', color=arm_color, pad=45)
        ax_trans.axis('on')
        ax_trans.legend(fontsize=12)
        
        plot_idx += 1
    
    # ============ Center region transformation ============
    center_info = arms['center']
    center_indices = center_info['indices']
    center_color = center_info['color']
    
    # Left: Original center
    row_center = 5
    ax_center_orig = fig.add_subplot(gs[row_center, 0])
    ax_center_orig.imshow(cropped_frame)
    
    # Draw center polygon
    center_poly = MplPolygon(src_vertices[center_indices], fill=True, 
                            facecolor=center_color, edgecolor='black', 
                            alpha=0.4, linewidth=2)
    ax_center_orig.add_patch(center_poly)
    
    # Highlight center vertices
    for idx in center_indices:
        ax_center_orig.scatter(src_vertices[idx][0], src_vertices[idx][1], 
                              c=[colors[idx]], s=240, edgecolors='black', 
                              linewidths=2.5, zorder=10, marker='s')
        ax_center_orig.annotate(str(idx), (src_vertices[idx][0], src_vertices[idx][1]), 
                               fontsize=12, fontweight='bold', ha='center', va='center', 
                               color='white',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[idx], alpha=0.9))
    
    # Generate test points in center
    center_vertices = src_vertices[center_indices]
    center_center = np.mean(center_vertices, axis=0)
    center_test_points = []
    for _ in range(50):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, np.min([
            np.linalg.norm(center_vertices[0] - center_center),
            np.linalg.norm(center_vertices[1] - center_center),
        ]) * 0.8)
        pt = center_center + radius * np.array([np.cos(angle), np.sin(angle)])
        center_test_points.append(pt)
    center_test_points = np.array(center_test_points)
    
    ax_center_orig.scatter(center_test_points[:, 0], center_test_points[:, 1], 
                          c='white', s=50, alpha=0.8, edgecolors='black', linewidths=0.8)
    
    ax_center_orig.set_title('Center - Original', 
                            fontsize=15, fontweight='bold', color=center_color, pad=45)
    ax_center_orig.axis('on')
    
    # Right: Transformed center
    ax_center_trans = fig.add_subplot(gs[row_center, 1])
    ax_center_trans.imshow(ideal_img)
    
    # Compute center transformation
    src_center = src_vertices[center_indices]
    dst_center = dst_vertices[center_indices]
    M_center = cv2.findHomography(src_center, dst_center, method=0)[0]
    
    # Transform center test points
    center_transformed_points = []
    for pt in center_test_points:
        pt_homogeneous = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_homogeneous, M_center)[0][0]
        center_transformed_points.append(transformed)
    center_transformed_points = np.array(center_transformed_points)
    
    # Draw transformed points
    ax_center_trans.scatter(center_transformed_points[:, 0], center_transformed_points[:, 1], 
                           c=center_color, s=60, alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # Draw transformed vertices
    for idx in center_indices:
        pt_homogeneous = np.array([[[src_vertices[idx][0], src_vertices[idx][1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_homogeneous, M_center)[0][0]
        ax_center_trans.scatter(transformed[0], transformed[1], c=[colors[idx]], s=220, 
                               edgecolors='black', linewidths=2.5, zorder=10, marker='s')
        ax_center_trans.annotate(str(idx), (transformed[0], transformed[1]), 
                                fontsize=11, fontweight='bold', ha='center', va='center', 
                                color='white',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[idx], alpha=0.9))
        
        # Draw ideal position
        ideal_pt = dst_vertices[idx]
        ax_center_trans.scatter(ideal_pt[0], ideal_pt[1], c=[colors[idx]], s=180, 
                               edgecolors='white', linewidths=3, zorder=9, marker='x')
    
    ax_center_trans.set_title('Center - Transformed', 
                             fontsize=15, fontweight='bold', color=center_color, pad=45)
    ax_center_trans.axis('on')
    
    fig.suptitle('Individual Arm Transformations for Plus Maze', 
                 fontsize=22, fontweight='bold', y=0.98)
    fig.text(0.5, 0.96, '(Each arm transformed separately using 4-point homography)', 
             ha='center', fontsize=14, style='italic')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    plt.close(fig)  # Close figure instead of showing
    
    return fig


def visualize_regional_transform(video_path, original_vertices, crop_params,
                                 center_size=119, arm_length_ratio=5,
                                 output_path=None, show_plot=True):
    """
    Visualize regional coordinate transformation.
    
    Parameters:
    -----------
    video_path : str
        Path to video file
    original_vertices : list
        12 original vertices
    crop_params : dict
        Crop parameters
    center_size : int
        Center square size
    arm_length_ratio : float
        Arm length ratio
    output_path : str
        Output image path
    show_plot : bool
        Whether to display plot
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
    x1 = min(x1, orig_w)
    y1 = min(y1, orig_h)
    
    cropped_frame = frame_rgb[y0:y1, x0:x1]
    h, w = cropped_frame.shape[:2]
    
    # Convert vertices to numpy array
    src_vertices = np.array(original_vertices, dtype=np.float32)
    
    # Create ideal Plus Maze
    canvas_size = 1409
    ideal_arm_length = int(center_size * arm_length_ratio)
    dst_vertices = create_ideal_plus_maze(center_size, arm_length_ratio, canvas_size)
    
    # Define regions
    regions = define_regions(src_vertices)
    
    # Generate test trajectory
    test_trajectory = generate_test_trajectory(src_vertices, num_points=300)
    
    # Try BOTH methods: regional and global
    print("\nComputing transformations...")
    
    # Method 1: Global 12-point transformation
    print("  Method 1: Global 12-point homography")
    M_global, mask_global = cv2.findHomography(src_vertices, dst_vertices, cv2.RANSAC, 5.0)
    inliers_global = np.sum(mask_global) if mask_global is not None else 12
    print(f"    Inliers: {inliers_global}/12")
    
    # Transform trajectory with global method
    global_trajectory = []
    for pt in test_trajectory:
        pt_homogeneous = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_homogeneous, M_global)[0][0]
        global_trajectory.append([transformed[0], transformed[1], 'global'])
    global_trajectory = np.array(global_trajectory)
    
    # Method 2: Regional transformation
    print("  Method 2: Regional piecewise transformation")
    transformed_trajectory = []
    region_colors = {
        'center': 'yellow',
        'left_arm': 'blue',
        'right_arm': 'red',
        'top_arm': 'green',
        'bottom_arm': 'purple',
        'global': 'cyan'
    }
    
    for pt in test_trajectory:
        x, y, region = transform_point_regional(pt[0], pt[1], src_vertices, dst_vertices, regions)
        transformed_trajectory.append([x, y, region])
    
    transformed_trajectory = np.array(transformed_trajectory)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    
    # ============ Top-Left: Original with trajectory ============
    ax1 = axes[0, 0]
    ax1.imshow(cropped_frame)
    
    # Draw polygon
    poly = MplPolygon(src_vertices, fill=True, facecolor='cyan', 
                     edgecolor='yellow', alpha=0.2, linewidth=2)
    ax1.add_patch(poly)
    
    # Draw trajectory
    ax1.plot(test_trajectory[:, 0], test_trajectory[:, 1], 
            'w-', linewidth=2, alpha=0.6, label='Test Trajectory')
    ax1.scatter(test_trajectory[::10, 0], test_trajectory[::10, 1], 
               c='white', s=20, alpha=0.8)
    
    # Draw vertices
    for i, (x, y) in enumerate(src_vertices):
        ax1.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax1.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    
    ax1.set_title(f'① Original Plus Maze with Test Trajectory\nSize: {w} x {h} px', fontsize=12)
    ax1.axis('on')
    ax1.legend()
    
    # ============ Top-Right: Ideal Plus Maze ============
    ax2 = axes[0, 1]
    ideal_img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 245
    ideal_pts_np = dst_vertices.astype(np.int32)
    cv2.fillPoly(ideal_img, [ideal_pts_np], (200, 255, 255))
    cv2.polylines(ideal_img, [ideal_pts_np], True, (0, 180, 180), 3)
    
    ax2.imshow(ideal_img)
    
    # Draw vertices
    for i, (x, y) in enumerate(dst_vertices):
        ax2.scatter(x, y, c=[colors[i]], s=150, edgecolors='black', 
                   linewidths=2, zorder=10)
        ax2.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.8))
    
    ax2.set_title(f'② Ideal Plus Maze (Ratio {arm_length_ratio}:1)\nCenter={center_size}px, Arm={ideal_arm_length}px', fontsize=11)
    ax2.axis('on')
    
    # ============ Bottom-Left: Global transform ============
    ax3 = axes[1, 0]
    ax3.imshow(ideal_img)
    
    # Draw global transformed trajectory
    ax3.plot(global_trajectory[:, 0], global_trajectory[:, 1], 
            color='cyan', linewidth=2, alpha=0.7, label='Global 12-point')
    ax3.scatter(global_trajectory[::10, 0], global_trajectory[::10, 1], 
               c='cyan', s=30, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    ax3.set_title('③ Global Transform (12-point)\nAll points transformed together', fontsize=11)
    ax3.axis('on')
    ax3.legend()
    
    # ============ Bottom-Middle: Regional transform ============
    ax4 = axes[1, 1]
    ax4.imshow(ideal_img)
    
    # Draw trajectory colored by region
    for region_name, color in region_colors.items():
        if region_name == 'global':
            continue
        mask = transformed_trajectory[:, 2] == region_name
        if np.any(mask):
            region_pts = transformed_trajectory[mask]
            ax4.plot(region_pts[:, 0], region_pts[:, 1], 
                    color=color, linewidth=2, alpha=0.7, label=region_name)
            ax4.scatter(region_pts[::5, 0], region_pts[::5, 1], 
                       c=color, s=30, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    ax4.set_title('④ Regional Transform\n(Colored by Region)', fontsize=11)
    ax4.axis('on')
    ax4.legend(loc='upper right', fontsize=8)
    
    # ============ Bottom-Right: Comparison and info ============
    ax5 = axes[1, 2]
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    
    ax5.text(0.5, 9.5, 'Comparison', fontsize=14, fontweight='bold', 
             transform=ax5.transAxes, verticalalignment='top', ha='center')
    
    ax5.text(0.1, 8.5, 'Global Method:', fontsize=11, fontweight='bold', 
             transform=ax5.transAxes, color='cyan')
    ax5.text(0.1, 8.0, f'• Uses all 12 points', fontsize=9, transform=ax5.transAxes)
    ax5.text(0.1, 7.5, f'• Single transformation', fontsize=9, transform=ax5.transAxes)
    ax5.text(0.1, 7.0, f'• Inliers: {inliers_global}/12', fontsize=9, transform=ax5.transAxes)
    
    ax5.text(0.1, 6.0, 'Regional Method:', fontsize=11, fontweight='bold', 
             transform=ax5.transAxes)
    ax5.text(0.1, 5.5, '• 5 regions (center + 4 arms)', fontsize=9, transform=ax5.transAxes)
    ax5.text(0.1, 5.0, '• Each region: 4-point transform', fontsize=9, transform=ax5.transAxes)
    ax5.text(0.1, 4.5, '• Handles non-uniform arms', fontsize=9, transform=ax5.transAxes)
    
    ax5.text(0.1, 3.0, 'Which is better?', fontsize=11, fontweight='bold', 
             transform=ax5.transAxes, style='italic')
    ax5.text(0.1, 2.5, 'Compare plots ③ and ④', fontsize=9, 
             transform=ax5.transAxes, style='italic')
    
    fig.suptitle('Regional Coordinate Transformation for Plus Maze', 
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
    # Your original polygon vertices
    original_vertices = [
        (10, 248), (16, 347), (405, 331), (424, 569), 
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
    
    # Video path
    video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    video_path = os.path.join(video_dir, "3pl1.mp4")
    
    # Output path
    output_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
    output_path = os.path.join(output_dir, "RegionalTransform_Visualization.png")
    
    print("=" * 60)
    print("Regional Coordinate Transformation Visualizer")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Original vertices: {len(original_vertices)} points")
    print(f"Center size: 119 px = 90mm")
    print(f"Arm length ratio: 5x (arm = 595 px = 450mm)")
    
    # Run visualization
    print("\nGenerating individual arm transformations visualization...")
    fig = visualize_individual_arms_transform(
        video_path=video_path,
        original_vertices=original_vertices,
        crop_params=crop_params,
        center_size=119,
        arm_length_ratio=5,
        output_path=output_path,
        show_plot=False  # Don't show, just save
    )
    
    print("\nDone!")
