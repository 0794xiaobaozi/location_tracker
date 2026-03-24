"""
Interactive Tool to Adjust Left Arm Vertices
============================================
This script helps you visualize and adjust the left arm vertices for transformation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

# Current original vertices
current_vertices = [
    (10, 248),   # 0: 左臂外侧上
    (16, 347),   # 1: 左臂外侧下
    (405, 331),  # 2: 中心左下 (shared with center)
    (424, 569),  # 3: 下臂左侧
    (536, 569),  # 4: 下臂右侧
    (525, 323),  # 5: 中心右下
    (822, 303),  # 6: 右臂外侧下
    (821, 195),  # 7: 右臂外侧上
    (514, 206),  # 8: 中心右上
    (474, 1),    # 9: 上臂右侧
    (372, 0),    # 10: 上臂左侧
    (396, 211),  # 11: 中心左上 (shared with center)
]

# Left arm vertices: 0, 1, 2, 11
left_arm_indices = [0, 1, 2, 11]

# Configuration
video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
video_file = "3pl1.mp4"  # Use one video as reference
video_path = os.path.join(video_dir, video_file)

crop_params = {
    'x0': 128,
    'x1': 954,
    'y0': 0,
    'y1': 604
}

def visualize_current_vertices():
    """Visualize current vertices with left arm highlighted"""
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not load video {video_path}")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    
    # Apply crop
    x0, x1 = crop_params['x0'], crop_params['x1']
    y0, y1 = crop_params['y0'], crop_params['y1']
    x1 = min(x1, orig_w)
    y1 = min(y1, orig_h)
    
    cropped_frame = frame_rgb[y0:y1, x0:x1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(cropped_frame)
    
    # Convert vertices to numpy array
    vertices = np.array(current_vertices, dtype=np.float32)
    
    # Draw all vertices
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    for i, (x, y) in enumerate(vertices):
        color = 'red' if i in left_arm_indices else colors[i]
        ax.scatter(x, y, c=[color], s=200, edgecolors='black', 
                  linewidths=2, zorder=10)
        ax.annotate(str(i), (x, y), fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    # Draw left arm polygon (highlighted)
    left_arm_vertices = vertices[left_arm_indices]
    poly = Polygon(left_arm_vertices, fill=True, facecolor='red', 
                  edgecolor='yellow', alpha=0.3, linewidth=3)
    ax.add_patch(poly)
    
    # Draw all ROI polygon
    all_poly = Polygon(vertices, fill=False, edgecolor='cyan', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(all_poly)
    
    ax.set_title('Current Vertices - Left Arm Highlighted in Red\n'
                'Left Arm uses vertices: 0, 1, 2, 11\n'
                'To adjust: Modify vertices 0 and 1 (outer left arm points)',
                fontsize=14, fontweight='bold')
    ax.axis('on')
    
    plt.tight_layout()
    
    output_path = os.path.join(video_dir, "LeftArm_CurrentVertices.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    print(f"\nCurrent Left Arm Vertices:")
    print(f"  Vertex 0 (左臂外侧上): {current_vertices[0]}")
    print(f"  Vertex 1 (左臂外侧下): {current_vertices[1]}")
    print(f"  Vertex 2 (中心左下): {current_vertices[2]}  [shared with center]")
    print(f"  Vertex 11 (中心左上): {current_vertices[11]}  [shared with center]")
    
    plt.show()
    
    return vertices

def update_vertices(new_vertex_0=None, new_vertex_1=None):
    """
    Update left arm vertices.
    
    Parameters:
    -----------
    new_vertex_0 : tuple or None
        New coordinates for vertex 0 (左臂外侧上)
    new_vertex_1 : tuple or None
        New coordinates for vertex 1 (左臂外侧下)
    """
    global current_vertices
    
    if new_vertex_0:
        current_vertices[0] = new_vertex_0
        print(f"Updated vertex 0 to: {new_vertex_0}")
    
    if new_vertex_1:
        current_vertices[1] = new_vertex_1
        print(f"Updated vertex 1 to: {new_vertex_1}")
    
    return current_vertices

if __name__ == "__main__":
    print("=" * 70)
    print("Left Arm Vertices Adjustment Tool")
    print("=" * 70)
    print(f"\nVideo reference: {video_file}")
    print(f"\nCurrent left arm vertices:")
    print(f"  Vertex 0 (左臂外侧上): {current_vertices[0]}")
    print(f"  Vertex 1 (左臂外侧下): {current_vertices[1]}")
    print(f"  Vertex 2 (中心左下): {current_vertices[2]}  [shared with center]")
    print(f"  Vertex 11 (中心左上): {current_vertices[11]}  [shared with center]")
    print("\n" + "=" * 70)
    
    # Visualize current vertices
    vertices = visualize_current_vertices()
    
    print("\n" + "=" * 70)
    print("To adjust left arm vertices:")
    print("=" * 70)
    print("1. Look at the visualization above")
    print("2. Identify the correct coordinates for vertices 0 and 1")
    print("3. Update the vertices in the script or provide new coordinates")
    print("\nExample usage:")
    print("  update_vertices(new_vertex_0=(20, 250), new_vertex_1=(25, 350))")
    print("\nAfter updating, you need to:")
    print("  1. Update original_vertices in GenerateTransformedTrajectoryHeatmap.py")
    print("  2. Update original_vertices in BatchGenerateTransformedTrajectories.py")
    print("  3. Regenerate all transformed trajectories")
