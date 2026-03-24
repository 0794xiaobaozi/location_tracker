"""
Interactive Tool to Adjust Left Arm Vertices by Clicking
========================================================
Click on the image to select new left arm vertex positions.
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

# Configuration
video_dir = r"F:\Neuro\ezTrack\LocationTracking\video\cropped_video\192.0.0.64_8000_1_2B1588C028414C97BC36CA24B9285625_"
video_file = "3pl1.mp4"
video_path = os.path.join(video_dir, video_file)

crop_params = {
    'x0': 128,
    'x1': 954,
    'y0': 0,
    'y1': 604
}

# Global variables for interactive selection
clicked_points = []
fig = None
ax = None
cropped_frame = None
vertices_display = None

def on_click(event):
    """Handle mouse clicks to select new vertex positions"""
    global clicked_points, ax, cropped_frame, vertices_display
    
    if event.inaxes != ax:
        return
    
    if event.button == 1:  # Left click
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        
        print(f"\nClicked point {len(clicked_points)}: ({x}, {y})")
        
        # Update display
        ax.clear()
        ax.imshow(cropped_frame)
        
        # Draw all vertices
        vertices = np.array(current_vertices, dtype=np.float32)
        colors = plt.cm.rainbow(np.linspace(0, 1, 12))
        
        for i, (vx, vy) in enumerate(vertices):
            if i in [0, 1]:  # Left arm outer vertices
                color = 'red'
                size = 300
            elif i in [2, 11]:  # Shared vertices
                color = 'orange'
                size = 250
            else:
                color = colors[i]
                size = 200
            
            ax.scatter(vx, vy, c=[color], s=size, edgecolors='black', 
                      linewidths=2, zorder=10)
            ax.annotate(str(i), (vx, vy), fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        
        # Draw clicked points
        for i, (cx, cy) in enumerate(clicked_points, 1):
            ax.scatter(cx, cy, c='yellow', s=400, edgecolors='black', 
                      linewidths=3, marker='X', zorder=15)
            ax.annotate(f'New {i}', (cx, cy), fontsize=14, fontweight='bold',
                       ha='center', va='center', color='black',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9))
        
        # Draw left arm polygon
        left_arm_indices = [0, 1, 2, 11]
        left_arm_vertices = vertices[left_arm_indices]
        poly = Polygon(left_arm_vertices, fill=True, facecolor='red', 
                      edgecolor='yellow', alpha=0.3, linewidth=3)
        ax.add_patch(poly)
        
        # Draw all ROI polygon
        all_poly = Polygon(vertices, fill=False, edgecolor='cyan', 
                          linewidth=2, alpha=0.8)
        ax.add_patch(all_poly)
        
        ax.set_title('Click to Select New Left Arm Vertices\n'
                    'Click 1: New Vertex 0 (左臂外侧上)\n'
                    'Click 2: New Vertex 1 (左臂外侧下)\n'
                    'Right-click or close window when done',
                    fontsize=14, fontweight='bold')
        ax.axis('on')
        
        plt.draw()
        
        # If we have 2 clicks, we're done
        if len(clicked_points) >= 2:
            print(f"\n{'='*70}")
            print("Selection Complete!")
            print(f"{'='*70}")
            print(f"New Vertex 0 (左臂外侧上): {clicked_points[0]}")
            print(f"New Vertex 1 (左臂外侧下): {clicked_points[1]}")
            print(f"\nCurrent Vertex 2 (中心左下): {current_vertices[2]} [keep as is]")
            print(f"Current Vertex 11 (中心左上): {current_vertices[11]} [keep as is]")
            print(f"\n{'='*70}")
            print("Updating vertices...")
            update_all_scripts(clicked_points[0], clicked_points[1])
            print("Done! Please regenerate transformed trajectories.")

def update_all_scripts(new_vertex_0, new_vertex_1):
    """Update vertices in all relevant scripts"""
    
    new_vertices = current_vertices.copy()
    new_vertices[0] = new_vertex_0
    new_vertices[1] = new_vertex_1
    
    # Format for Python code
    vertices_str = "[\n"
    for i, (x, y) in enumerate(new_vertices):
        vertices_str += f"        ({x}, {y}),   # {i}"
        if i == 0:
            vertices_str += ": 左臂外侧上"
        elif i == 1:
            vertices_str += ": 左臂外侧下"
        elif i == 2:
            vertices_str += ": 中心左下"
        elif i == 11:
            vertices_str += ": 中心左上"
        vertices_str += "\n"
    vertices_str += "    ]"
    
    print(f"\nNew vertices array:")
    print(vertices_str)
    
    # Save to a file for easy copy-paste
    output_file = os.path.join(video_dir, "Updated_Vertices.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Updated original_vertices:\n")
        f.write("="*70 + "\n\n")
        f.write(vertices_str)
        f.write("\n\n")
        f.write("Files to update:\n")
        f.write("1. GenerateTransformedTrajectoryHeatmap.py\n")
        f.write("2. BatchGenerateTransformedTrajectories.py\n")
        f.write("3. GenerateGroupMeanHeatmap.py (if needed)\n")
    
    print(f"\nSaved updated vertices to: {output_file}")
    print("\nPlease update the following files manually:")
    print("  1. GenerateTransformedTrajectoryHeatmap.py")
    print("  2. BatchGenerateTransformedTrajectories.py")
    print("  3. GenerateGroupMeanHeatmap.py (if it has vertices)")
    print("\nOr I can update them automatically. Should I proceed?")

def interactive_adjust():
    """Main interactive adjustment function"""
    global fig, ax, cropped_frame, vertices_display
    
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
        if i in [0, 1]:  # Left arm outer vertices
            color = 'red'
            size = 300
        elif i in [2, 11]:  # Shared vertices
            color = 'orange'
            size = 250
        else:
            color = colors[i]
            size = 200
        
        ax.scatter(x, y, c=[color], s=size, edgecolors='black', 
                  linewidths=2, zorder=10)
        ax.annotate(str(i), (x, y), fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    # Draw left arm polygon (highlighted)
    left_arm_indices = [0, 1, 2, 11]
    left_arm_vertices = vertices[left_arm_indices]
    poly = Polygon(left_arm_vertices, fill=True, facecolor='red', 
                  edgecolor='yellow', alpha=0.3, linewidth=3)
    ax.add_patch(poly)
    
    # Draw all ROI polygon
    all_poly = Polygon(vertices, fill=False, edgecolor='cyan', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(all_poly)
    
    ax.set_title('Click to Select New Left Arm Vertices\n'
                'Click 1: New Vertex 0 (左臂外侧上) - Red point\n'
                'Click 2: New Vertex 1 (左臂外侧下) - Red point\n'
                'Right-click or close window when done',
                fontsize=14, fontweight='bold')
    ax.axis('on')
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.show()
    
    return clicked_points

if __name__ == "__main__":
    print("=" * 70)
    print("Interactive Left Arm Vertices Adjustment")
    print("=" * 70)
    print(f"\nInstructions:")
    print("1. A window will open showing the current vertices")
    print("2. Click on the image to select new Vertex 0 (左臂外侧上)")
    print("3. Click again to select new Vertex 1 (左臂外侧下)")
    print("4. The window will show your selections and update the vertices")
    print("5. Close the window when done")
    print("\n" + "=" * 70)
    
    points = interactive_adjust()
    
    if len(points) >= 2:
        print(f"\nSelected points:")
        print(f"  Vertex 0: {points[0]}")
        print(f"  Vertex 1: {points[1]}")
