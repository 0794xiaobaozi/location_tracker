"""
Interactive Drag Tool to Adjust Left Arm Vertices
==================================================
Drag the red points (vertices 0 and 1) to adjust the left arm region.
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

# Global variables
fig = None
ax = None
cropped_frame = None
vertices = None
scatter_objects = []
poly_objects = []
selected_index = None
drag_threshold = 10  # pixels

class DraggablePoint:
    """A draggable point on the plot"""
    def __init__(self, point, index, ax, vertices_array, update_callback):
        self.point = point
        self.index = index
        self.ax = ax
        self.vertices = vertices_array
        self.update_callback = update_callback
        self.press = None
        self.background = None
        
        # Create scatter point
        self.scatter = ax.scatter(point[0], point[1], s=400, c='red' if index in [0, 1] else 'orange',
                                 edgecolors='black', linewidths=3, zorder=20, picker=5)
        self.text = ax.annotate(str(index), point, fontsize=14, fontweight='bold',
                               ha='center', va='center', color='white',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='red' if index in [0, 1] else 'orange', alpha=0.9),
                               zorder=21)
        
        # Connect events
        self.cidpress = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        # Check if this point was clicked
        contains, _ = self.scatter.contains(event)
        if not contains:
            return
        
        # Only allow dragging for left arm vertices (0 and 1)
        if self.index not in [0, 1]:
            return
        
        self.press = (self.point, event.xdata, event.ydata)
        self.background = fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.scatter)
        self.ax.draw_artist(self.text)
        fig.canvas.blit(self.ax.bbox)
    
    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        
        # Calculate new position
        orig_point, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        
        new_x = orig_point[0] + dx
        new_y = orig_point[1] + dy
        
        # Update point position
        self.point = (new_x, new_y)
        self.vertices[self.index] = (new_x, new_y)
        
        # Restore background
        fig.canvas.restore_region(self.background)
        
        # Update scatter and text positions
        self.scatter.set_offsets([[new_x, new_y]])
        self.text.set_position((new_x, new_y))
        
        # Redraw
        self.ax.draw_artist(self.scatter)
        self.ax.draw_artist(self.text)
        fig.canvas.blit(self.ax.bbox)
        
        # Update polygons
        self.update_callback()
    
    def on_release(self, event):
        if self.press is None:
            return
        
        self.press = None
        self.background = None
        
        # Final update
        self.update_callback()
        fig.canvas.draw()
        
        print(f"Updated Vertex {self.index}: ({int(self.point[0])}, {int(self.point[1])})")
    
    def disconnect(self):
        fig.canvas.mpl_disconnect(self.cidpress)
        fig.canvas.mpl_disconnect(self.cidrelease)
        fig.canvas.mpl_disconnect(self.cidmotion)

def update_polygons():
    """Update the polygon displays"""
    global poly_objects, vertices
    
    # Remove old polygons
    for poly in poly_objects:
        poly.remove()
    poly_objects.clear()
    
    # Draw left arm polygon
    left_arm_indices = [0, 1, 2, 11]
    left_arm_vertices = np.array([vertices[i] for i in left_arm_indices])
    left_poly = Polygon(left_arm_vertices, fill=True, facecolor='red', 
                       edgecolor='yellow', alpha=0.3, linewidth=3, zorder=5)
    ax.add_patch(left_poly)
    poly_objects.append(left_poly)
    
    # Draw all ROI polygon
    all_vertices = np.array(vertices)
    all_poly = Polygon(all_vertices, fill=False, edgecolor='cyan', 
                      linewidth=2, alpha=0.8, zorder=4)
    ax.add_patch(all_poly)
    poly_objects.append(all_poly)
    
    fig.canvas.draw_idle()

def save_vertices():
    """Save updated vertices to file"""
    global vertices
    
    vertices_str = "[\n"
    for i, (x, y) in enumerate(vertices):
        vertices_str += f"        ({int(x)}, {int(y)}),   # {i}"
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
    
    output_file = os.path.join(video_dir, "Updated_Vertices.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Updated original_vertices:\n")
        f.write("="*70 + "\n\n")
        f.write(vertices_str)
        f.write("\n\n")
        f.write("Files to update:\n")
        f.write("1. GenerateTransformedTrajectoryHeatmap.py\n")
        f.write("2. BatchGenerateTransformedTrajectories.py\n")
    
    print(f"\n{'='*70}")
    print("Vertices saved to: Updated_Vertices.txt")
    print(f"{'='*70}")

def on_key(event):
    """Handle keyboard events"""
    if event.key == 's' or event.key == 'S':
        save_vertices()
        print("Vertices saved! Press 'q' to quit.")
    elif event.key == 'q' or event.key == 'Q':
        save_vertices()
        plt.close('all')

def interactive_drag():
    """Main interactive drag function"""
    global fig, ax, cropped_frame, vertices
    
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
    
    # Convert vertices to list (mutable)
    vertices = list(current_vertices)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(cropped_frame)
    
    # Create draggable points for left arm vertices (0 and 1)
    draggable_points = []
    for i in [0, 1]:
        dp = DraggablePoint(vertices[i], i, ax, vertices, update_polygons)
        draggable_points.append(dp)
    
    # Draw other vertices (non-draggable)
    colors = plt.cm.rainbow(np.linspace(0, 1, 12))
    for i in range(12):
        if i not in [0, 1]:
            x, y = vertices[i]
            color = 'orange' if i in [2, 11] else colors[i]
            ax.scatter(x, y, c=[color], s=250, edgecolors='black', 
                      linewidths=2, zorder=10)
            ax.annotate(str(i), (x, y), fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    # Initial polygon drawing
    update_polygons()
    
    ax.set_title('Drag Red Points to Adjust Left Arm Vertices\n'
                'Red points (0, 1): Draggable - Left arm outer vertices\n'
                'Orange points (2, 11): Fixed - Shared with center\n'
                'Press "S" to save, "Q" to quit',
                fontsize=14, fontweight='bold')
    ax.axis('on')
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    return vertices

if __name__ == "__main__":
    print("=" * 70)
    print("Interactive Drag Tool - Left Arm Vertices Adjustment")
    print("=" * 70)
    print(f"\nInstructions:")
    print("1. A window will open showing the current vertices")
    print("2. Click and DRAG the RED points (vertices 0 and 1) to adjust")
    print("3. Orange points (2, 11) are fixed (shared with center)")
    print("4. Press 'S' to save the updated vertices")
    print("5. Press 'Q' to quit and save")
    print("\n" + "=" * 70)
    
    final_vertices = interactive_drag()
    
    if final_vertices:
        print(f"\nFinal vertices:")
        print(f"  Vertex 0: {final_vertices[0]}")
        print(f"  Vertex 1: {final_vertices[1]}")
