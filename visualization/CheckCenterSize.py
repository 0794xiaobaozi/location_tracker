"""
Check the actual center size from original vertices
"""
import numpy as np

# Original polygon vertices (from notebook)
original_vertices = [
    (10, 248), (16, 347), (405, 331), (424, 569), 
    (536, 569), (525, 323), (822, 303), (821, 195), 
    (514, 206), (474, 1), (372, 0), (396, 211)
]

# Convert to numpy array
vertices = np.array(original_vertices, dtype=np.float32)

# Center vertices (indices 2, 5, 8, 11)
# 2: center-left-bottom
# 5: center-right-bottom  
# 8: center-right-top
# 11: center-left-top

p2 = vertices[2]   # (405, 331)
p5 = vertices[5]   # (525, 323)
p8 = vertices[8]   # (514, 206)
p11 = vertices[11] # (396, 211)

print("=" * 60)
print("Original Center Region Dimensions")
print("=" * 60)
print(f"\nVertices:")
print(f"  Point 2 (center-left-bottom):  {p2}")
print(f"  Point 5 (center-right-bottom): {p5}")
print(f"  Point 8 (center-right-top):    {p8}")
print(f"  Point 11 (center-left-top):    {p11}")

# Calculate center edges
top_edge = np.linalg.norm(p8 - p11)      # Top edge (11 to 8)
bottom_edge = np.linalg.norm(p5 - p2)    # Bottom edge (2 to 5)
left_edge = np.linalg.norm(p11 - p2)     # Left edge (2 to 11)
right_edge = np.linalg.norm(p8 - p5)     # Right edge (5 to 8)

print(f"\nCenter Edge Lengths:")
print(f"  Top edge (11→8):    {top_edge:.2f} px")
print(f"  Bottom edge (2→5):  {bottom_edge:.2f} px")
print(f"  Left edge (2→11):   {left_edge:.2f} px")
print(f"  Right edge (5→8):   {right_edge:.2f} px")

avg_center_size = np.mean([top_edge, bottom_edge, left_edge, right_edge])
print(f"\nAverage center size: {avg_center_size:.2f} px")

# User mentioned: "原图是center top 的left和right两个点之间的像素代表现实的距离是90 mm"
# This is the top edge: p11 to p8
print(f"\nTop edge (11→8) = {top_edge:.2f} px = 90mm")
print(f"Current center_size setting: 119 px")
print(f"Difference: {119 - top_edge:.2f} px")

# Calculate what center_size should be based on actual measurements
print(f"\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)
print(f"If top edge = {top_edge:.2f} px = 90mm, then:")
print(f"  center_size should be approximately {top_edge:.0f} px")
print(f"  (or use average: {avg_center_size:.0f} px)")
