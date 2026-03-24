"""
Check the actual ROI bounding box size
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

# Calculate bounding box
all_x = vertices[:, 0]
all_y = vertices[:, 1]

min_x, max_x = all_x.min(), all_x.max()
min_y, max_y = all_y.min(), all_y.max()

width = max_x - min_x
height = max_y - min_y

print("=" * 60)
print("Original ROI Bounding Box")
print("=" * 60)
print(f"\nX range: {min_x:.0f} to {max_x:.0f} (width: {width:.0f} px)")
print(f"Y range: {min_y:.0f} to {max_y:.0f} (height: {height:.0f} px)")
print(f"\nBounding box size: {width:.0f} x {height:.0f} px")

# Crop parameters
crop_params = {
    'x0': 128,
    'x1': 954,
    'y0': 0,
    'y1': 604
}

crop_width = crop_params['x1'] - crop_params['x0']
crop_height = crop_params['y1'] - crop_params['y0']

print(f"\nCrop region size: {crop_width} x {crop_height} px")
print(f"\nROI occupies: {width/crop_width*100:.1f}% of crop width, {height/crop_height*100:.1f}% of crop height")

# Check if ROI is too small relative to crop
print(f"\n" + "=" * 60)
print("Analysis:")
print("=" * 60)
if width < crop_width * 0.5 or height < crop_height * 0.5:
    print("[WARNING] ROI might be too small relative to crop region")
    print(f"   Consider expanding the ROI selection")
else:
    print("[OK] ROI size seems reasonable")
