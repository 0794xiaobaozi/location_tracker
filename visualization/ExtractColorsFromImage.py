#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Colors from Reference Image
====================================
Extract the exact colors used in bar charts from the reference image.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

def extract_bar_colors(image_path, show_plot=True):
    """Extract colors from bar charts in the image"""
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print("=" * 80)
    print(f"Extracting colors from: {image_path}")
    print(f"Image size: {img_array.shape}")
    print("=" * 80)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
    
    # Sample regions where bars typically are (middle-left and middle-right of each subplot)
    # Assuming 2x3 grid, we'll sample from approximate bar locations
    
    height, width = img_array.shape[:2]
    
    # Define approximate regions for bars in a 2x3 grid
    # Each subplot is roughly width/3 x height/2
    subplot_width = width // 3
    subplot_height = height // 2
    
    # Bar positions (approximate - bars are usually in center-left and center-right of each subplot)
    bar_width = subplot_width // 4
    bar_center_y = subplot_height // 2
    
    colors_found = []
    
    for row in range(2):
        for col in range(3):
            # Calculate subplot position
            subplot_x = col * subplot_width
            subplot_y = row * subplot_height
            
            # Sample from left bar (Control - usually blue)
            left_bar_x = subplot_x + subplot_width // 4
            left_bar_region = img_array[
                subplot_y + bar_center_y - bar_width:subplot_y + bar_center_y + bar_width,
                left_bar_x - bar_width:left_bar_x + bar_width
            ]
            
            # Sample from right bar (pp3r1 - usually orange)
            right_bar_x = subplot_x + 3 * subplot_width // 4
            right_bar_region = img_array[
                subplot_y + bar_center_y - bar_width:subplot_y + bar_center_y + bar_width,
                right_bar_x - bar_width:right_bar_x + bar_width
            ]
            
            # Get most common color in each region (excluding white/background)
            def get_dominant_color(region):
                # Reshape to list of pixels
                pixels = region.reshape(-1, 3)
                
                # Method 1: Filter out very light colors (background) and get median
                mask = np.sum(pixels, axis=1) < 700  # Exclude white/light colors
                if mask.sum() > 0:
                    colored_pixels = pixels[mask]
                    color1 = np.median(colored_pixels, axis=0).astype(int)
                else:
                    color1 = np.median(pixels, axis=0).astype(int)
                
                # Method 2: Use mode (most frequent color) after filtering
                # Filter out background colors (very light or very dark)
                brightness = np.sum(pixels, axis=1)
                mask2 = (brightness > 100) & (brightness < 700)
                if mask2.sum() > 0:
                    colored_pixels2 = pixels[mask2]
                    # Get mode by finding most common color (rounded to nearest 10 for grouping)
                    rounded = (colored_pixels2 // 10 * 10)
                    unique_colors, counts = np.unique(rounded, axis=0, return_counts=True)
                    mode_idx = np.argmax(counts)
                    color2 = unique_colors[mode_idx].astype(int) + 5  # Add 5 to center in bin
                else:
                    color2 = color1
                
                # Use average of both methods for robustness
                color = ((color1 + color2) / 2).astype(int)
                return tuple(color)
            
            left_color = get_dominant_color(left_bar_region)
            right_color = get_dominant_color(right_bar_region)
            
            colors_found.append({
                'subplot': f"Row {row+1}, Col {col+1}",
                'control_color': left_color,
                'pp3r1_color': right_color
            })
            
            print(f"\nSubplot {row+1}-{col+1}:")
            print(f"  Control (left bar): RGB{left_color} = #{left_color[0]:02x}{left_color[1]:02x}{left_color[2]:02x}")
            print(f"  pp3r1 (right bar): RGB{right_color} = #{right_color[0]:02x}{right_color[1]:02x}{right_color[2]:02x}")
    
    # Calculate average colors across all subplots
    control_colors = [c['control_color'] for c in colors_found]
    pp3r1_colors = [c['pp3r1_color'] for c in colors_found]
    
    avg_control = np.median(control_colors, axis=0).astype(int)
    avg_pp3r1 = np.median(pp3r1_colors, axis=0).astype(int)
    
    print("\n" + "=" * 80)
    print("AVERAGE COLORS (across all subplots):")
    print("=" * 80)
    print(f"Control (Blue): RGB{tuple(avg_control)} = #{avg_control[0]:02x}{avg_control[1]:02x}{avg_control[2]:02x}")
    print(f"pp3r1 (Orange): RGB{tuple(avg_pp3r1)} = #{avg_pp3r1[0]:02x}{avg_pp3r1[1]:02x}{avg_pp3r1[2]:02x}")
    print("=" * 80)
    
    # Create visualization
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        
        # Show control color
        control_color_array = np.ones((100, 100, 3), dtype=np.uint8) * avg_control
        axes[0].imshow(control_color_array)
        axes[0].set_title(f'Control Color\nRGB{tuple(avg_control)}\n#{avg_control[0]:02x}{avg_control[1]:02x}{avg_control[2]:02x}')
        axes[0].axis('off')
        
        # Show pp3r1 color
        pp3r1_color_array = np.ones((100, 100, 3), dtype=np.uint8) * avg_pp3r1
        axes[1].imshow(pp3r1_color_array)
        axes[1].set_title(f'pp3r1 Color\nRGB{tuple(avg_pp3r1)}\n#{avg_pp3r1[0]:02x}{avg_pp3r1[1]:02x}{avg_pp3r1[2]:02x}')
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = Path(image_path).parent / 'ExtractedColors.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] Color visualization saved to: {output_path}")
        plt.close()
    
    return tuple(avg_control), tuple(avg_pp3r1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract colors from bar chart image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the reference image')
    
    args = parser.parse_args()
    
    control_color, pp3r1_color = extract_bar_colors(args.image)
    
    print("\n" + "=" * 80)
    print("COLOR CODES TO USE IN CODE:")
    print("=" * 80)
    print(f"colors = ['#{control_color[0]:02x}{control_color[1]:02x}{control_color[2]:02x}', "
          f"'#{pp3r1_color[0]:02x}{pp3r1_color[1]:02x}{pp3r1_color[2]:02x}']")
    print("=" * 80)
