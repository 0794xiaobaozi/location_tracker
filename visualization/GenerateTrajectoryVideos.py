#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch generate trajectory videos for all analyzed videos

Creates a video for each analyzed video showing:
- Original video frames (cropped)
- Real-time tracking point overlay
- Trajectory trail (optional)

Output: All videos saved to a subfolder 'trajectory_videos/'
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_trajectory_video(video_path, csv_path, output_path, 
                              crop=None, marker_size=8, trail_length=50,
                              fps=None, show_trail=True):
    """Generate trajectory video for one video
    
    Args:
        video_path: Path to source video
        csv_path: Path to LocationOutput CSV
        output_path: Path to output video
        crop: tuple (x0, x1, y0, y1) or None
        marker_size: Size of tracking marker
        trail_length: Number of frames to show trail (0 to disable)
        fps: Output FPS (None = use source FPS)
        show_trail: Whether to show trajectory trail
    """
    print(f"  Processing: {os.path.basename(video_path)}")
    
    # Load location data
    try:
        location_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    [ERROR] Cannot read CSV: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Apply crop dimensions
    if crop is not None:
        x0, x1, y0, y1 = crop
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        out_width = x1 - x0
        out_height = y1 - y0
    else:
        x0, y0 = 0, 0
        out_width, out_height = width, height
    
    # Setup output video
    output_fps = fps if fps else src_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
    
    if not out.isOpened():
        print(f"    [ERROR] Cannot create output video")
        cap.release()
        return False
    
    # Extract coordinates
    x_coords = location_df['X'].values if 'X' in location_df.columns else None
    y_coords = location_df['Y'].values if 'Y' in location_df.columns else None
    
    if x_coords is None or y_coords is None:
        print(f"    [ERROR] CSV missing X or Y columns")
        cap.release()
        out.release()
        return False
    
    # Trail history
    trail_points = []
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=min(total_frames, len(x_coords)), 
                desc="    Frames", leave=False, ncols=60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= len(x_coords):
            break
        
        # Apply crop
        if crop is not None:
            frame = frame[y0:y1, x0:x1]
        
        # Get current position
        x, y = x_coords[frame_idx], y_coords[frame_idx]
        
        # Draw trail
        if show_trail and not np.isnan(x) and not np.isnan(y):
            trail_points.append((int(x), int(y)))
            if len(trail_points) > trail_length:
                trail_points.pop(0)
            
            # Draw trail with fading effect
            for i, pt in enumerate(trail_points[:-1]):
                alpha = (i + 1) / len(trail_points)
                color = (0, int(255 * alpha), int(255 * alpha))  # Yellow fading
                thickness = max(1, int(2 * alpha))
                cv2.line(frame, pt, trail_points[i + 1], color, thickness)
        
        # Draw current position marker
        if not np.isnan(x) and not np.isnan(y):
            center = (int(x), int(y))
            # Outer circle (white border)
            cv2.circle(frame, center, marker_size + 2, (255, 255, 255), 2)
            # Inner circle (red fill)
            cv2.circle(frame, center, marker_size, (0, 0, 255), -1)
            # Cross marker
            cv2.line(frame, (center[0] - marker_size - 3, center[1]), 
                    (center[0] + marker_size + 3, center[1]), (255, 255, 255), 1)
            cv2.line(frame, (center[0], center[1] - marker_size - 3), 
                    (center[0], center[1] + marker_size + 3), (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{len(x_coords)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{len(x_coords)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        if not np.isnan(x) and not np.isnan(y):
            cv2.putText(frame, f"Pos: ({int(x)}, {int(y)})", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Pos: ({int(x)}, {int(y)})", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Write frame
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"    [SAVED] {os.path.basename(output_path)} ({frame_idx} frames)")
    return True


def batch_generate_trajectory_videos(video_dir, crop=None, marker_size=8, 
                                     trail_length=50, fps=None, show_trail=True):
    """Generate trajectory videos for all videos in directory"""
    video_dir = Path(video_dir)
    
    # Create output subfolder
    output_dir = video_dir / 'trajectory_videos'
    output_dir.mkdir(exist_ok=True)
    print(f"Output folder: {output_dir}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
    
    if not video_files:
        print(f"[ERROR] No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    if crop:
        print(f"Crop region: ({crop[0]}, {crop[2]}) to ({crop[1]}, {crop[3]})")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"[{i}/{len(video_files)}] {video_path.name}")
        
        # Find corresponding CSV
        csv_path = video_path.with_name(video_path.stem + '_LocationOutput.csv')
        
        if not csv_path.exists():
            print(f"  [SKIP] No CSV found")
            fail_count += 1
            continue
        
        # Output path (in subfolder)
        output_path = output_dir / f"{video_path.stem}_tracked.mp4"
        
        # Generate
        if generate_trajectory_video(str(video_path), str(csv_path), 
                                    str(output_path), crop, marker_size,
                                    trail_length, fps, show_trail):
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 70)
    print(f"COMPLETE!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(video_files)}")
    print(f"  Output folder: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch generate trajectory videos')
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing videos and CSV files')
    parser.add_argument('--crop', type=str, default=None,
                       help='Crop region as "x0,x1,y0,y1" (e.g., "128,954,0,604")')
    parser.add_argument('--marker-size', type=int, default=8,
                       help='Size of tracking marker (default: 8)')
    parser.add_argument('--trail-length', type=int, default=50,
                       help='Number of frames to show trail (default: 50, 0 to disable)')
    parser.add_argument('--fps', type=float, default=None,
                       help='Output FPS (default: same as source)')
    parser.add_argument('--no-trail', action='store_true',
                       help='Disable trajectory trail')
    
    args = parser.parse_args()
    
    # Parse crop argument
    crop = None
    if args.crop:
        try:
            crop = tuple(map(int, args.crop.split(',')))
            if len(crop) != 4:
                print(f"[WARNING] Crop must have 4 values (x0,x1,y0,y1), ignoring")
                crop = None
        except:
            print(f"[WARNING] Cannot parse crop, ignoring")
    
    # Run batch generation
    batch_generate_trajectory_videos(
        args.directory, 
        crop=crop,
        marker_size=args.marker_size,
        trail_length=args.trail_length,
        fps=args.fps,
        show_trail=not args.no_trail
    )

