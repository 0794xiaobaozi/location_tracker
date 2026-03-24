#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate trajectory visualization images for all analyzed videos

Creates an image for each video showing:
- Reference frame (background)
- Animal trajectory (colored path)
- Analysis ROI (polygon outline)
- Functional ROIs (Left/Right/Top/Bottom regions)
- Start/End markers
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import yaml


def draw_polygon_roi(img, vertices, color=(0, 255, 0), thickness=2, fill_alpha=0.1):
    """Draw polygon ROI with optional fill"""
    overlay = img.copy()
    for poly in vertices:
        pts = np.array(poly, dtype=np.int32)
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], color)
        # Draw border
        cv2.polylines(img, [pts], True, color, thickness)
    
    # Blend for transparency
    cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    return img


def draw_functional_rois(img, roi_data, region_names, colors=None):
    """Draw functional ROIs (Left/Right/Top/Bottom)"""
    if colors is None:
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    xs = roi_data['xs']
    ys = roi_data['ys']
    
    for i, (x_coords, y_coords, name) in enumerate(zip(xs, ys, region_names)):
        if i >= len(colors):
            break
        
        pts = np.array([(x, y) for x, y in zip(x_coords, y_coords)], dtype=np.int32)
        
        # Draw with transparency
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], colors[i])
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        
        # Draw border
        cv2.polylines(img, [pts], True, colors[i], 2)
        
        # Add label
        if len(pts) > 0:
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(img, name, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
    
    return img


def draw_trajectory(img, location_df, line_color=(0, 255, 255), 
                   start_color=(0, 255, 0), end_color=(0, 0, 255),
                   line_thickness=2, marker_size=8):
    """Draw trajectory on image"""
    # Extract X, Y coordinates
    x_coords = location_df['X'].values
    y_coords = location_df['Y'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    if len(x_coords) < 2:
        print(f"  [WARNING] Not enough valid points to draw trajectory")
        return img
    
    # Convert to integer coordinates
    points = np.column_stack((x_coords, y_coords)).astype(np.int32)
    
    # Draw trajectory lines with color gradient (optional: could add heatmap)
    for i in range(len(points) - 1):
        cv2.line(img, tuple(points[i]), tuple(points[i+1]), line_color, line_thickness)
    
    # Draw start marker (green circle)
    cv2.circle(img, tuple(points[0]), marker_size, start_color, -1)
    cv2.circle(img, tuple(points[0]), marker_size + 2, (255, 255, 255), 2)
    cv2.putText(img, "START", (points[0][0] + 15, points[0][1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, start_color, 2)
    
    # Draw end marker (red circle)
    cv2.circle(img, tuple(points[-1]), marker_size, end_color, -1)
    cv2.circle(img, tuple(points[-1]), marker_size + 2, (255, 255, 255), 2)
    cv2.putText(img, "END", (points[-1][0] + 15, points[-1][1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, end_color, 2)
    
    return img


def generate_trajectory_image(video_path, csv_path, output_path, 
                              analysis_roi=None, functional_rois=None, 
                              region_names=None, crop=None):
    """Generate trajectory visualization for one video
    
    Args:
        crop: tuple (x0, x1, y0, y1) crop region, or None for no crop
    """
    print(f"Processing: {os.path.basename(video_path)}")
    
    # Load video to get reference frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return False
    
    # Read first frame as reference
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  [ERROR] Cannot read frame from video")
        return False
    
    # Convert to color (if grayscale)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Apply crop if specified
    if crop is not None:
        x0, x1, y0, y1 = crop
        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        frame = frame[y0:y1, x0:x1]
        print(f"  [CROP] Applied crop: ({x0}, {y0}) to ({x1}, {y1}), new size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Load location data
    try:
        location_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read CSV: {e}")
        return False
    
    # Create visualization
    vis_img = frame.copy()
    
    # 1. Draw Analysis ROI (green)
    if analysis_roi is not None:
        if isinstance(analysis_roi, dict) and analysis_roi.get('type') == 'polygon':
            vis_img = draw_polygon_roi(vis_img, analysis_roi['vertices'], 
                                      color=(0, 255, 0), thickness=3, fill_alpha=0.1)
        elif isinstance(analysis_roi, tuple) and len(analysis_roi) == 4:
            # Rectangle
            x1, y1, x2, y2 = analysis_roi
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # 2. Draw Functional ROIs
    if functional_rois is not None and region_names is not None:
        vis_img = draw_functional_rois(vis_img, functional_rois, region_names)
    
    # 3. Draw trajectory
    vis_img = draw_trajectory(vis_img, location_df, 
                             line_color=(0, 255, 255), 
                             line_thickness=2, 
                             marker_size=8)
    
    # 4. Add info text
    total_frames = len(location_df)
    valid_frames = location_df['X'].notna().sum()
    total_distance = location_df['Distance_px'].sum()
    
    info_text = [
        f"Video: {os.path.basename(video_path)}",
        f"Frames: {valid_frames}/{total_frames}",
        f"Distance: {total_distance:.1f} px"
    ]
    
    y_offset = 30
    for text in info_text:
        cv2.putText(vis_img, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y_offset += 30
    
    # Save image
    cv2.imwrite(output_path, vis_img)
    print(f"  [SAVED] {output_path}")
    
    return True


def batch_generate_trajectories(video_dir, analysis_roi=None, 
                                functional_rois=None, region_names=None,
                                crop=None):
    """Generate trajectory images for all videos in directory"""
    video_dir = Path(video_dir)
    
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
    
    for video_path in sorted(video_files):
        # Find corresponding CSV
        csv_path = video_path.with_name(video_path.stem + '_LocationOutput.csv')
        
        if not csv_path.exists():
            print(f"[SKIP] {video_path.name} - No CSV found")
            fail_count += 1
            continue
        
        # Output path
        output_path = video_path.with_name(video_path.stem + '_Trajectory.png')
        
        # Generate
        if generate_trajectory_image(str(video_path), str(csv_path), 
                                    str(output_path), analysis_roi, 
                                    functional_rois, region_names, crop):
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 70)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {len(video_files)}")


def _parse_crop_text(crop_text):
    if not crop_text:
        return None
    try:
        crop = tuple(map(int, crop_text.split(',')))
        if len(crop) != 4:
            print("[WARNING] Crop must have 4 values (x0,x1,y0,y1), ignoring")
            return None
        return crop
    except Exception:
        print("[WARNING] Cannot parse crop, ignoring")
        return None


def _load_visualization_config(config_path, skip_config_crop=False):
    """Load directory/ROI/crop settings from project YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")

    project = cfg.get("project", {}) or {}
    directory = project.get("video_dir")
    if not directory:
        raise KeyError("Config missing required key: project.video_dir")

    analysis_roi = cfg.get("analysis_roi")

    # Convert functional_roi.regions -> {'xs': [...], 'ys': [...]}
    functional_rois = None
    region_names = None
    functional_cfg = cfg.get("functional_roi", {}) or {}
    regions = functional_cfg.get("regions", []) or []
    if regions:
        xs, ys, names = [], [], []
        for region in regions:
            vertices = region.get("vertices", []) or []
            if not vertices:
                continue
            xs.append([float(p[0]) for p in vertices])
            ys.append([float(p[1]) for p in vertices])
            names.append(str(region.get("name", f"ROI_{len(names)+1}")))
        if xs and ys and names:
            functional_rois = {"xs": xs, "ys": ys}
            region_names = names

    # Crop policy for config:
    # - Apply config crop by default (most users expect YAML to be the source of truth).
    # - Allow explicit opt-out with --skip-config-crop.
    crop = None
    crop_cfg = cfg.get("crop")
    if crop_cfg and all(k in crop_cfg for k in ("x0", "x1", "y0", "y1")):
        cfg_crop = (
            int(crop_cfg["x0"]),
            int(crop_cfg["x1"]),
            int(crop_cfg["y0"]),
            int(crop_cfg["y1"]),
        )
        normalized_dir = str(directory).replace("/", "\\").lower()
        points_to_cropped_videos = "\\cropped_video\\" in normalized_dir
        if skip_config_crop:
            print("[INFO] Skipping config crop because --skip-config-crop is set.")
        else:
            crop = cfg_crop
            if points_to_cropped_videos:
                print("[INFO] Config video_dir is inside 'cropped_video'; applying config crop as requested by YAML.")
                print("[INFO] Use --skip-config-crop if you want full-frame background without this crop.")

    return directory, analysis_roi, functional_rois, region_names, crop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate trajectory visualization images')
    parser.add_argument('--directory', type=str, default=None,
                       help='Directory containing videos and CSV files')
    parser.add_argument('--config', type=str, default=None,
                       help='Project YAML config path (auto-loads directory/ROI/crop)')
    parser.add_argument('--skip-config-crop', action='store_true',
                       help='Ignore crop from --config and keep full-frame background')
    parser.add_argument('--crop', type=str, default=None,
                       help='Crop region as "x0,x1,y0,y1" (e.g., "128,954,0,604")')
    parser.add_argument('--analysis-roi', type=str, default=None,
                       help='Analysis ROI as Python dict string')
    parser.add_argument('--functional-rois', type=str, default=None,
                       help='Functional ROIs as Python dict string')
    parser.add_argument('--region-names', type=str, default='Left,Right,Top,Bottom',
                       help='Comma-separated region names')
    
    args = parser.parse_args()

    # Defaults from CLI args
    directory = args.directory
    crop = _parse_crop_text(args.crop)
    analysis_roi = None
    functional_rois = None
    region_names = args.region_names.split(',') if args.region_names else None

    # Parse ROI arguments if provided by CLI
    if args.analysis_roi:
        try:
            analysis_roi = eval(args.analysis_roi)
        except Exception:
            print("[WARNING] Cannot parse analysis_roi, ignoring")

    if args.functional_rois:
        try:
            functional_rois = eval(args.functional_rois)
        except Exception:
            print("[WARNING] Cannot parse functional_rois, ignoring")

    # If config is provided, auto-load settings as defaults.
    if args.config:
        try:
            cfg_dir, cfg_analysis, cfg_functional, cfg_region_names, cfg_crop = _load_visualization_config(
                args.config, skip_config_crop=args.skip_config_crop
            )
            directory = directory or cfg_dir
            if analysis_roi is None:
                analysis_roi = cfg_analysis
            if functional_rois is None:
                functional_rois = cfg_functional
            if (region_names is None or len(region_names) == 0) and cfg_region_names is not None:
                region_names = cfg_region_names
            if crop is None:
                crop = cfg_crop
            print(f"[INFO] Loaded visualization defaults from config: {args.config}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            sys.exit(1)

    if not directory:
        print("[ERROR] You must provide --directory or --config with project.video_dir")
        sys.exit(1)

    # Run batch generation
    batch_generate_trajectories(directory, analysis_roi,
                                functional_rois, region_names, crop)

