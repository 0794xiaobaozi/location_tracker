#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ROI entry statistics from LocationOutput CSV files.

This script is config-driven and requires project YAML:
- Reads project.video_dir and functional_roi.regions from YAML
- Counts ROI entries/exits from frame-wise X/Y
- Writes ROI_Entry_Statistics.csv
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")

    project = cfg.get("project", {}) or {}
    video_dir = project.get("video_dir")
    if not video_dir:
        raise KeyError("Missing required key: project.video_dir")
    video_dir = Path(video_dir)

    functional_cfg = cfg.get("functional_roi", {}) or {}
    regions = functional_cfg.get("regions", []) or []
    if not regions:
        raise KeyError("Missing required key: functional_roi.regions")

    region_names = []
    polygons = []
    for i, region in enumerate(regions):
        name = str(region.get("name", f"ROI_{i+1}"))
        vertices = region.get("vertices", []) or []
        if len(vertices) < 3:
            raise ValueError(f"Region '{name}' has fewer than 3 vertices.")
        poly = np.array(vertices, dtype=np.float32)
        region_names.append(name)
        polygons.append(poly)

    return video_dir, region_names, polygons


def point_in_polygon(x, y, polygon):
    if np.isnan(x) or np.isnan(y):
        return False
    result = cv2.pointPolygonTest(polygon.astype(np.float32), (float(x), float(y)), False)
    return result >= 0


def infer_video_stem_from_csv(csv_path):
    stem = csv_path.stem
    suffix = "_LocationOutput"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def count_entries_for_csv(csv_path, region_names, polygons, include_first_frame_entry):
    df = pd.read_csv(csv_path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"Missing X/Y columns: {csv_path}")

    x_coords = df["X"].to_numpy(dtype=float)
    y_coords = df["Y"].to_numpy(dtype=float)
    n_frames = len(df)

    # -1 means outside all ROIs; otherwise index in region_names/polygons
    roi_status = np.full(n_frames, -1, dtype=int)

    for i in range(n_frames):
        x = x_coords[i]
        y = y_coords[i]
        if np.isnan(x) or np.isnan(y):
            continue
        for roi_idx, poly in enumerate(polygons):
            if point_in_polygon(x, y, poly):
                roi_status[i] = roi_idx
                break  # assume non-overlapping ROIs

    entry_counts = {name: 0 for name in region_names}
    exit_counts = {name: 0 for name in region_names}
    time_frames = {name: 0 for name in region_names}

    for i in range(n_frames):
        if roi_status[i] >= 0:
            time_frames[region_names[roi_status[i]]] += 1

    for i in range(1, n_frames):
        prev_roi = roi_status[i - 1]
        curr_roi = roi_status[i]
        if curr_roi >= 0 and prev_roi != curr_roi:
            entry_counts[region_names[curr_roi]] += 1
        if prev_roi >= 0 and curr_roi != prev_roi:
            exit_counts[region_names[prev_roi]] += 1

    if include_first_frame_entry and n_frames > 0 and roi_status[0] >= 0:
        entry_counts[region_names[roi_status[0]]] += 1

    # bout durations per ROI in frames
    bout_durations = {name: [] for name in region_names}
    current_roi = None
    bout_start = None
    for i in range(n_frames):
        status = roi_status[i]
        if status >= 0:
            roi_name = region_names[status]
            if current_roi != roi_name:
                if current_roi is not None and bout_start is not None:
                    bout_durations[current_roi].append(i - bout_start)
                current_roi = roi_name
                bout_start = i
        else:
            if current_roi is not None and bout_start is not None:
                bout_durations[current_roi].append(i - bout_start)
            current_roi = None
            bout_start = None
    if current_roi is not None and bout_start is not None:
        bout_durations[current_roi].append(n_frames - bout_start)

    avg_bout_frames = {}
    for name in region_names:
        vals = bout_durations[name]
        avg_bout_frames[name] = float(np.mean(vals)) if vals else 0.0

    return {
        "n_frames": int(n_frames),
        "entries": entry_counts,
        "exits": exit_counts,
        "time_frames": time_frames,
        "avg_bout_frames": avg_bout_frames,
    }


def read_video_fps(video_path, fallback_fps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return fallback_fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return fallback_fps
    return float(fps)


def generate_entry_statistics(
    config_path,
    output_csv=None,
    include_first_frame_entry=True,
    fps_default=25.0,
):
    video_dir, region_names, polygons = load_config(config_path)
    if not video_dir.exists():
        raise FileNotFoundError(f"Config video_dir does not exist: {video_dir}")

    csv_files = sorted(video_dir.glob("*_LocationOutput.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_LocationOutput.csv found in: {video_dir}")

    rows = []
    print("=" * 90)
    print("ROI ENTRY STATISTICS")
    print("=" * 90)
    print(f"Config: {config_path}")
    print(f"Directory: {video_dir}")
    print(f"Regions: {', '.join(region_names)}")
    print(f"CSV files: {len(csv_files)}")
    print("=" * 90)

    for csv_path in csv_files:
        stem = infer_video_stem_from_csv(csv_path)
        video_path = video_dir / f"{stem}.mp4"
        fps_used = read_video_fps(video_path, fallback_fps=fps_default)

        stats = count_entries_for_csv(
            csv_path,
            region_names=region_names,
            polygons=polygons,
            include_first_frame_entry=include_first_frame_entry,
        )

        row = {
            "Video": stem,
            "n_frames": stats["n_frames"],
            "fps": fps_used,
        }

        total_entries = 0
        for name in region_names:
            row[f"{name}_entries"] = int(stats["entries"][name])
            row[f"{name}_exits"] = int(stats["exits"][name])
            row[f"{name}_avg_bout_sec"] = float(stats["avg_bout_frames"][name] / fps_used)
            total_entries += int(stats["entries"][name])

        row["Total_entries"] = total_entries
        rows.append(row)
        print(f"[OK] {csv_path.name} -> entries={total_entries}")

    out_path = Path(output_csv) if output_csv else (video_dir / "ROI_Entry_Statistics.csv")
    out_df = pd.DataFrame(rows).round(4)
    out_df.to_csv(out_path, index=False)

    print("-" * 90)
    print(f"[SAVED] {out_path}")
    print("=" * 90)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ROI entry statistics from project YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project_tracking_config.yml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path (default: <video_dir>/ROI_Entry_Statistics.csv)",
    )
    parser.add_argument(
        "--exclude-first-frame-entry",
        action="store_true",
        help="Do not count first frame as an entry when it starts inside ROI.",
    )
    parser.add_argument(
        "--fps-default",
        type=float,
        default=25.0,
        help="Fallback FPS when video metadata cannot be read.",
    )
    args = parser.parse_args()

    generate_entry_statistics(
        config_path=args.config,
        output_csv=args.output,
        include_first_frame_entry=not args.exclude_first_frame_entry,
        fps_default=float(args.fps_default),
    )
