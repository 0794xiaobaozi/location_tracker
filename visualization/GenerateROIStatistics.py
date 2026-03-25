#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate detailed ROI time statistics from LocationOutput CSV files.

This script is config-driven and requires project YAML:
- Reads project.video_dir and functional_roi.regions from YAML
- Computes per-video ROI time (sec/pct) from X/Y against ROI polygons
- Writes ROI_Statistics_Detailed.csv and ROI_Statistics_Summary.csv
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


def read_video_meta(video_path, fps_default):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float(fps_default), None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps is None or fps <= 0:
        fps = float(fps_default)
    if frame_count is None or frame_count <= 0:
        frame_count = None
    else:
        frame_count = int(round(frame_count))
    return float(fps), frame_count


def compute_roi_status(df, polygons):
    x_coords = df["X"].to_numpy(dtype=float)
    y_coords = df["Y"].to_numpy(dtype=float)
    n_frames = len(df)
    roi_status = np.full(n_frames, -1, dtype=int)

    for i in range(n_frames):
        x = x_coords[i]
        y = y_coords[i]
        if np.isnan(x) or np.isnan(y):
            continue
        for roi_idx, poly in enumerate(polygons):
            if point_in_polygon(x, y, poly):
                roi_status[i] = roi_idx
                break
    return roi_status


def summarize_one_video(csv_path, region_names, polygons, fps_default=25.0):
    df = pd.read_csv(csv_path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"Missing X/Y columns: {csv_path}")

    stem = infer_video_stem_from_csv(csv_path)
    video_path = csv_path.with_name(stem + ".mp4")
    fps_used, frame_count_video = read_video_meta(video_path, fps_default=fps_default)
    total_frames_csv = len(df)

    roi_status = compute_roi_status(df, polygons)
    roi_frame_counts = {name: 0 for name in region_names}
    for status in roi_status:
        if status >= 0:
            roi_frame_counts[region_names[status]] += 1

    # Distance: prefer Distance_cm/Distance_mm columns from LocationOutput.
    # In ezTrack outputs this is frame-to-frame displacement; summing gives total path length.
    distance_mm = np.nan
    if "Distance_mm" in df.columns:
        distance_mm = float(pd.to_numeric(df["Distance_mm"], errors="coerce").fillna(0).sum())
    elif "Distance_cm" in df.columns:
        distance_mm = float(pd.to_numeric(df["Distance_cm"], errors="coerce").fillna(0).sum() * 10.0)
    elif "Distance_px" in df.columns:
        distance_mm = np.nan  # no physical scale available here

    row = {
        "Video": stem,
        "n_frames_csv": int(total_frames_csv),
        "n_frames_video": frame_count_video if frame_count_video is not None else np.nan,
        "fps": float(fps_used),
        "Duration_sec": float(total_frames_csv / fps_used),
        "Distance_mm": distance_mm,
    }

    total_roi_sec = 0.0
    total_roi_pct = 0.0
    for name in region_names:
        frames_in_roi = int(roi_frame_counts[name])
        sec = float(frames_in_roi / fps_used)
        pct = float((frames_in_roi / total_frames_csv) * 100.0) if total_frames_csv > 0 else 0.0
        row[f"{name}_frames"] = frames_in_roi
        row[f"{name}_sec"] = sec
        row[f"{name}_pct"] = pct
        total_roi_sec += sec
        total_roi_pct += pct

    row["Total_ROI_sec"] = total_roi_sec
    row["Total_ROI_pct"] = total_roi_pct
    row["Outside_ROI_sec"] = max(0.0, row["Duration_sec"] - total_roi_sec)
    row["Outside_ROI_pct"] = max(0.0, 100.0 - total_roi_pct)

    return row


def build_summary(detailed_df, region_names):
    rows = []
    for name in region_names:
        sec_col = f"{name}_sec"
        pct_col = f"{name}_pct"
        rows.append(
            {
                "ROI": name,
                "Mean_sec": detailed_df[sec_col].mean(),
                "Std_sec": detailed_df[sec_col].std(),
                "Min_sec": detailed_df[sec_col].min(),
                "Max_sec": detailed_df[sec_col].max(),
                "Mean_pct": detailed_df[pct_col].mean(),
                "Std_pct": detailed_df[pct_col].std(),
                "Min_pct": detailed_df[pct_col].min(),
                "Max_pct": detailed_df[pct_col].max(),
            }
        )
    return pd.DataFrame(rows)


def generate_roi_statistics(config_path, output_detailed=None, output_summary=None, fps_default=25.0):
    video_dir, region_names, polygons = load_config(config_path)
    if not video_dir.exists():
        raise FileNotFoundError(f"Config video_dir does not exist: {video_dir}")

    csv_files = sorted(video_dir.glob("*_LocationOutput.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_LocationOutput.csv found in: {video_dir}")

    print("=" * 90)
    print("ROI TIME STATISTICS")
    print("=" * 90)
    print(f"Config: {config_path}")
    print(f"Directory: {video_dir}")
    print(f"Regions: {', '.join(region_names)}")
    print(f"CSV files: {len(csv_files)}")
    print("=" * 90)

    rows = []
    for csv_path in csv_files:
        row = summarize_one_video(
            csv_path=csv_path,
            region_names=region_names,
            polygons=polygons,
            fps_default=fps_default,
        )
        rows.append(row)
        print(
            f"[OK] {csv_path.name} -> duration={row['Duration_sec']:.2f}s "
            f"total_roi={row['Total_ROI_sec']:.2f}s outside={row['Outside_ROI_sec']:.2f}s"
        )

    detailed_df = pd.DataFrame(rows).round(4)
    summary_df = build_summary(detailed_df, region_names).round(4)

    detailed_path = Path(output_detailed) if output_detailed else (video_dir / "ROI_Statistics_Detailed.csv")
    summary_path = Path(output_summary) if output_summary else (video_dir / "ROI_Statistics_Summary.csv")
    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("-" * 90)
    print(f"[SAVED] Detailed: {detailed_path}")
    print(f"[SAVED] Summary : {summary_path}")
    print("=" * 90)
    return detailed_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ROI time statistics from project YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project_tracking_config.yml",
    )
    parser.add_argument(
        "--output-detailed",
        default=None,
        help="Optional detailed CSV path (default: <video_dir>/ROI_Statistics_Detailed.csv)",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional summary CSV path (default: <video_dir>/ROI_Statistics_Summary.csv)",
    )
    parser.add_argument(
        "--fps-default",
        type=float,
        default=25.0,
        help="Fallback FPS when video metadata cannot be read.",
    )
    args = parser.parse_args()

    generate_roi_statistics(
        config_path=args.config,
        output_detailed=args.output_detailed,
        output_summary=args.output_summary,
        fps_default=float(args.fps_default),
    )
