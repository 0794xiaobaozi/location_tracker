#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate tracking overlay videos using project YAML config.

Design rule:
- Both single-video and batch modes read the same project config YAML.
- Crop behavior is taken from YAML by default.
"""

import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import pandas as pd
import yaml


def load_project_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")

    project = cfg.get("project", {}) or {}
    video_dir = project.get("video_dir")
    if not video_dir:
        raise KeyError("Missing required key: project.video_dir")
    video_dir = os.path.normpath(video_dir)

    crop_cfg = cfg.get("crop", {}) or {}
    crop = None
    if all(k in crop_cfg for k in ("x0", "x1", "y0", "y1")):
        crop = (
            int(crop_cfg["x0"]),
            int(crop_cfg["x1"]),
            int(crop_cfg["y0"]),
            int(crop_cfg["y1"]),
        )

    return video_dir, crop


def normalize_crop(crop, frame_w, frame_h):
    if crop is None:
        return None
    x0, x1, y0, y1 = crop
    x0 = max(0, min(frame_w, x0))
    x1 = max(0, min(frame_w, x1))
    y0 = max(0, min(frame_h, y0))
    y1 = max(0, min(frame_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, x1, y0, y1)


def infer_csv_path(video_path):
    p = Path(video_path)
    return str(p.with_name(p.stem + "_LocationOutput.csv"))


def infer_output_path(video_path, output_path=None):
    if output_path:
        return output_path
    p = Path(video_path)
    return str(p.with_name(p.stem + "_TrackingOverlay.mp4"))


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x0, x1, y0, y1 = crop
    return frame[y0:y1, x0:x1]


def generate_overlay_video(
    video_path,
    csv_path,
    output_path,
    crop=None,
    trail_length=40,
    marker_radius=6,
    show_text=False,
):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    h0, w0 = frame0.shape[:2]
    crop_norm = normalize_crop(crop, w0, h0)
    frame0 = apply_crop(frame0, crop_norm)
    out_h, out_w = frame0.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trail = deque(maxlen=max(1, int(trail_length)))
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = apply_crop(frame, crop_norm)

        if i < len(df):
            x = df.at[i, "X"]
            y = df.at[i, "Y"]
            if pd.notna(x) and pd.notna(y):
                # IMPORTANT: use raw CSV coordinates directly (no extra shift)
                px = int(round(float(x)))
                py = int(round(float(y)))
                if 0 <= px < out_w and 0 <= py < out_h:
                    trail.append((px, py))

        if len(trail) > 1:
            pts = list(trail)
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                color = (0, int(255 * alpha), 255)
                cv2.line(frame, pts[j - 1], pts[j], color, 2)

        if trail:
            cv2.circle(frame, trail[-1], marker_radius, (0, 255, 255), -1)
            cv2.circle(frame, trail[-1], marker_radius + 3, (0, 0, 0), 2)

        if show_text:
            cv2.putText(
                frame,
                f"Frame: {i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    return i, (out_w, out_h), crop_norm


def resolve_video_path(video_arg, config_video_dir):
    if video_arg is None:
        return None
    if os.path.isabs(video_arg):
        return os.path.normpath(video_arg)
    return os.path.normpath(os.path.join(config_video_dir, video_arg))


def batch_generate(config_video_dir, crop, trail_length, marker_radius, show_text):
    d = Path(config_video_dir)
    videos = sorted([p for p in d.glob("*.mp4") if not p.name.endswith("_TrackingOverlay.mp4")])
    if not videos:
        print(f"[ERROR] No .mp4 videos found in: {config_video_dir}")
        return

    print(f"Found {len(videos)} videos in config directory")
    if crop is not None:
        print(f"Using config crop: {crop}")
    print("=" * 70)

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for video in videos:
        csv_path = infer_csv_path(str(video))
        if not os.path.isfile(csv_path):
            print(f"[SKIP] {video.name} - missing CSV")
            skip_count += 1
            continue
        out_path = infer_output_path(str(video))
        try:
            n, size, crop_norm = generate_overlay_video(
                str(video),
                csv_path,
                out_path,
                crop=crop,
                trail_length=trail_length,
                marker_radius=marker_radius,
                show_text=show_text,
            )
            crop_label = f"{crop_norm}" if crop_norm else "None"
            print(f"[OK] {video.name} -> {Path(out_path).name} | frames={n} size={size[0]}x{size[1]} crop={crop_label}")
            ok_count += 1
        except Exception as e:
            print(f"[FAIL] {video.name} - {e}")
            fail_count += 1

    print("=" * 70)
    print(f"Done. ok={ok_count} skip={skip_count} fail={fail_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tracking overlay videos (single or batch) using project YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project_tracking_config.yml (required for both single and batch).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Single video mode. Use filename (e.g. 1-5.mp4) or absolute path.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path for single mode. Default: <video_stem>_LocationOutput.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for single mode. Default: <video_stem>_TrackingOverlay.mp4",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Ignore crop from YAML config.",
    )
    parser.add_argument("--trail-length", type=int, default=40, help="Trail length in frames.")
    parser.add_argument("--marker-radius", type=int, default=6, help="Marker radius in pixels.")
    parser.add_argument("--show-text", action="store_true", help="Draw frame index text.")
    args = parser.parse_args()

    video_dir, crop = load_project_config(args.config)
    if args.no_crop:
        crop = None

    print(f"[INFO] Config loaded: {args.config}")
    print(f"[INFO] Config video_dir: {video_dir}")
    if crop is not None:
        print(f"[INFO] Config crop: {crop}")
    else:
        print("[INFO] Crop disabled (or not provided in config).")

    if args.video:
        video_path = resolve_video_path(args.video, video_dir)
        csv_path = args.csv if args.csv else infer_csv_path(video_path)
        output_path = infer_output_path(video_path, args.output)
        n, size, crop_norm = generate_overlay_video(
            video_path,
            csv_path,
            output_path,
            crop=crop,
            trail_length=args.trail_length,
            marker_radius=args.marker_radius,
            show_text=args.show_text,
        )
        crop_label = f"{crop_norm}" if crop_norm else "None"
        print(f"[OK] {output_path} | frames={n} size={size[0]}x{size[1]} crop={crop_label}")
    else:
        batch_generate(
            video_dir,
            crop,
            trail_length=args.trail_length,
            marker_radius=args.marker_radius,
            show_text=args.show_text,
        )
