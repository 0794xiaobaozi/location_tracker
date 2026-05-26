#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a YAML template for freeze analysis."""

import argparse
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description="Create YAML template for freeze analysis.")
    parser.add_argument("--output", "-o", default="project_freeze_config.yml", help="Output YAML file path")
    parser.add_argument("--video-dir", "-d", required=True, help="Behavior video directory")
    parser.add_argument("--calibration-video", help="Empty-chamber calibration video path")
    parser.add_argument("--file-type", default="mp4", help="Video extension (default: mp4)")
    args = parser.parse_args()

    template = {
        "project": {
            "video_dir": os.path.normpath(args.video_dir),
            "file_type": args.file_type,
            "start_frame": 0,
            "end_frame": None,
            "dsmpl": 1.0,
        },
        "crop": {
            "enabled": False,
            "x0": 0,
            "x1": 960,
            "y0": 0,
            "y1": 576,
        },
        "calibration": {
            "video_path": os.path.normpath(args.calibration_video) if args.calibration_video else "",
            "cal_frames": 250,
            "cal_pixels": 10000,
            "sigma": 1.0,
            "percentile": 99.99,
            "cutoff_multiplier": 2.8,
            "motion_cutoff": None,
        },
        "freeze": {
            "freeze_threshold": 50,
            "min_duration_seconds": 0.5,
        },
        "summary": {
            "bins": None,
        },
        "run": {
            "accept_p_frames": False,
            "save_frame_data": True,
        },
    }

    out_path = os.path.normpath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(template, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[OK] Template created: {out_path}")
    print("[TIP] Edit thresholds, then run:")
    print(f'      pixi run -e location-tracker python RunFreezeAnalysisFromYAML.py --config "{out_path}"')


if __name__ == "__main__":
    main()
