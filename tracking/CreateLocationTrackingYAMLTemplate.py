#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a project-level YAML template for RunLocationTrackingFromYAML.py

Example:
  pixi run -e location-tracker python CreateLocationTrackingYAMLTemplate.py ^
    --output ".\\project_tracking_config.yml" ^
    --video-dir "F:\\Neuro\\ezTrack\\LocationTracking\\video\\cropped_video\\EPM_later"
"""

import os
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Create YAML template for batch tracking.")
    parser.add_argument("--output", "-o", default="project_tracking_config.yml", help="Output YAML file path")
    parser.add_argument("--video-dir", "-d", required=True, help="Video directory for this project")
    parser.add_argument("--file-type", default="mp4", help="Video extension (default: mp4)")
    args = parser.parse_args()

    template = {
        "project": {
            "video_dir": os.path.normpath(args.video_dir),
            "file_type": args.file_type,
            "start_frame": 0,
            "end_frame": None,
            "dsmpl": 1.0,
            "stretch_width": 1.0,
            "stretch_height": 1.0,
        },
        "crop": {
            "x0": 0,
            "x1": 960,
            "y0": 0,
            "y1": 576,
        },
        "analysis_roi": {
            # type: none | rectangle | polygon
            "type": "none",
            # rectangle mode
            "x1": 0,
            "y1": 0,
            "x2": 960,
            "y2": 576,
            # polygon mode (list of polygons, each polygon is [[x,y], ...])
            "vertices": [
                [[100, 100], [860, 100], [860, 500], [100, 500]]
            ],
        },
        "functional_roi": {
            # Optional. Each region needs name + vertices.
            "regions": [
                {"name": "Left", "vertices": [[100, 100], [480, 100], [480, 500], [100, 500]]},
                {"name": "Right", "vertices": [[480, 100], [860, 100], [860, 500], [480, 500]]},
            ]
        },
        "scale": {
            # Optional. If unknown, keep factor=0.
            "px_distance": 100.0,
            "true_distance": 10.0,
            "true_scale": "cm",
            "factor": 0.1,
        },
        "tracking": {
            "loc_thresh": 99.0,
            "use_window": False,
            "window_size": 100,
            "window_weight": 0.9,
            "method": "abs",
            "rmv_wire": False,
            "wire_krn": 3,
        },
        "run": {
            "parallel": True,
            "n_processes": None,
            "accept_p_frames": False,
            "bin_dict": None,
        },
    }

    out_path = os.path.normpath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(template, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[OK] Template created: {out_path}")
    print("[TIP] Edit this YAML, then run:")
    print(f'      pixi run -e location-tracker python RunLocationTrackingFromYAML.py --config "{out_path}"')


if __name__ == "__main__":
    main()

