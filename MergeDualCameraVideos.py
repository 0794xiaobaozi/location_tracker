#!/usr/bin/env python3
"""
Batch-merge dual-camera clips from a JSON plan written by MergeDualCameraVideosGUI.py.

Reads `dual_camera_merge_jobs.json` (or `--jobs`) and runs ffmpeg for each clip.
Requires: ffmpeg in PATH; pixi env `location-tracker` for OpenCV (probe).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dual_camera_merge import default_jobs_path, merge_all_from_jobs


def main() -> None:
    p = argparse.ArgumentParser(description="Merge dual-camera MP4s using a saved jobs JSON plan.")
    p.add_argument(
        "--jobs",
        type=Path,
        default=None,
        help="Path to dual_camera_merge_jobs.json (default: next to this script)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned merges without calling ffmpeg.")
    p.add_argument("--clip", type=str, default=None, help="Only process this filename (e.g. 1-1.mp4).")
    args = p.parse_args()

    jobs_path = args.jobs if args.jobs is not None else default_jobs_path(Path(__file__).resolve().parent)
    if not jobs_path.is_file():
        print(f"Jobs file not found: {jobs_path}", file=sys.stderr)
        print("Create it with MergeDualCameraVideosGUI.py (保存合并配置).", file=sys.stderr)
        sys.exit(1)

    code = merge_all_from_jobs(
        jobs_path,
        only_filename=args.clip,
        dry_run=args.dry_run,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
