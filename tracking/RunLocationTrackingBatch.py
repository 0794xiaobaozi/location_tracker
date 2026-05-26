#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line batch location tracking without notebooks.

Example:
    pixi run -e location-tracker python RunLocationTrackingBatch.py ^
      --directory "F:\\Neuro\\ezTrack\\LocationTracking\\video\\cropped_video\\EPM_later" ^
      --ftype mp4 --loc-thresh 99.0 --parallel
"""

import os
import argparse

import LocationTracking_Functions as lt


def build_video_dict(args):
    # Minimal defaults aligned with notebook usage.
    return {
        "dpath": os.path.normpath(args.directory),
        "file": "",
        "ftype": args.ftype,
        "start": args.start,
        "end": args.end,
        "dsmpl": args.dsmpl,
        "stretch": {"width": 1.0, "height": 1.0},
        "region_names": None,  # no functional ROI in CLI baseline
        "roi_stream": None,
        "crop": None,
        "mask": {"mask": None, "stream": None},
        "scale": None,
        "analysis_roi": None,  # track full frame by default
    }


def build_tracking_params(args):
    return {
        "loc_thresh": args.loc_thresh,
        "use_window": args.use_window,
        "window_size": args.window_size,
        "window_weight": args.window_weight,
        "method": args.method,
        "rmv_wire": args.rmv_wire,
        "wire_krn": args.wire_krn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch location tracking from CLI (no notebook required)."
    )
    parser.add_argument("--directory", "-d", required=True, help="Input video directory")
    parser.add_argument("--ftype", default="mp4", help="Video extension (default: mp4)")

    parser.add_argument("--start", type=int, default=0, help="Start frame (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End frame (default: full video)")
    parser.add_argument("--dsmpl", type=float, default=1.0, help="Downsample ratio, 0-1 (default: 1.0)")

    parser.add_argument("--loc-thresh", type=float, default=99.0, help="Location threshold percentile")
    parser.add_argument("--method", choices=["abs", "light", "dark"], default="abs", help="Diff method")
    parser.add_argument("--use-window", action="store_true", help="Enable prior-location window weighting")
    parser.add_argument("--window-size", type=int, default=100, help="Window size when --use-window")
    parser.add_argument("--window-weight", type=float, default=0.9, help="Window weight when --use-window")
    parser.add_argument("--rmv-wire", action="store_true", help="Enable wire removal")
    parser.add_argument("--wire-krn", type=int, default=3, help="Kernel size for wire removal")

    parser.add_argument("--parallel", action="store_true", help="Use parallel batch processing")
    parser.add_argument("--n-processes", type=int, default=None, help="Parallel process count (default: auto)")
    parser.add_argument("--accept-p-frames", action="store_true", help="Allow videos with p-frames")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise FileNotFoundError(f"Directory not found: {args.directory}")

    print("=" * 70)
    print("BATCH LOCATION TRACKING (CLI)")
    print("=" * 70)
    print(f"Input directory: {os.path.normpath(args.directory)}")
    print(f"File type: .{args.ftype}")
    print(f"Parallel: {args.parallel}")
    print("=" * 70)

    video_dict = build_video_dict(args)
    tracking_params = build_tracking_params(args)
    bin_dict = None  # summary bins optional; baseline keeps None

    video_dict = lt.Batch_LoadFiles(video_dict)
    n_files = len(video_dict.get("FileNames", []))
    if n_files == 0:
        raise RuntimeError(f"No .{args.ftype} videos found in {args.directory}")
    print(f"Found {n_files} videos.")

    if args.parallel:
        summary, _layout = lt.Batch_Process_Parallel(
            video_dict,
            tracking_params,
            bin_dict,
            n_processes=args.n_processes,
            accept_p_frames=args.accept_p_frames,
        )
    else:
        summary, _layout = lt.Batch_Process(
            video_dict,
            tracking_params,
            bin_dict,
            accept_p_frames=args.accept_p_frames,
        )

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Rows in summary: {len(summary)}")
    print(f"Summary file: {os.path.join(os.path.normpath(args.directory), 'BatchSummary.csv')}")


if __name__ == "__main__":
    main()

