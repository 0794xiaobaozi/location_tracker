#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Open the freeze GUI or run ezTrack freeze analysis from a project YAML file."""

import argparse
import os

import cv2
import pandas as pd
import yaml

from FreezeAnalysis_Functions import (
    Measure_Freezing,
    Measure_Motion,
    SaveData,
    Summarize,
)
from AutoFreezeCalibration import auto_calibrate_motion_cutoff


class CropBox:
    """Small adapter for ezTrack's cropframe(), which expects a stream-like object."""

    def __init__(self, crop):
        self.data = {
            "x0": [crop["x0"]],
            "x1": [crop["x1"]],
            "y0": [crop["y0"]],
            "y1": [crop["y1"]],
        }


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 25.0


def make_crop(crop_config):
    if not crop_config or not crop_config.get("enabled", False):
        return None
    return CropBox(crop_config)


def make_video_dict(video_dir, file_name, config):
    project = config["project"]
    video_path = os.path.join(video_dir, file_name)
    return {
        "dpath": video_dir,
        "file": file_name,
        "fpath": video_path,
        "fps": get_video_fps(video_path),
        "start": int(project.get("start_frame", 0)),
        "end": project.get("end_frame"),
        "dsmpl": float(project.get("dsmpl", 1.0)),
        "stretch": {"width": 1.0, "height": 1.0},
        "ftype": project.get("file_type", "mp4"),
        "FileNames": [],
        "cal_frms": int(config.get("calibration", {}).get("cal_frames", 250)),
        "crop": make_crop(config.get("crop")),
    }


def list_video_files(video_dir, file_type, calibration_video=None):
    suffix = "." + file_type.lower().lstrip(".")
    calibration_abs = os.path.abspath(calibration_video) if calibration_video else None
    files = []
    for name in sorted(os.listdir(video_dir)):
        path = os.path.join(video_dir, name)
        if not os.path.isfile(path) or not name.lower().endswith(suffix):
            continue
        if calibration_abs and os.path.abspath(path) == calibration_abs:
            continue
        files.append(name)
    return files


def convert_bins_to_frames(bins, fps):
    if bins is None:
        return None
    return {
        name: (int(rng[0] * fps), int(rng[1] * fps))
        for name, rng in bins.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Open the freeze GUI or run ezTrack freeze analysis from YAML.")
    parser.add_argument("--config", "-c", help="Freeze analysis YAML file")
    parser.add_argument("--calibrate-only", action="store_true", help="Only run robust MAD calibration")
    args = parser.parse_args()

    if not args.config:
        from BuildFreezeConfigGUI import FreezeConfigBuilderApp

        FreezeConfigBuilderApp().run()
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project = config["project"]
    calibration = config.get("calibration", {})
    freeze = config.get("freeze", {})
    run = config.get("run", {})

    video_dir = os.path.normpath(project["video_dir"])
    file_type = project.get("file_type", "mp4")
    calibration_video = os.path.normpath(calibration.get("video_path") or "")
    motion_cutoff = calibration.get("motion_cutoff")

    if args.calibrate_only:
        if not calibration_video:
            raise ValueError("Set calibration.video_path in the YAML.")
        stats = auto_calibrate_motion_cutoff(
            calibration_video,
            start_frame=int(project.get("start_frame", 0)),
            dsmpl=float(project.get("dsmpl", 1.0)),
            crop=config.get("crop"),
            cal_frames=int(calibration.get("cal_frames", 250)),
            cal_pixels=int(calibration.get("cal_pixels", 10000)),
            sigma=float(calibration.get("sigma", 1.0)),
            percentile=float(calibration.get("percentile", 99.99)),
            cutoff_multiplier=float(calibration.get("cutoff_multiplier", 2.8)),
            accept_p_frames=bool(run.get("accept_p_frames", False)),
        )
        print("[OK] Percentile multiplier calibration complete.")
        print(f"percentile: {stats['percentile']}")
        print(f"percentile_value: {stats['percentile_value']}")
        print(f"cutoff_multiplier: {stats['cutoff_multiplier']}")
        print(f"zero_fraction: {stats['zero_fraction']}")
        print(f"motion_cutoff: {stats['motion_cutoff']}")
        return

    if motion_cutoff is None:
        raise ValueError(
            "Set calibration.motion_cutoff before running batch analysis. "
            "Use --calibrate-only first if you need ezTrack's suggested cutoff."
        )

    file_names = list_video_files(video_dir, file_type, calibration_video=calibration_video)
    if not file_names:
        raise FileNotFoundError(f"No .{file_type} videos found in {video_dir}")

    summaries = []
    freeze_threshold = float(freeze.get("freeze_threshold", 50))

    for file_name in file_names:
        video_dict = make_video_dict(video_dir, file_name, config)
        min_duration_frames = int(round(float(freeze.get("min_duration_seconds", 0.5)) * video_dict["fps"]))

        motion = Measure_Motion(video_dict, motion_cutoff, SIGMA=float(calibration.get("sigma", 1.0)))
        freezing = Measure_Freezing(motion, freeze_threshold, MinDuration=min_duration_frames)

        if run.get("save_frame_data", True):
            SaveData(video_dict, motion, freezing, motion_cutoff, freeze_threshold, min_duration_frames)

        bins = convert_bins_to_frames(config.get("summary", {}).get("bins"), video_dict["fps"])
        summaries.append(
            Summarize(
                video_dict,
                motion,
                freezing,
                freeze_threshold,
                min_duration_frames,
                motion_cutoff,
                bin_dict=bins,
            )
        )

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_path = os.path.join(video_dir, "FreezeBatchSummary.csv")
    summary_all.to_csv(summary_path, index=False)
    print(f"[OK] Batch summary saved: {summary_path}")


if __name__ == "__main__":
    main()
