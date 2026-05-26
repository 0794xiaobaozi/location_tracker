#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automatic empty-chamber calibration for freeze analysis."""

import os

import cv2
import numpy as np


def _crop_frame(frame, crop):
    if not crop or not crop.get("enabled", False):
        return frame
    x0 = int(crop.get("x0", 0))
    x1 = int(crop.get("x1", frame.shape[1]))
    y0 = int(crop.get("y0", 0))
    y1 = int(crop.get("y1", frame.shape[0]))
    x_min, x_max = sorted((max(0, x0), min(frame.shape[1], x1)))
    y_min, y_max = sorted((max(0, y0), min(frame.shape[0], y1)))
    return frame[y_min:y_max, x_min:x_max]


def _read_gray_frame(cap, dsmpl, crop):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if dsmpl < 1:
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * dsmpl), int(frame.shape[0] * dsmpl)),
            cv2.INTER_NEAREST,
        )
    return _crop_frame(frame, crop)


def check_p_frames(cap, p_prop_allowed=0.01, frames_checked=300):
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frames_checked <= 0:
        return

    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    p_frames = 0
    for _ in range(frames_checked):
        ret, _ = cap.read()
        if ret is False:
            p_frames += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if p_frames > int(frames_checked * p_prop_allowed):
        raise RuntimeError(
            "Video compression method may not be supported. "
            f"Approximately {(p_frames / frames_checked) * 100:.2f}% frames are blank."
        )


def auto_calibrate_motion_cutoff(
    video_path,
    start_frame=0,
    dsmpl=1.0,
    crop=None,
    cal_frames=250,
    cal_pixels=10000,
    sigma=1.0,
    percentile=99.99,
    cutoff_multiplier=2.8,
    accept_p_frames=False,
    random_seed=0,
):
    """Return a percentile-based motion cutoff from an empty-chamber video.

    The cutoff is cutoff_multiplier * percentile(diff), where diff is the
    sampled empty-chamber frame-to-frame grayscale difference.
    """

    video_path = os.path.normpath(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open calibration video: {video_path}")

    if not accept_p_frames:
        check_p_frames(cap)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(int(cal_frames), max(0, frame_count - int(start_frame)))
    if max_frames < 2:
        cap.release()
        raise RuntimeError("Calibration needs at least two readable frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    first = _read_gray_frame(cap, float(dsmpl), crop)
    if first is None:
        cap.release()
        raise RuntimeError(f"No readable frames found in calibration video: {video_path}")

    previous = cv2.GaussianBlur(first.astype("float"), (0, 0), float(sigma))
    height, width = previous.shape
    rng = np.random.default_rng(int(random_seed))
    h_loc = rng.integers(0, height, size=int(cal_pixels))
    w_loc = rng.integers(0, width, size=int(cal_pixels))

    differences = []
    for _ in range(1, max_frames):
        current = _read_gray_frame(cap, float(dsmpl), crop)
        if current is None:
            break
        current = cv2.GaussianBlur(current.astype("float"), (0, 0), float(sigma))
        differences.append(np.abs(current[h_loc, w_loc] - previous[h_loc, w_loc]))
        previous = current

    cap.release()
    if not differences:
        raise RuntimeError("Calibration needs at least two readable frames.")

    values = np.concatenate(differences)
    percentile_value = float(np.percentile(values, float(percentile)))
    motion_cutoff = float(cutoff_multiplier) * percentile_value
    hist_counts, hist_edges = np.histogram(values, bins=80)
    return {
        "method": "percentile_multiplier",
        "average_pixel_difference": float(np.mean(values)),
        "zero_fraction": float(np.mean(values == 0)),
        "nonzero_samples": int(np.sum(values > 0)),
        "percentile": float(percentile),
        "percentile_value": percentile_value,
        "cutoff_multiplier": float(cutoff_multiplier),
        "motion_cutoff": float(motion_cutoff),
        "frames_used": len(differences) + 1,
        "pixels_sampled": int(cal_pixels),
        "sigma": float(sigma),
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
    }
