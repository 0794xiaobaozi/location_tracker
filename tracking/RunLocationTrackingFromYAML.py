#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run batch location tracking from a single YAML config file.

Example:
  pixi run -e location-tracker python RunLocationTrackingFromYAML.py ^
    --config ".\\project_tracking_config.yml"
"""

import os
import argparse
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

import LocationTracking_Functions as lt


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return data


def _require(mapping: Dict[str, Any], key: str):
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def _to_mock_crop(crop_cfg: Optional[Dict[str, Any]]):
    if not crop_cfg:
        return None
    required = ["x0", "x1", "y0", "y1"]
    for k in required:
        if k not in crop_cfg:
            raise KeyError(f"crop missing key: {k}")
    return lt.MockCrop(
        {
            "x0": [int(crop_cfg["x0"])],
            "x1": [int(crop_cfg["x1"])],
            "y0": [int(crop_cfg["y0"])],
            "y1": [int(crop_cfg["y1"])],
        }
    )


def _to_mock_roi_stream(functional_roi_cfg: Optional[Dict[str, Any]]):
    if not functional_roi_cfg:
        return None, None
    regions = functional_roi_cfg.get("regions", [])
    if not regions:
        return None, None

    region_names = []
    xs, ys = [], []
    for region in regions:
        name = region.get("name")
        vertices = region.get("vertices")
        if not name or not vertices:
            raise ValueError("Each functional ROI region needs 'name' and 'vertices'.")
        region_names.append(str(name))
        x_coords = [float(p[0]) for p in vertices]
        y_coords = [float(p[1]) for p in vertices]
        xs.append(x_coords)
        ys.append(y_coords)

    # Match Holoviews PolyDraw stream schema expected by ezTrack internals.
    return region_names, lt.MockStream({"xs": xs, "ys": ys})


def _read_first_gray_frame(video_dict: Dict[str, Any]) -> np.ndarray:
    first_file = video_dict["FileNames"][0]
    fpath = os.path.join(os.path.normpath(video_dict["dpath"]), first_file)
    cap = cv2.VideoCapture(fpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_dict.get("start", 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from: {fpath}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dsmpl = float(video_dict.get("dsmpl", 1.0))
    if dsmpl < 1.0:
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * dsmpl), int(frame.shape[0] * dsmpl)),
            interpolation=cv2.INTER_AREA,
        )
    frame = lt.cropframe(frame, video_dict.get("crop"))
    return frame


def _build_analysis_roi(
    analysis_cfg: Optional[Dict[str, Any]], video_dict: Dict[str, Any]
) -> Optional[Any]:
    if not analysis_cfg:
        return None

    roi_type = analysis_cfg.get("type", "none")
    if roi_type == "none":
        return None

    if roi_type == "rectangle":
        x1 = int(_require(analysis_cfg, "x1"))
        y1 = int(_require(analysis_cfg, "y1"))
        x2 = int(_require(analysis_cfg, "x2"))
        y2 = int(_require(analysis_cfg, "y2"))
        return (x1, y1, x2, y2)

    if roi_type == "polygon":
        vertices = analysis_cfg.get("vertices", [])
        if not vertices:
            raise ValueError("analysis_roi.type=polygon requires 'vertices'.")

        frame_gray = _read_first_gray_frame(video_dict)
        mask = np.zeros(frame_gray.shape, dtype=np.uint8)

        polygons_px = []
        for poly in vertices:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            polygons_px.append(pts.tolist())
            cv2.fillPoly(mask, [pts], 255)

        return {"type": "polygon", "vertices": polygons_px, "mask": mask > 0}

    raise ValueError(f"Unsupported analysis_roi.type: {roi_type}")


def _build_video_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    project = _require(cfg, "project")
    tracking = _require(cfg, "tracking")
    dpath = os.path.normpath(_require(project, "video_dir"))
    ftype = project.get("file_type", "mp4")

    video_dict = {
        "dpath": dpath,
        "file": "",
        "ftype": ftype,
        "start": int(project.get("start_frame", 0)),
        "end": project.get("end_frame", None),
        "dsmpl": float(project.get("dsmpl", 1.0)),
        "stretch": {
            "width": float(project.get("stretch_width", 1.0)),
            "height": float(project.get("stretch_height", 1.0)),
        },
        "region_names": None,
        "roi_stream": None,
        "crop": None,
        "mask": {"mask": None, "stream": None},
        "scale": None,
        "analysis_roi": None,
    }

    crop_cfg = cfg.get("crop")
    video_dict["crop"] = _to_mock_crop(crop_cfg)

    # Load files now because polygon ROI creation needs frame shape.
    video_dict = lt.Batch_LoadFiles(video_dict)
    if len(video_dict.get("FileNames", [])) == 0:
        raise RuntimeError(f"No .{ftype} videos found in: {dpath}")

    video_dict["analysis_roi"] = _build_analysis_roi(cfg.get("analysis_roi"), video_dict)
    region_names, roi_stream = _to_mock_roi_stream(cfg.get("functional_roi"))
    video_dict["region_names"] = region_names
    video_dict["roi_stream"] = roi_stream

    scale_cfg = cfg.get("scale")
    if scale_cfg:
        video_dict["scale"] = {
            "px_distance": float(scale_cfg.get("px_distance", 0)),
            "true_distance": float(scale_cfg.get("true_distance", 0)),
            "true_scale": str(scale_cfg.get("true_scale", "cm")),
            "factor": float(scale_cfg.get("factor", 0)),
        }

    # Keep tracking in local for validation side effects.
    _ = tracking
    return video_dict


def _build_tracking_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tracking = _require(cfg, "tracking")
    return {
        "loc_thresh": float(tracking.get("loc_thresh", 99.0)),
        "use_window": bool(tracking.get("use_window", False)),
        "window_size": int(tracking.get("window_size", 100)),
        "window_weight": float(tracking.get("window_weight", 0.9)),
        "method": str(tracking.get("method", "abs")),
        "rmv_wire": bool(tracking.get("rmv_wire", False)),
        "wire_krn": int(tracking.get("wire_krn", 3)),
    }


def _build_run_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = cfg.get("run", {})
    return {
        "parallel": bool(run_cfg.get("parallel", True)),
        "n_processes": run_cfg.get("n_processes", None),
        "accept_p_frames": bool(run_cfg.get("accept_p_frames", False)),
        "bin_dict": run_cfg.get("bin_dict", None),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run LocationTracking batch processing from YAML config."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = os.path.normpath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_yaml(config_path)
    video_dict = _build_video_dict(cfg)
    tracking_params = _build_tracking_params(cfg)
    run_opts = _build_run_options(cfg)

    print("=" * 70)
    print("BATCH LOCATION TRACKING (YAML)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Directory: {video_dict['dpath']}")
    print(f"Videos: {len(video_dict.get('FileNames', []))}")
    print(f"Parallel: {run_opts['parallel']}")
    print("=" * 70)

    if run_opts["parallel"]:
        try:
            summary, _layout = lt.Batch_Process_Parallel(
                video_dict,
                tracking_params,
                run_opts["bin_dict"],
                n_processes=run_opts["n_processes"],
                accept_p_frames=run_opts["accept_p_frames"],
            )
        except UnicodeEncodeError:
            print("[WARN] Console encoding does not support emoji output from parallel mode.")
            print("[WARN] Falling back to sequential Batch_Process for this run.")
            summary, _layout = lt.Batch_Process(
                video_dict,
                tracking_params,
                run_opts["bin_dict"],
                accept_p_frames=run_opts["accept_p_frames"],
            )
    else:
        summary, _layout = lt.Batch_Process(
            video_dict,
            tracking_params,
            run_opts["bin_dict"],
            accept_p_frames=run_opts["accept_p_frames"],
        )

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Rows in summary: {len(summary)}")
    print(f"Summary file: {os.path.join(video_dict['dpath'], 'BatchSummary.csv')}")


if __name__ == "__main__":
    main()

