#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate transformed EPM heatmaps from LocationOutput CSV using YAML config.

This script performs regional geometric transformation from real maze space
to an idealized plus-maze template, then generates standardized heatmaps.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def normalize_video_id(name):
    s = str(name)
    if s.endswith("_LocationOutput"):
        s = s[: -len("_LocationOutput")]
    if s.lower().endswith(".mp4"):
        s = s[:-4]
    return s


def create_ideal_plus_maze(center_size, arm_ratio, canvas_size):
    """Create ideal 12-vertex plus-maze polygon with fixed vertex order."""
    arm_length = center_size * arm_ratio
    mid = canvas_size / 2.0
    half = center_size / 2.0
    al = arm_length

    # Vertex order must match source order in config.epm_transform.original_vertices
    return np.array(
        [
            [mid - al - half, mid - half],  # 0
            [mid - al - half, mid + half],  # 1
            [mid - half, mid + half],  # 2
            [mid - half, mid + al + half],  # 3
            [mid + half, mid + al + half],  # 4
            [mid + half, mid + half],  # 5
            [mid + al + half, mid + half],  # 6
            [mid + al + half, mid - half],  # 7
            [mid + half, mid - half],  # 8
            [mid + half, mid - al - half],  # 9
            [mid - half, mid - al - half],  # 10
            [mid - half, mid - half],  # 11
        ],
        dtype=np.float32,
    )


def define_regions(vertices):
    """Define quadrilateral regions by vertex indices."""
    v = np.asarray(vertices, dtype=np.float32)
    return {
        "center": v[[11, 8, 5, 2]],
        "left_arm": v[[0, 11, 2, 1]],
        "bottom_arm": v[[2, 5, 4, 3]],
        "right_arm": v[[5, 8, 7, 6]],
        "top_arm": v[[10, 9, 8, 11]],
    }


def point_in_quad(x, y, quad):
    return cv2.pointPolygonTest(np.asarray(quad, dtype=np.float32), (float(x), float(y)), False) >= 0


def determine_region(x, y, src_regions):
    # Prefer center first for overlap edge behavior.
    order = ["center", "left_arm", "right_arm", "top_arm", "bottom_arm"]
    for name in order:
        if point_in_quad(x, y, src_regions[name]):
            return name

    # Fallback: assign nearest region boundary by signed distance.
    best = None
    best_dist = -1e9
    for name, poly in src_regions.items():
        d = cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), (float(x), float(y)), True)
        if d > best_dist:
            best_dist = d
            best = name
    return best


def compute_region_homographies(src_vertices, dst_vertices):
    src_regions = define_regions(src_vertices)
    dst_regions = define_regions(dst_vertices)
    homographies = {}
    for name in src_regions:
        h = cv2.getPerspectiveTransform(
            np.asarray(src_regions[name], dtype=np.float32),
            np.asarray(dst_regions[name], dtype=np.float32),
        )
        homographies[name] = h
    return src_regions, homographies


def transform_point(x, y, src_regions, homographies):
    region = determine_region(x, y, src_regions)
    h = homographies[region]
    inp = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(inp, h)
    tx, ty = out[0, 0, 0], out[0, 0, 1]
    return float(tx), float(ty), region


def read_video_fps(video_path, fallback):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float(fallback)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0 or np.isnan(fps):
        return float(fallback)
    return float(fps)


def _load_functional_regions(cfg, require_quad=True):
    functional = cfg.get("functional_roi", {}) or {}
    regions = functional.get("regions", []) or []
    if not regions:
        raise KeyError("functional_roi.regions is required for this script.")
    out = {}
    for r in regions:
        name = str(r.get("name", "")).strip()
        vertices = r.get("vertices", []) or []
        if not name:
            raise ValueError("Each functional ROI region must have a non-empty 'name'.")
        if require_quad and len(vertices) != 4:
            raise ValueError(
                f"ROI '{name}' has {len(vertices)} vertices. "
                "This script requires exactly 4 vertices per ROI."
            )
        if len(vertices) < 3:
            raise ValueError(f"ROI '{name}' must have at least 3 vertices.")
        out[name] = np.asarray(vertices, dtype=np.float32)
    return out


def _load_crop_from_config(cfg):
    crop_cfg = cfg.get("crop", {}) or {}
    if all(k in crop_cfg for k in ("x0", "x1", "y0", "y1")):
        return (
            int(crop_cfg["x0"]),
            int(crop_cfg["x1"]),
            int(crop_cfg["y0"]),
            int(crop_cfg["y1"]),
        )
    return None


def _require_epm_vertices_from_config(cfg):
    epm = cfg.get("epm_transform", {}) or {}
    original_vertices = epm.get("original_vertices")
    if original_vertices is None:
        raise KeyError(
            "Missing required key: epm_transform.original_vertices. "
            "Expected 12 ordered vertices."
        )
    if not isinstance(original_vertices, list) or len(original_vertices) != 12:
        raise ValueError("epm_transform.original_vertices must be a list of 12 [x,y] points.")
    return np.asarray(original_vertices, dtype=np.float32)


def _derive_epm_vertices_from_functional(cfg):
    regions = _load_functional_regions(cfg)
    # Accept common case-insensitive names.
    by_lower = {k.lower(): v for k, v in regions.items()}
    required = ["left", "right", "top", "bottom"]
    for key in required:
        if key not in by_lower:
            raise KeyError(
                "functional_roi.regions must include Left/Right/Top/Bottom "
                f"to derive EPM vertices automatically. Missing: {key}"
            )

    left = by_lower["left"]
    right = by_lower["right"]
    top = by_lower["top"]
    bottom = by_lower["bottom"]

    def split_by_axis(pts, axis, outer_is_min):
        vals = pts[:, axis]
        order = np.argsort(vals)
        if outer_is_min:
            outer_idx = order[:2]
            inner_idx = order[-2:]
        else:
            outer_idx = order[-2:]
            inner_idx = order[:2]
        outer = pts[outer_idx]
        inner = pts[inner_idx]
        return outer, inner

    # Left arm: outer=min x, inner=max x
    left_outer, left_inner = split_by_axis(left, axis=0, outer_is_min=True)
    left_outer_top = left_outer[np.argmin(left_outer[:, 1])]
    left_outer_bottom = left_outer[np.argmax(left_outer[:, 1])]
    left_inner_top = left_inner[np.argmin(left_inner[:, 1])]
    left_inner_bottom = left_inner[np.argmax(left_inner[:, 1])]

    # Right arm: outer=max x, inner=min x
    right_outer, right_inner = split_by_axis(right, axis=0, outer_is_min=False)
    right_outer_top = right_outer[np.argmin(right_outer[:, 1])]
    right_outer_bottom = right_outer[np.argmax(right_outer[:, 1])]
    right_inner_top = right_inner[np.argmin(right_inner[:, 1])]
    right_inner_bottom = right_inner[np.argmax(right_inner[:, 1])]

    # Top arm: outer=min y, inner=max y
    top_outer, top_inner = split_by_axis(top, axis=1, outer_is_min=True)
    top_outer_left = top_outer[np.argmin(top_outer[:, 0])]
    top_outer_right = top_outer[np.argmax(top_outer[:, 0])]
    top_inner_left = top_inner[np.argmin(top_inner[:, 0])]
    top_inner_right = top_inner[np.argmax(top_inner[:, 0])]

    # Bottom arm: outer=max y, inner=min y
    bottom_outer, bottom_inner = split_by_axis(bottom, axis=1, outer_is_min=False)
    bottom_outer_left = bottom_outer[np.argmin(bottom_outer[:, 0])]
    bottom_outer_right = bottom_outer[np.argmax(bottom_outer[:, 0])]
    bottom_inner_left = bottom_inner[np.argmin(bottom_inner[:, 0])]
    bottom_inner_right = bottom_inner[np.argmax(bottom_inner[:, 0])]

    # Build 12-point order used by this script.
    verts = np.array(
        [
            left_outer_top,      # 0
            left_outer_bottom,   # 1
            left_inner_bottom,   # 2
            bottom_outer_left,   # 3
            bottom_outer_right,  # 4
            right_inner_bottom,  # 5
            right_outer_bottom,  # 6
            right_outer_top,     # 7
            right_inner_top,     # 8
            top_outer_right,     # 9
            top_outer_left,      # 10
            left_inner_top,      # 11
        ],
        dtype=np.float32,
    )

    # Small sanity check: center corners from top/bottom and left/right should be close.
    # We keep this permissive and only warn through exception message if geometry is wild.
    center_candidates = np.array([left_inner_top, left_inner_bottom, right_inner_top, right_inner_bottom,
                                  top_inner_left, top_inner_right, bottom_inner_left, bottom_inner_right], dtype=np.float32)
    spread = np.max(np.linalg.norm(center_candidates - center_candidates.mean(axis=0), axis=1))
    if spread > 250:
        raise ValueError(
            "Derived vertices look unstable from functional_roi (large center spread). "
            "Use --vertex-mode gui or provide epm_transform.original_vertices."
        )
    return verts


def _save_vertices_to_config(config_path, vertices, cfg):
    cfg = dict(cfg)
    epm = dict(cfg.get("epm_transform", {}) or {})
    epm["original_vertices"] = [[float(x), float(y)] for x, y in np.asarray(vertices, dtype=float)]
    epm.setdefault("center_size_px", 119)
    epm.setdefault("arm_length_ratio", 5.0)
    epm.setdefault("canvas_size", 1409)
    epm.setdefault("num_bins", 80)
    epm.setdefault("sigma", 1.2)
    epm.setdefault("colormap", "viridis")
    epm.setdefault("skip_seconds", 15.0)
    epm.setdefault("fps_default", 25.0)
    cfg["epm_transform"] = epm
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def _pick_random_frame_with_overlay(video_dir, functional_regions, crop=None):
    videos = sorted([p for p in Path(video_dir).glob("*.mp4") if not p.name.endswith("_TrackingOverlay.mp4")])
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found in: {video_dir}")
    video_path = random.choice(videos)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 1:
        idx = random.randint(0, max(0, n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from video: {video_path}")

    # Apply config crop first so displayed frame coordinates match tracking coordinates.
    if crop is not None:
        x0, x1, y0, y1 = crop
        h, w = frame.shape[:2]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            frame = frame[y0:y1, x0:x1]

    # Draw functional ROI overlay.
    overlay = frame.copy()
    colors = [(255, 120, 80), (80, 120, 255), (110, 255, 140), (255, 230, 90), (200, 120, 255)]
    for i, (name, poly) in enumerate(functional_regions.items()):
        c = colors[i % len(colors)]
        pts = np.asarray(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], c)
        cv2.polylines(frame, [pts], True, c, 2)
        centroid = np.mean(pts, axis=0).astype(int)
        cv2.putText(frame, name, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    return frame, video_path


def _pick_vertices_gui(video_dir, functional_regions, crop=None, save_to_config=False, config_path=None):
    # Lazy imports to keep non-GUI mode lightweight.
    import tkinter as tk
    import customtkinter as ctk
    from PIL import Image, ImageTk

    frame_bgr, picked_video = _pick_random_frame_with_overlay(video_dir, functional_regions, crop=crop)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    max_w, max_h = 980, 620
    scale = min(max_w / w, max_h / h, 1.0)
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
    frame_resized = cv2.resize(frame_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    frame_img = Image.fromarray(frame_resized)

    # Build reference EPM coordinates on small panel.
    ref_size = 360
    ref_vertices = create_ideal_plus_maze(center_size=48, arm_ratio=2.5, canvas_size=ref_size)
    labels = [
        "0 L-outer-top", "1 L-outer-bottom", "2 Center-L-bottom",
        "3 Bottom-L", "4 Bottom-R", "5 Center-R-bottom",
        "6 R-outer-bottom", "7 R-outer-top", "8 Center-R-top",
        "9 Top-R", "10 Top-L", "11 Center-L-top",
    ]
    selected = [None] * 12
    history = []
    active_idx = {"value": 0}
    result = {"vertices": None}

    # Candidate points are strictly functional ROI vertices (deduplicated).
    all_pts = []
    for poly in functional_regions.values():
        for p in np.asarray(poly, dtype=np.float32):
            all_pts.append((float(p[0]), float(p[1])))
    # Deduplicate by rounding to 0.1 px.
    unique = {}
    for x, y in all_pts:
        unique[(round(x, 1), round(y, 1))] = (x, y)
    candidate_pts = np.asarray(list(unique.values()), dtype=np.float32)
    if candidate_pts.shape[0] == 0:
        raise ValueError("No functional ROI vertices found for GUI picking.")

    ctk.set_appearance_mode("dark")
    app = ctk.CTk()
    tk_frame_img = ImageTk.PhotoImage(frame_img, master=app)
    app.title("Pick EPM Transform Vertices")
    app.geometry("1500x860")

    title = ctk.CTkLabel(
        app,
        text=f"Video: {picked_video.name} | Select reference point (right), then click frame (left).",
        font=("Segoe UI", 16, "bold"),
    )
    title.pack(pady=(10, 6))

    main = ctk.CTkFrame(app)
    main.pack(fill="both", expand=True, padx=10, pady=8)

    left = ctk.CTkFrame(main)
    left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
    right = ctk.CTkFrame(main, width=420)
    right.pack(side="right", fill="y", padx=(4, 8), pady=8)
    right.pack_propagate(False)

    canvas_frame = tk.Canvas(left, width=disp_w, height=disp_h, bg="black", highlightthickness=0)
    canvas_frame.pack(padx=8, pady=8)
    canvas_frame.create_image(0, 0, image=tk_frame_img, anchor="nw")

    canvas_ref = tk.Canvas(right, width=ref_size, height=ref_size, bg="#1a1a1a", highlightthickness=0)
    canvas_ref.pack(padx=10, pady=(10, 6))

    pts = ref_vertices.astype(int)
    canvas_ref.create_polygon(pts.flatten().tolist(), outline="white", fill="", width=2)
    for i, (x, y) in enumerate(pts):
        canvas_ref.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#9aa0a6", outline="")
        canvas_ref.create_text(x + 10, y - 10, text=str(i), fill="#d7dce3", font=("Segoe UI", 9, "bold"))

    list_var = tk.StringVar(value=[f"[ ] {i}: {labels[i]}" for i in range(12)])
    listbox = tk.Listbox(right, listvariable=list_var, width=48, height=14, exportselection=False)
    listbox.pack(padx=10, pady=(4, 8), fill="x")
    listbox.selection_set(0)

    save_target = config_path if (save_to_config and config_path) else "(session only; not writing config)"
    hint = ctk.CTkLabel(
        right,
        text="Tip: pick reference point (right), then click left frame. "
             "Selection snaps to nearest functional ROI vertex.\n"
             f"Save target: {save_target}",
        font=("Segoe UI", 12),
    )
    hint.pack(pady=(2, 10))

    def redraw():
        canvas_ref.delete("active_marker")
        # Active marker on reference.
        ax, ay = ref_vertices[active_idx["value"]]
        canvas_ref.create_oval(ax - 9, ay - 9, ax + 9, ay + 9, outline="#f7d154", width=2, tags="active_marker")

        canvas_frame.delete("picked")
        # Draw all selectable candidate vertices.
        for vx, vy in candidate_pts:
            sx, sy = vx * scale, vy * scale
            canvas_frame.create_oval(sx - 2, sy - 2, sx + 2, sy + 2, fill="#8ecae6", outline="", tags="picked")
        rows = []
        for i, v in enumerate(selected):
            if v is None:
                rows.append(f"[ ] {i}: {labels[i]}")
            else:
                x, y = v
                rows.append(f"[x] {i}: {labels[i]} -> ({x:.1f}, {y:.1f})")
                sx, sy = x * scale, y * scale
                canvas_frame.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#ffd54f", outline="", tags="picked")
                canvas_frame.create_text(sx + 9, sy - 9, text=str(i), fill="#ffe082", font=("Segoe UI", 9, "bold"), tags="picked")
        list_var.set(rows)
        listbox.selection_clear(0, "end")
        listbox.selection_set(active_idx["value"])

    def choose_next():
        start = active_idx["value"]
        for step in range(1, 13):
            j = (start + step) % 12
            if selected[j] is None:
                active_idx["value"] = j
                return

    def on_ref_click(event):
        p = np.array([event.x, event.y], dtype=float)
        d = np.linalg.norm(ref_vertices - p, axis=1)
        active_idx["value"] = int(np.argmin(d))
        redraw()

    def on_frame_click(event):
        x = float(event.x) / scale
        y = float(event.y) / scale
        d = np.linalg.norm(candidate_pts - np.array([x, y], dtype=np.float32), axis=1)
        nearest = candidate_pts[int(np.argmin(d))]
        idx = active_idx["value"]
        prev = None if selected[idx] is None else [selected[idx][0], selected[idx][1]]
        selected[idx] = [float(nearest[0]), float(nearest[1])]
        history.append((idx, prev))
        choose_next()
        redraw()

    def on_list_select(_event):
        sel = listbox.curselection()
        if sel:
            active_idx["value"] = int(sel[0])
            redraw()

    def on_clear():
        for i in range(12):
            selected[i] = None
        history.clear()
        active_idx["value"] = 0
        redraw()

    def on_undo():
        if not history:
            hint.configure(text="Nothing to undo.", text_color="#ffd166")
            return
        idx, prev = history.pop()
        selected[idx] = prev
        active_idx["value"] = idx
        hint.configure(
            text="Undid last point selection.",
            text_color="#c3e88d",
        )
        redraw()

    def on_cancel():
        result["vertices"] = None
        app.destroy()

    def on_save():
        if any(v is None for v in selected):
            missing = [str(i) for i, v in enumerate(selected) if v is None]
            hint.configure(text=f"Missing points: {', '.join(missing)}", text_color="#ff8a80")
            return
        result["vertices"] = np.asarray(selected, dtype=np.float32)
        app.destroy()

    canvas_ref.bind("<Button-1>", on_ref_click)
    canvas_frame.bind("<Button-1>", on_frame_click)
    listbox.bind("<<ListboxSelect>>", on_list_select)

    # Fixed bottom action bar so Save button is always visible.
    action = ctk.CTkFrame(app)
    action.pack(fill="x", padx=10, pady=(0, 10))
    ctk.CTkButton(action, text="Undo (Ctrl+Z)", command=on_undo).pack(side="left", padx=6, pady=6)
    ctk.CTkButton(action, text="Clear (C)", command=on_clear).pack(side="left", padx=6, pady=6)
    ctk.CTkButton(action, text="Cancel (Esc)", command=on_cancel).pack(side="right", padx=6, pady=6)
    ctk.CTkButton(action, text="Save 12 Vertices (Enter)", command=on_save, fg_color="#1F6AA5").pack(side="right", padx=6, pady=6)

    # Keyboard shortcuts
    app.bind("<Return>", lambda _e: on_save())
    app.bind("<Escape>", lambda _e: on_cancel())
    app.bind("<Control-z>", lambda _e: on_undo())
    app.bind("<Control-Z>", lambda _e: on_undo())
    app.bind("c", lambda _e: on_clear())
    app.bind("C", lambda _e: on_clear())

    def on_close():
        on_cancel()

    app.protocol("WM_DELETE_WINDOW", on_close)

    redraw()
    app.mainloop()
    if result["vertices"] is None:
        return None
    return result["vertices"]


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

    # Hard requirement for this script: all functional ROI polygons must be quadrilaterals.
    _load_functional_regions(cfg, require_quad=True)

    epm = cfg.get("epm_transform", {}) or {}
    params = {
        "center_size_px": int(epm.get("center_size_px", 119)),
        "arm_length_ratio": float(epm.get("arm_length_ratio", 5.0)),
        "canvas_size": int(epm.get("canvas_size", 1409)),
        "num_bins": int(epm.get("num_bins", 80)),
        "sigma": float(epm.get("sigma", 1.2)),
        "colormap": str(epm.get("colormap", "viridis")),
        "skip_seconds": float(epm.get("skip_seconds", 15.0)),
        "fps_default": float(epm.get("fps_default", 25.0)),
    }
    return cfg, video_dir, params


def process_one_csv(csv_path, src_vertices, params, output_prefix=None):
    df = pd.read_csv(csv_path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"CSV missing X/Y columns: {csv_path}")

    stem = normalize_video_id(csv_path.stem)
    video_path = csv_path.with_name(f"{stem}.mp4")
    fps = read_video_fps(video_path, fallback=params["fps_default"])
    skip_frames = int(round(params["skip_seconds"] * fps))

    x = pd.to_numeric(df["X"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=float)
    frame_col = (
        pd.to_numeric(df.get("Frame", pd.Series(np.arange(len(df)))), errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    frames = frame_col[valid]
    valid_df = df.loc[valid].reset_index(drop=True)

    if len(x) == 0:
        raise ValueError(f"No valid X/Y points in: {csv_path.name}")

    if len(x) > skip_frames:
        x = x[skip_frames:]
        y = y[skip_frames:]
        frames = frames[skip_frames:]
        valid_df = valid_df.iloc[skip_frames:].reset_index(drop=True)

    dst_vertices = create_ideal_plus_maze(
        center_size=params["center_size_px"],
        arm_ratio=params["arm_length_ratio"],
        canvas_size=params["canvas_size"],
    )
    src_regions, homographies = compute_region_homographies(src_vertices, dst_vertices)

    transformed = np.zeros((len(x), 2), dtype=np.float32)
    regions = []
    for i, (px, py) in enumerate(zip(x, y)):
        tx, ty, region = transform_point(px, py, src_regions, homographies)
        transformed[i, 0] = tx
        transformed[i, 1] = ty
        regions.append(region)

    tx = transformed[:, 0]
    ty = transformed[:, 1]

    # Histogram + smooth
    canvas = params["canvas_size"]
    heatmap, _, _ = np.histogram2d(
        tx,
        ty,
        bins=params["num_bins"],
        range=[[0, canvas], [0, canvas]],
    )
    heatmap_s = gaussian_filter(heatmap, sigma=params["sigma"])
    heatmap_rescaled = cv2.resize(heatmap_s, (canvas, canvas), interpolation=cv2.INTER_LINEAR).T

    # Maze mask
    maze_path = MplPath(dst_vertices.astype(np.float32))
    y_grid, x_grid = np.mgrid[0:canvas, 0:canvas]
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask_inside = maze_path.contains_points(points).reshape(canvas, canvas)
    heatmap_final = np.zeros((canvas, canvas), dtype=np.float32)
    heatmap_final[mask_inside] = heatmap_rescaled[mask_inside]
    heatmap_masked = np.ma.masked_where(~mask_inside, heatmap_final)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("black")
    im = ax.imshow(
        heatmap_masked,
        cmap=params["colormap"],
        extent=[0, canvas, canvas, 0],
        interpolation="bilinear",
        alpha=0.9,
        vmin=0,
    )
    ax.add_patch(Polygon(dst_vertices, fill=False, edgecolor="white", linewidth=2, alpha=0.8))
    ax.set_title(
        f"EPM Standardized Heatmap ({params['num_bins']}x{params['num_bins']})",
        fontsize=14,
        color="white",
        fontweight="bold",
    )
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Stay Duration (Density)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    fig.patch.set_facecolor("black")
    plt.tight_layout()

    if output_prefix is None:
        output_prefix = csv_path.with_name(f"{stem}_EPM_Heatmap_Standard")
    output_prefix = Path(output_prefix)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="black", edgecolor="none", transparent=False)
    plt.close(fig)

    transformed_csv = csv_path.with_name(f"{stem}_TransformedTrajectory.csv")
    out_df = pd.DataFrame(
        {
            "Frame": frames,
            "X_Original": x,
            "Y_Original": y,
            "X_Transformed": tx,
            "Y_Transformed": ty,
            "Region": regions,
        }
    )
    if "Distance_px" in valid_df.columns:
        out_df["Distance_px"] = pd.to_numeric(valid_df["Distance_px"], errors="coerce").to_numpy(dtype=float)
    if "Distance_mm" in valid_df.columns:
        out_df["Distance_mm"] = pd.to_numeric(valid_df["Distance_mm"], errors="coerce").to_numpy(dtype=float)
    out_df.to_csv(transformed_csv, index=False)

    return png_path, pdf_path, transformed_csv, len(out_df), fps


def main():
    parser = argparse.ArgumentParser(
        description="Generate transformed EPM heatmaps from YAML project config."
    )
    parser.add_argument("--config", required=True, help="Path to project_tracking_config.yml")
    parser.add_argument("--video", default=None, help="Single-video mode (stem or filename), e.g. 1-5 or 1-5.mp4")
    parser.add_argument(
        "--vertex-mode",
        choices=["config", "functional", "gui"],
        default="config",
        help="How to obtain transform vertices: config=epm_transform.original_vertices, "
             "functional=derive from functional_roi Left/Right/Top/Bottom, gui=interactive picker",
    )
    parser.add_argument(
        "--save-picked-vertices",
        action="store_true",
        help="When --vertex-mode gui/functional is used, write resolved vertices back to config.epm_transform.original_vertices",
    )
    parser.add_argument("--num-bins", type=int, default=None, help="Override epm_transform.num_bins")
    parser.add_argument("--sigma", type=float, default=None, help="Override epm_transform.sigma")
    parser.add_argument("--skip-seconds", type=float, default=None, help="Override epm_transform.skip_seconds")
    args = parser.parse_args()

    cfg, video_dir, params = load_config(args.config)
    if args.num_bins is not None:
        params["num_bins"] = int(args.num_bins)
    if args.sigma is not None:
        params["sigma"] = float(args.sigma)
    if args.skip_seconds is not None:
        params["skip_seconds"] = float(args.skip_seconds)

    if not video_dir.exists():
        raise FileNotFoundError(f"Config video_dir does not exist: {video_dir}")

    if args.video:
        stem = normalize_video_id(args.video)
        csv_files = [video_dir / f"{stem}_LocationOutput.csv"]
    else:
        csv_files = sorted(video_dir.glob("*_LocationOutput.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No *_LocationOutput.csv found in: {video_dir}")

    if args.vertex_mode == "config":
        src_vertices = _require_epm_vertices_from_config(cfg)
    elif args.vertex_mode == "functional":
        src_vertices = _derive_epm_vertices_from_functional(cfg)
        print("[INFO] Using vertices derived from functional_roi.")
        if args.save_picked_vertices:
            _save_vertices_to_config(args.config, src_vertices, cfg)
            print("[INFO] Saved derived vertices to config.epm_transform.original_vertices")
    else:
        functional_regions = _load_functional_regions(cfg)
        if not functional_regions:
            raise KeyError("functional_roi.regions is required for GUI overlay in --vertex-mode gui")
        crop = _load_crop_from_config(cfg)
        src_vertices = _pick_vertices_gui(
            video_dir,
            functional_regions,
            crop=crop,
            save_to_config=bool(args.save_picked_vertices),
            config_path=args.config,
        )
        if src_vertices is None:
            print(
                "[ABORT] Vertex picking was canceled (Esc, Cancel, or closed window without Save).\n"
                "        Pick all 12 points on the left frame (reference on the right), then click "
                "'Save 12 Vertices' or press Enter.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("[INFO] Using vertices selected from GUI.")
        if args.save_picked_vertices:
            _save_vertices_to_config(args.config, src_vertices, cfg)
            print("[INFO] Saved GUI-picked vertices to config.epm_transform.original_vertices")

    print("=" * 88)
    print("Generate Transformed Trajectory Heatmap (YAML-driven)")
    print("=" * 88)
    print(f"Config: {args.config}")
    print(f"Video dir: {video_dir}")
    print(f"Files: {len(csv_files)}")
    print(f"Vertex mode: {args.vertex_mode}")
    print(
        f"Params: center={params['center_size_px']}, ratio={params['arm_length_ratio']}, "
        f"canvas={params['canvas_size']}, bins={params['num_bins']}, sigma={params['sigma']}, "
        f"skip={params['skip_seconds']}s"
    )
    print("=" * 88)

    ok = 0
    fail = 0
    for csv_path in csv_files:
        try:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            png_path, pdf_path, transformed_csv, n_points, fps = process_one_csv(csv_path, src_vertices, params)
            print(
                f"[OK] {csv_path.name} -> {png_path.name}, {pdf_path.name}, {transformed_csv.name} "
                f"(points={n_points}, fps={fps:.2f})"
            )
            ok += 1
        except Exception as e:
            print(f"[FAIL] {csv_path.name}: {e}")
            fail += 1

    print("-" * 88)
    print(f"Done. ok={ok} fail={fail}")
    print("=" * 88)


if __name__ == "__main__":
    main()
