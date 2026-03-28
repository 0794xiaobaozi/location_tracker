#!/usr/bin/env python3
"""
Dual-camera merge GUI for a fixed pair of folders (cam3 left + cam2 right).

Reads folder paths from `dual_camera_merge_paths.json` (see `dual_camera_merge_paths.example.json`).
MP4s are paired by **same filename** in both folders, sorted in display order. The GUI walks
through that list one clip at a time — no per-session file picking.

Writes a merge **plan** to `dual_camera_merge_jobs.json` (start frames per clip). Run
`MergeDualCameraVideos.py` to execute ffmpeg using that plan.

Pipeline (MergeDualCameraVideos.py):
  - Video A (cam3 folder): left half, transpose=1 (90° CW)
  - Video B (cam2 folder): right half, transpose=2 (90° CCW)
  - hstack; output length = frame intersection from chosen start_a / start_b

Requires: pixi env `location-tracker` (OpenCV, customtkinter, Pillow). Actual merge needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dual_camera_merge import (
    alignments_from_jobs,
    build_jobs_payload,
    clips_from_alignments,
    default_jobs_path,
    discover_pairs,
    intersection_length,
    load_jobs_file,
    load_paths_config,
    probe_video,
    save_jobs_file,
)


class DualAxisMergeApp:
    """Queue of paired clips from two fixed directories; edit alignment, save merge plan JSON."""

    def __init__(self, config_path: Path) -> None:
        import customtkinter as ctk
        from PIL import Image

        self.ctk = ctk
        self.Image = Image
        self.config_path = config_path
        self.jobs_path = default_jobs_path(Path(__file__).resolve().parent)

        cfg = load_paths_config(config_path)
        self.dir_a = Path(cfg["dir_video_a_cam3_left"])
        self.dir_b = Path(cfg["dir_video_b_cam2_right"])
        self.output_dir = Path(cfg["output_dir"])

        self.pairs = discover_pairs(self.dir_a, self.dir_b)
        self.alignments: dict[str, tuple[int, int]] = {}
        self._load_jobs_alignments_if_present()
        self._idx_initialized = False
        self.idx = 0

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Dual camera merge — queue")
        self.root.geometry("1100x720")
        self.root.minsize(900, 560)

        self.path_a = ctk.StringVar(value="")
        self.path_b = ctk.StringVar(value="")
        self.out_path = ctk.StringVar(value="")

        self.n_a = 0
        self.n_b = 0
        self.fps_a = 25.0
        self.fps_b = 25.0

        self.start_a = ctk.IntVar(value=0)
        self.start_b = ctk.IntVar(value=0)
        # Index k along the merged output [0 … L-1]: both streams show (start_a+k, start_b+k).
        self.preview_k = ctk.IntVar(value=0)

        self._drag_axis: Optional[str] = None
        self._preview_photo = None

        self.queue_var = ctk.StringVar(value="")
        self.config_var = ctk.StringVar(value="")

        scroll = ctk.CTkScrollableFrame(self.root, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=0, pady=0)
        self._scroll = scroll

        header = ctk.CTkLabel(
            scroll,
            text="Cam3 left (transpose 1) + Cam2 right (transpose 2) — intersection length",
            anchor="w",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(fill="x", padx=14, pady=(12, 4))

        cfg_line = ctk.CTkLabel(
            scroll,
            textvariable=self.config_var,
            anchor="w",
            font=("Segoe UI", 11),
            text_color=("gray70", "gray60"),
        )
        cfg_line.pack(fill="x", padx=14, pady=(0, 4))

        queue_line = ctk.CTkLabel(
            scroll,
            textvariable=self.queue_var,
            anchor="w",
            font=("Segoe UI", 13, "bold"),
        )
        queue_line.pack(fill="x", padx=14, pady=(0, 8))

        nav = ctk.CTkFrame(scroll, fg_color="transparent")
        nav.pack(fill="x", padx=14, pady=4)
        ctk.CTkButton(nav, text="← Previous", width=120, command=lambda: self._navigate(-1)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(nav, text="Next →", width=120, command=lambda: self._navigate(1)).pack(side="left", padx=0)
        ctk.CTkLabel(nav, text="  Shortcuts: [  prev   ]  next   |   Ctrl+←  Ctrl+→", text_color="gray60").pack(
            side="left", padx=16
        )

        jump_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        jump_fr.pack(fill="x", padx=14, pady=(4, 2))
        ctk.CTkLabel(jump_fr, text="跳到片段:", width=72, anchor="w").pack(side="left")
        clip_names = [p[2] for p in self.pairs] if self.pairs else [""]
        self._suppress_combo_nav = False
        self.clip_combo = ctk.CTkComboBox(
            jump_fr,
            values=clip_names,
            width=320,
            command=self._on_clip_combo,
        )
        self.clip_combo.pack(side="left", padx=4)
        if not self.pairs:
            self.clip_combo.configure(state="disabled")

        hint = ctk.CTkLabel(
            scroll,
            text="Adjust start frames; 「保存合并配置」写入全部片段。只想改一段时用「仅更新当前片段」。"
            " 合并执行: MergeDualCameraVideos.py → output_dir。",
            anchor="w",
            font=("Segoe UI", 12),
            text_color=("gray70", "gray65"),
        )
        hint.pack(fill="x", padx=14, pady=(4, 8))

        path_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        path_fr.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(path_fr, text="A (cam3):", width=72, anchor="w").pack(side="left")
        ctk.CTkEntry(path_fr, textvariable=self.path_a, state="readonly").pack(side="left", padx=4, fill="x", expand=True)

        path_fr2 = ctk.CTkFrame(scroll, fg_color="transparent")
        path_fr2.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(path_fr2, text="B (cam2):", width=72, anchor="w").pack(side="left")
        ctk.CTkEntry(path_fr2, textvariable=self.path_b, state="readonly").pack(side="left", padx=4, fill="x", expand=True)

        out_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        out_fr.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(out_fr, text="Output:", width=72, anchor="w").pack(side="left")
        ctk.CTkEntry(out_fr, textvariable=self.out_path, state="readonly").pack(side="left", padx=4, fill="x", expand=True)

        # Preview: horizontally centered; natural height (scroll handles overflow)
        prev_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        prev_fr.pack(fill="x", padx=14, pady=(8, 6))
        self.preview_label = ctk.CTkLabel(
            prev_fr,
            text="Preview (merged output, 1/4 size) — use slider or timeline row M below",
            anchor="center",
        )
        self.preview_label.pack(anchor="center", pady=4)

        preview_ctl = ctk.CTkFrame(scroll, fg_color="transparent")
        preview_ctl.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            preview_ctl,
            text="合并播放位置 k（两路同一时刻 | output frame 0…L−1）:",
            width=340,
            anchor="w",
        ).pack(side="left", padx=(0, 8))
        self.preview_slider = ctk.CTkSlider(
            preview_ctl, from_=0, to=1, number_of_steps=1, command=self._on_preview_slider
        )
        self.preview_slider.pack(side="left", fill="x", expand=True, padx=4)
        self.entry_preview_k = ctk.CTkEntry(preview_ctl, width=72)
        self.entry_preview_k.pack(side="left", padx=4)
        self.entry_preview_k.bind("<Return>", lambda e: self._apply_entry_preview_k())
        self.entry_preview_k.bind("<FocusOut>", lambda e: self._apply_entry_preview_k())
        self.lbl_preview_k = ctk.CTkLabel(preview_ctl, text="", width=120, anchor="w", text_color="gray70")
        self.lbl_preview_k.pack(side="left", padx=8)

        self.stats_var = ctk.StringVar(value="")
        ctk.CTkLabel(scroll, textvariable=self.stats_var, anchor="w", font=("Segoe UI", 13, "bold")).pack(
            fill="x", padx=14, pady=(4, 4)
        )

        # Timelines: A / B = alignment starts; row M = scrub merged output position k (same as slider above)
        self.timeline_h = 152
        self._y_split_ab = 52
        self._y_split_m = 100
        self.canvas = ctk.CTkCanvas(scroll, height=self.timeline_h, bg="#1a1a1e", highlightthickness=0)
        self.canvas.pack(fill="x", padx=14, pady=(2, 6))
        self.canvas.bind("<Configure>", lambda e: self._redraw_timelines())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.slider_a = ctk.CTkSlider(scroll, from_=0, to=1, number_of_steps=1, command=self._on_slider_a)
        self.slider_a.pack(fill="x", padx=14, pady=(2, 2))
        row_a = ctk.CTkFrame(scroll, fg_color="transparent")
        row_a.pack(fill="x", padx=14)
        ctk.CTkLabel(row_a, text="Start frame A", width=100, anchor="w").pack(side="left")
        self.entry_a = ctk.CTkEntry(row_a, width=100)
        self.entry_a.pack(side="left", padx=4)
        self.entry_a.bind("<Return>", lambda e: self._apply_entry_a())
        self.entry_a.bind("<FocusOut>", lambda e: self._apply_entry_a())
        self.lbl_a = ctk.CTkLabel(row_a, text="", anchor="w")
        self.lbl_a.pack(side="left", padx=12)

        self.slider_b = ctk.CTkSlider(scroll, from_=0, to=1, number_of_steps=1, command=self._on_slider_b)
        self.slider_b.pack(fill="x", padx=14, pady=(4, 2))
        row_b = ctk.CTkFrame(scroll, fg_color="transparent")
        row_b.pack(fill="x", padx=14)
        ctk.CTkLabel(row_b, text="Start frame B", width=100, anchor="w").pack(side="left")
        self.entry_b = ctk.CTkEntry(row_b, width=100)
        self.entry_b.pack(side="left", padx=4)
        self.entry_b.bind("<Return>", lambda e: self._apply_entry_b())
        self.entry_b.bind("<FocusOut>", lambda e: self._apply_entry_b())
        self.lbl_b = ctk.CTkLabel(row_b, text="", anchor="w")
        self.lbl_b.pack(side="left", padx=12)

        btn_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_fr.pack(fill="x", padx=14, pady=(10, 20))
        ctk.CTkButton(btn_fr, text="Refresh preview", command=self._refresh_preview).pack(side="left", padx=4)
        ctk.CTkButton(
            btn_fr,
            text="保存合并配置",
            fg_color="#2d6a9f",
            hover_color="#255a87",
            command=lambda: self._save_jobs(and_next=False),
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            btn_fr,
            text="保存并下一段",
            fg_color="#1e5580",
            hover_color="#18486b",
            command=lambda: self._save_jobs(and_next=True),
        ).pack(side="left", padx=4)
        self.merge_status = ctk.StringVar(value="")
        ctk.CTkLabel(btn_fr, textvariable=self.merge_status, anchor="w").pack(side="left", padx=12)

        single_fr = ctk.CTkFrame(scroll, fg_color="transparent")
        single_fr.pack(fill="x", padx=14, pady=(0, 16))
        ctk.CTkButton(
            single_fr,
            text="仅更新当前片段到配置",
            width=200,
            fg_color="#5c4d3a",
            hover_color="#4a3f32",
            command=self._save_current_clip_only,
        ).pack(side="left", padx=4)
        ctk.CTkLabel(
            single_fr,
            text="只改写 dual_camera_merge_jobs.json 里当前文件名对应的一条，其它片段不动（需已存在 jobs 文件）。",
            anchor="w",
            text_color="gray60",
            font=("Segoe UI", 11),
        ).pack(side="left", padx=12)

        self._update_config_label()
        self.root.bind("<Control-Left>", lambda e: self._navigate(-1))
        self.root.bind("<Control-Right>", lambda e: self._navigate(1))
        self.root.bind("<bracketleft>", lambda e: self._navigate(-1))
        self.root.bind("<bracketright>", lambda e: self._navigate(1))

        if not self.pairs:
            self.stats_var.set("No matching *.mp4 pairs in the two folders (same file name in each). Check paths in JSON.")
            self.queue_var.set("Queue: empty")
        else:
            self._apply_index(0)

        self.root.after(150, self._redraw_timelines)

    def _update_config_label(self) -> None:
        self.config_var.set(
            f"Paths JSON: {self.config_path}  |  merge plan: {self.jobs_path}  |  out: {self.output_dir}"
        )

    def _load_jobs_alignments_if_present(self) -> None:
        if not self.jobs_path.is_file():
            return
        try:
            data = load_jobs_file(self.jobs_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return
        self.alignments.update(alignments_from_jobs(data))

    def _persist_current_alignment(self) -> None:
        if not self.pairs:
            return
        _, _, name = self.pairs[self.idx]
        self.alignments[name] = (self.start_a.get(), self.start_b.get())

    def _save_jobs(self, and_next: bool = False) -> None:
        self._persist_current_alignment()
        clips = clips_from_alignments(self.pairs, self.alignments)
        payload = build_jobs_payload(
            str(self.dir_a.resolve()),
            str(self.dir_b.resolve()),
            str(self.output_dir.resolve()),
            clips,
        )
        try:
            save_jobs_file(self.jobs_path, payload)
        except OSError as e:
            self.merge_status.set(f"写入失败: {e}")
            return
        self.merge_status.set(f"已保存: {self.jobs_path}")
        self._update_config_label()
        if and_next and self.pairs:
            self._navigate(1)

    def _save_current_clip_only(self) -> None:
        """Patch only the current clip in dual_camera_merge_jobs.json; leave other entries unchanged."""
        self._persist_current_alignment()
        if not self.pairs:
            self.merge_status.set("队列为空。")
            return
        name = self.pairs[self.idx][2]
        sa, sb = self.alignments[name]
        if not self.jobs_path.is_file():
            self.merge_status.set("尚无 jobs 文件：请先点「保存合并配置」生成 dual_camera_merge_jobs.json。")
            return
        try:
            data = load_jobs_file(self.jobs_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.merge_status.set(f"无法读取 jobs: {e}")
            return
        data["dir_video_a_cam3_left"] = str(self.dir_a.resolve())
        data["dir_video_b_cam2_right"] = str(self.dir_b.resolve())
        data["output_dir"] = str(self.output_dir.resolve())
        clips = data["clips"]
        found = False
        for c in clips:
            if isinstance(c, dict) and c.get("filename") == name:
                c["start_frame_a"] = int(sa)
                c["start_frame_b"] = int(sb)
                found = True
                break
        if not found:
            clips.append({"filename": name, "start_frame_a": int(sa), "start_frame_b": int(sb)})
            order = {p[2]: i for i, p in enumerate(self.pairs)}
            clips.sort(key=lambda x: order.get(str(x.get("filename", "")), 9999))
        try:
            save_jobs_file(self.jobs_path, data)
        except OSError as e:
            self.merge_status.set(f"写入失败: {e}")
            return
        self.merge_status.set(f"已仅更新: {name}")
        self._update_config_label()

    def _sync_clip_combo(self) -> None:
        if not self.pairs or not hasattr(self, "clip_combo"):
            return
        name = self.pairs[self.idx][2]
        self._suppress_combo_nav = True
        try:
            self.clip_combo.set(name)
        finally:
            self._suppress_combo_nav = False

    def _on_clip_combo(self, choice: str) -> None:
        if self._suppress_combo_nav or not self.pairs:
            return
        for i, p in enumerate(self.pairs):
            if p[2] == choice:
                self._apply_index(i)
                return

    def _navigate(self, delta: int) -> None:
        if not self.pairs:
            return
        n = len(self.pairs)
        self._apply_index((self.idx + delta) % n)

    def _apply_index(self, new_idx: int) -> None:
        if not self.pairs:
            return
        if self._idx_initialized:
            _, _, cur_name = self.pairs[self.idx]
            self.alignments[cur_name] = (self.start_a.get(), self.start_b.get())
        self._idx_initialized = True
        self.idx = max(0, min(len(self.pairs) - 1, new_idx))
        pa, pb, name = self.pairs[self.idx]
        self.path_a.set(pa)
        self.path_b.set(pb)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_path.set(str((self.output_dir / name).resolve()))
        self.queue_var.set(f"Clip {self.idx + 1} / {len(self.pairs)} — {name}")
        self.root.title(f"Dual camera merge — {name} ({self.idx + 1}/{len(self.pairs)})")
        self._load_metadata_for_current()
        self._sync_clip_combo()

    def _load_metadata_for_current(self) -> None:
        pa, pb = self.path_a.get().strip(), self.path_b.get().strip()
        if not pa or not pb:
            return
        ia = probe_video(pa)
        ib = probe_video(pb)
        if not ia or not ib:
            self.stats_var.set("Could not read video metadata.")
            return
        self.n_a, self.fps_a, _, _ = ia
        self.n_b, self.fps_b, _, _ = ib
        clip_name = self.pairs[self.idx][2]
        sa, sb = self.alignments.get(clip_name, (0, 0))
        self.start_a.set(sa)
        self.start_b.set(sb)
        self.preview_k.set(0)
        self.slider_a.configure(from_=0, to=max(0, self.n_a - 1), number_of_steps=max(1, self.n_a - 1))
        self.slider_b.configure(from_=0, to=max(0, self.n_b - 1), number_of_steps=max(1, self.n_b - 1))
        self.slider_a.set(sa)
        self.slider_b.set(sb)
        self._sync_entries_from_vars()
        self._update_stats()
        self._redraw_timelines()

    def _on_slider_a(self, _v: Optional[float] = None) -> None:
        self.start_a.set(int(round(float(self.slider_a.get()))))
        self._sync_entries_from_vars()
        self._update_stats()
        self._redraw_timelines()

    def _on_slider_b(self, _v: Optional[float] = None) -> None:
        self.start_b.set(int(round(float(self.slider_b.get()))))
        self._sync_entries_from_vars()
        self._update_stats()
        self._redraw_timelines()

    def _on_preview_slider(self, _v: Optional[float] = None) -> None:
        L = self._intersection_len()
        if L <= 0:
            return
        max_k = L - 1
        k = int(round(float(self.preview_slider.get())))
        k = max(0, min(max_k, k))
        self.preview_k.set(k)
        self.preview_slider.set(k)
        self._sync_preview_entry()
        self._redraw_timelines()
        self._refresh_preview()

    def _sync_preview_entry(self) -> None:
        self.entry_preview_k.delete(0, "end")
        self.entry_preview_k.insert(0, str(self.preview_k.get()))

    def _apply_entry_preview_k(self) -> None:
        try:
            v = int(self.entry_preview_k.get().strip())
        except ValueError:
            self._sync_preview_entry()
            return
        L = self._intersection_len()
        if L <= 0:
            self.preview_k.set(0)
            self._sync_preview_entry()
            return
        v = max(0, min(L - 1, v))
        self.preview_k.set(v)
        self.preview_slider.set(v)
        self._redraw_timelines()
        self._refresh_preview()

    def _intersection_len(self) -> int:
        if self.n_a <= 0 or self.n_b <= 0:
            return 0
        return intersection_length(self.n_a, self.n_b, self.start_a.get(), self.start_b.get())

    def _update_preview_slider_range(self) -> None:
        L = self._intersection_len()
        if L <= 0:
            self.preview_slider.configure(from_=0, to=1, number_of_steps=1)
            self.preview_slider.set(0)
            self.preview_k.set(0)
            self._sync_preview_entry()
            self.lbl_preview_k.configure(text="")
            return
        max_k = L - 1
        self.preview_slider.configure(from_=0, to=max_k, number_of_steps=max(1, max_k))
        pk = max(0, min(max_k, self.preview_k.get()))
        self.preview_k.set(pk)
        self.preview_slider.set(pk)
        self._sync_preview_entry()
        self.lbl_preview_k.configure(text=f"L={L}")

    def _sync_entries_from_vars(self) -> None:
        self.entry_a.delete(0, "end")
        self.entry_a.insert(0, str(self.start_a.get()))
        self.entry_b.delete(0, "end")
        self.entry_b.insert(0, str(self.start_b.get()))

    def _apply_entry_a(self) -> None:
        try:
            v = int(self.entry_a.get().strip())
        except ValueError:
            self._sync_entries_from_vars()
            return
        if self.n_a > 0:
            v = max(0, min(self.n_a - 1, v))
        self.start_a.set(v)
        self.slider_a.set(v)
        self._update_stats()
        self._redraw_timelines()

    def _apply_entry_b(self) -> None:
        try:
            v = int(self.entry_b.get().strip())
        except ValueError:
            self._sync_entries_from_vars()
            return
        if self.n_b > 0:
            v = max(0, min(self.n_b - 1, v))
        self.start_b.set(v)
        self.slider_b.set(v)
        self._update_stats()
        self._redraw_timelines()

    def _update_stats(self) -> None:
        if self.n_a <= 0 or self.n_b <= 0:
            self.stats_var.set("")
            return
        sa, sb = self.start_a.get(), self.start_b.get()
        L = intersection_length(self.n_a, self.n_b, sa, sb)
        fps = self.fps_a
        sec = L / fps if fps > 0 else 0.0
        self.lbl_a.configure(text=f"frames 0…{self.n_a - 1}")
        self.lbl_b.configure(text=f"frames 0…{self.n_b - 1}")
        out_file = Path(self.out_path.get())
        exists = " (file already exists)" if out_file.is_file() else ""
        if L <= 0:
            self.stats_var.set("Intersection: 0 frames — move starts so both tracks still have footage.")
        else:
            self.stats_var.set(
                f"Intersection: {L} frames (~{sec:.2f} s @ {fps:.2f} fps) | "
                f"A [{sa}…{sa + L - 1}] | B [{sb}…{sb + L - 1}] | "
                f"preview uses same k on both → A[{sa}+k], B[{sb}+k]{exists}"
            )
        self._update_preview_slider_range()
        if L > 0:
            self._refresh_preview()

    def _canvas_width(self) -> int:
        self.root.update_idletasks()
        w = self.canvas.winfo_width()
        return max(w, 700)

    def _frame_from_x(self, x: float, n_total: int, pad: int, draw_w: int) -> int:
        if n_total <= 1:
            return 0
        x0 = pad
        xf = pad + draw_w
        if x <= x0:
            return 0
        if x >= xf:
            return n_total - 1
        t = (x - x0) / max(1e-6, (xf - x0))
        return int(round(t * (n_total - 1)))

    def _x_from_frame(self, frame: int, n_total: int, pad: int, draw_w: int) -> float:
        if n_total <= 1:
            return pad
        t = frame / max(1e-6, (n_total - 1))
        return pad + t * draw_w

    def _on_canvas_press(self, e) -> None:
        if self.n_a <= 0:
            return
        w = self._canvas_width()
        pad = 24
        draw_w = w - 2 * pad
        y = e.y
        L = self._intersection_len()
        if y < self._y_split_ab:
            self._drag_axis = "a"
            self.start_a.set(self._frame_from_x(e.x, self.n_a, pad, draw_w))
            self.slider_a.set(self.start_a.get())
            self._sync_entries_from_vars()
            self._update_stats()
            self._redraw_timelines()
        elif y < self._y_split_m:
            self._drag_axis = "b"
            self.start_b.set(self._frame_from_x(e.x, self.n_b, pad, draw_w))
            self.slider_b.set(self.start_b.get())
            self._sync_entries_from_vars()
            self._update_stats()
            self._redraw_timelines()
        else:
            self._drag_axis = "merged"
            if L > 0:
                k = self._frame_from_x(e.x, L, pad, draw_w) if L > 1 else 0
                self.preview_k.set(k)
                self.preview_slider.set(k)
                self._sync_preview_entry()
                self._redraw_timelines()
                self._refresh_preview()

    def _on_canvas_drag(self, e) -> None:
        if self._drag_axis is None or self.n_a <= 0:
            return
        w = self._canvas_width()
        pad = 24
        draw_w = w - 2 * pad
        L = self._intersection_len()
        if self._drag_axis == "a":
            self.start_a.set(self._frame_from_x(e.x, self.n_a, pad, draw_w))
            self.slider_a.set(self.start_a.get())
            self._sync_entries_from_vars()
            self._update_stats()
            self._redraw_timelines()
        elif self._drag_axis == "b":
            self.start_b.set(self._frame_from_x(e.x, self.n_b, pad, draw_w))
            self.slider_b.set(self.start_b.get())
            self._sync_entries_from_vars()
            self._update_stats()
            self._redraw_timelines()
        elif self._drag_axis == "merged" and L > 0:
            k = self._frame_from_x(e.x, L, pad, draw_w) if L > 1 else 0
            self.preview_k.set(k)
            self.preview_slider.set(k)
            self._sync_preview_entry()
            self._redraw_timelines()
            self._refresh_preview()

    def _on_canvas_release(self, _e) -> None:
        self._drag_axis = None
        self._refresh_preview()

    def _redraw_timelines(self, _event=None) -> None:
        self.canvas.delete("all")
        w = self._canvas_width()
        h = self.timeline_h
        pad = 24
        draw_w = w - 2 * pad

        if self.n_a <= 0:
            self.canvas.create_text(
                w // 2,
                h // 2,
                text="No clips or metadata failed.",
                fill="#888",
                font=("Segoe UI", 11),
            )
            return

        sa, sb = self.start_a.get(), self.start_b.get()
        La = intersection_length(self.n_a, self.n_b, sa, sb)
        fs = ("Segoe UI", 9)
        pk = max(0, min(La - 1, self.preview_k.get())) if La > 0 else 0

        y1, y2 = 8, 46
        self.canvas.create_rectangle(pad, y1, pad + draw_w, y2, outline="#555", width=1)
        if La > 0:
            xa0 = self._x_from_frame(sa, self.n_a, pad, draw_w)
            xa1 = self._x_from_frame(sa + La - 1, self.n_a, pad, draw_w)
            self.canvas.create_rectangle(xa0, y1 + 1, max(xa0 + 3, xa1), y2 - 1, fill="#3d7dae", outline="")
        xh = self._x_from_frame(sa, self.n_a, pad, draw_w)
        self.canvas.create_line(xh, y1, xh, y2, fill="#ffcc66", width=2)
        self.canvas.create_text(pad, 0, text=f"A — align start {sa}", anchor="nw", fill="#ccc", font=fs)

        y1b, y2b = 50, 88
        self.canvas.create_rectangle(pad, y1b, pad + draw_w, y2b, outline="#555", width=1)
        if La > 0:
            xb0 = self._x_from_frame(sb, self.n_b, pad, draw_w)
            xb1 = self._x_from_frame(sb + La - 1, self.n_b, pad, draw_w)
            self.canvas.create_rectangle(xb0, y1b + 1, max(xb0 + 3, xb1), y2b - 1, fill="#3d8f5a", outline="")
        xhb = self._x_from_frame(sb, self.n_b, pad, draw_w)
        self.canvas.create_line(xhb, y1b, xhb, y2b, fill="#ffcc66", width=2)
        self.canvas.create_text(pad, 48, text=f"B — align start {sb}", anchor="nw", fill="#ccc", font=fs)

        ym1, ym2 = 92, h - 6
        self.canvas.create_rectangle(pad, ym1, pad + draw_w, ym2, outline="#666", width=1)
        if La > 0:
            xpk = self._x_from_frame(pk, La, pad, draw_w)
            if La > 1 and xpk > pad + 2:
                self.canvas.create_rectangle(pad, ym1 + 2, xpk, ym2 - 2, fill="#3a3a44", outline="")
            self.canvas.create_line(xpk, ym1, xpk, ym2, fill="#88ddff", width=2)
            km = max(0, La - 1)
            self.canvas.create_text(
                pad + 4, ym1 + 2, text=f"k={pk}/{km}", anchor="nw", fill="#9cf", font=("Segoe UI", 8)
            )

    def _refresh_preview(self) -> None:
        pa, pb = self.path_a.get().strip(), self.path_b.get().strip()
        if not pa or not pb or self.n_a <= 0:
            return
        sa, sb = self.start_a.get(), self.start_b.get()
        L = intersection_length(self.n_a, self.n_b, sa, sb)
        if L <= 0:
            self.preview_label.configure(image="", text="No intersection to preview.")
            return
        k = max(0, min(L - 1, self.preview_k.get()))
        fa_idx = sa + k
        fb_idx = sb + k
        cap_a = cv2.VideoCapture(pa)
        cap_b = cv2.VideoCapture(pb)
        if not cap_a.isOpened() or not cap_b.isOpened():
            cap_a.release()
            cap_b.release()
            return
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, fa_idx)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, fb_idx)
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        cap_a.release()
        cap_b.release()
        if not ok_a or not ok_b or fa is None or fb is None:
            self.preview_label.configure(image="", text="Could not read preview frames.")
            return
        a_rot = cv2.rotate(fa, cv2.ROTATE_90_CLOCKWISE)
        b_rot = cv2.rotate(fb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if a_rot.shape[0] != b_rot.shape[0]:
            nh = a_rot.shape[0]
            nw = int(b_rot.shape[1] * (nh / b_rot.shape[0]))
            b_rot = cv2.resize(b_rot, (nw, nh), interpolation=cv2.INTER_AREA)
        merged = np.hstack([a_rot, b_rot])
        h0, w0 = merged.shape[:2]
        # Display at 1/4 of merged frame size (horizontal-centered in UI)
        nw, nh = max(1, int(round(w0 * 0.25))), max(1, int(round(h0 * 0.25)))
        merged = cv2.resize(merged, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
        im = self.Image.fromarray(rgb)
        from customtkinter import CTkImage

        self._preview_photo = CTkImage(light_image=im, dark_image=im, size=im.size)
        self.preview_label.configure(image=self._preview_photo, text="")

    def run(self) -> None:
        self.root.mainloop()


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "dual_camera_merge_paths.json"


def ensure_config_file(cfg_path: Path) -> None:
    """If JSON is missing, copy from dual_camera_merge_paths.example.json (first-run bootstrap)."""
    if cfg_path.is_file():
        return
    example = cfg_path.parent / "dual_camera_merge_paths.example.json"
    if not example.is_file():
        print(
            f"Config not found: {cfg_path}\n"
            f"Also missing example file: {example}",
            file=sys.stderr,
        )
        sys.exit(1)
    shutil.copy2(example, cfg_path)
    print(
        f"Created {cfg_path} from dual_camera_merge_paths.example.json\n"
        "Edit the three paths if your folders differ, then run again.",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-camera merge GUI: queue clips from two folders (see dual_camera_merge_paths.json)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config (default: dual_camera_merge_paths.json next to this script)",
    )
    args = parser.parse_args()

    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print("Install customtkinter (pixi env location-tracker).", file=sys.stderr)
        sys.exit(1)

    cfg_path = args.config if args.config is not None else default_config_path()
    ensure_config_file(cfg_path)

    try:
        load_paths_config(cfg_path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Invalid config: {e}", file=sys.stderr)
        sys.exit(1)

    app = DualAxisMergeApp(cfg_path)
    app.run()


if __name__ == "__main__":
    main()
