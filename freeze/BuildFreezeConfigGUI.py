#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modern GUI to create or edit a YAML config for freeze analysis."""

import os
import argparse
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from AutoFreezeCalibration import auto_calibrate_motion_cutoff


class CropBox:
    def __init__(self, crop):
        self.data = {
            "x0": [crop["x0"]],
            "x1": [crop["x1"]],
            "y0": [crop["y0"]],
            "y1": [crop["y1"]],
        }


class FreezePreviewPlayer:
    def __init__(self, app, params, motion, freezing):
        self.app = app
        self.params = params
        self.motion = motion
        self.freezing = freezing
        self.index = 0
        self.playing = True
        self.seeking = False
        self.photo = None
        self.cap = cv2.VideoCapture(params["video_path"])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, params["sample_start"])

        self.window = ctk.CTkToplevel(app.root)
        self.window.title("Freeze Preview: Last 60 Seconds")
        self.window.geometry("920x760")
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        controls = ctk.CTkFrame(self.window)
        controls.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(controls, text="freeze_threshold").pack(side="left", padx=(8, 4))
        self.threshold_entry = ctk.CTkEntry(controls, width=90)
        self.threshold_entry.insert(0, f"{params['freeze_threshold']:g}")
        self.threshold_entry.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(controls, text="min_duration_seconds").pack(side="left", padx=(0, 4))
        self.duration_entry = ctk.CTkEntry(controls, width=90)
        self.duration_entry.insert(0, f"{params['min_duration_seconds']:g}")
        self.duration_entry.pack(side="left", padx=(0, 12))

        ctk.CTkButton(controls, text="Re-run Analysis & Replay", width=170, command=self._rerun_and_replay).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="Apply To Main GUI", width=150, command=self._apply_to_main).pack(side="left", padx=(0, 8))
        self.play_button = ctk.CTkButton(controls, text="Pause", width=90, command=self._toggle_play)
        self.play_button.pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="Restart", width=90, command=self._restart).pack(side="left")

        axis_controls = ctk.CTkFrame(self.window)
        axis_controls.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(axis_controls, text="Plot Y min").pack(side="left", padx=(8, 4))
        self.ymin_entry = ctk.CTkEntry(axis_controls, width=90)
        self.ymin_entry.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(axis_controls, text="Y max").pack(side="left", padx=(0, 4))
        self.ymax_entry = ctk.CTkEntry(axis_controls, width=90)
        self.ymax_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(axis_controls, text="Apply Y Axis", width=120, command=self._apply_y_axis).pack(side="left", padx=(0, 8))
        ctk.CTkButton(axis_controls, text="Auto Y Axis", width=120, command=self._auto_y_axis).pack(side="left")

        self.video_label = ctk.CTkLabel(self.window, text="")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        seek_frame = ctk.CTkFrame(self.window)
        seek_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.seek_slider = ctk.CTkSlider(
            seek_frame,
            from_=0,
            to=max(1, len(self.motion) - 1),
            number_of_steps=max(1, len(self.motion) - 1),
            command=self._on_seek_drag,
        )
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=(8, 10), pady=8)
        self.time_label = ctk.CTkLabel(seek_frame, text="00:00 / 00:00", width=120)
        self.time_label.pack(side="left", padx=(0, 8))
        self.seek_slider.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_slider.bind("<ButtonRelease-1>", self._on_seek_release)

        plot_frame = ctk.CTkFrame(self.window)
        plot_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.fig = Figure(figsize=(8.6, 2.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill="x", expand=True, padx=6, pady=6)
        self.motion_line = None
        self.threshold_line = None
        self.current_line = None
        self._init_motion_plot()

        self.info_label = ctk.CTkLabel(self.window, text="")
        self.info_label.pack(anchor="w", padx=12, pady=(0, 10))

        self._update_stats_label()
        self._tick()

    def _close(self):
        self.playing = False
        self.cap.release()
        self.window.destroy()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def _restart(self):
        self._seek_to_index(0)
        self.playing = True
        self.play_button.configure(text="Pause")

    def _rerun_and_replay(self):
        try:
            from FreezeAnalysis_Functions import Measure_Freezing

            threshold = float(self.threshold_entry.get().strip())
            duration = float(self.duration_entry.get().strip())
            min_duration_frames = int(round(duration * self.params["fps"]))
            self.freezing = Measure_Freezing(self.motion, threshold, MinDuration=min_duration_frames)
            self.params["freeze_threshold"] = threshold
            self.params["min_duration_seconds"] = duration
            self._update_stats_label()
            self._init_motion_plot()
            self._restart()
        except Exception as e:
            messagebox.showerror("Re-run Failed", str(e))

    def _apply_to_main(self):
        self.app._set_text(self.app.freeze_threshold_entry, self.threshold_entry.get().strip())
        self.app._set_text(self.app.min_duration_entry, self.duration_entry.get().strip())
        self.app._log("[OK] Preview freeze parameters applied to main GUI.")

    def _update_stats_label(self):
        freeze_percent = float(self.freezing.mean()) if len(self.freezing) else 0.0
        self.info_label.configure(
            text=(
                f"{self.params['file_name']} | final-minute preview | "
                f"freeze={freeze_percent:.2f}% | "
                f"motion_cutoff={self.params['motion_cutoff']:.4g} | "
                f"threshold={self.params['freeze_threshold']:g} | "
                f"min_duration={self.params['min_duration_seconds']:g}s"
            )
        )

    def _init_motion_plot(self):
        self.ax.clear()
        if len(self.motion) == 0:
            self.plot_canvas.draw()
            return

        times = np.arange(len(self.motion)) / max(self.params["fps"], 1)
        self.motion_line, = self.ax.plot(times, self.motion, color="#4C78A8", linewidth=1.2, label="Motion")
        self.threshold_line = self.ax.axhline(
            self.params["freeze_threshold"],
            color="#E45756",
            linestyle="--",
            linewidth=1.4,
            label="freeze_threshold",
        )
        self.current_line = self.ax.axvline(0, color="#222222", linewidth=1.2, label="current frame")
        self.ax.set_title("Motion Trace With Freeze Threshold")
        self.ax.set_xlabel("Preview time (s)")
        self.ax.set_ylabel("Motion pixels")
        y_max = max(float(np.nanmax(self.motion)) if len(self.motion) else 0.0, float(self.params["freeze_threshold"]), 1.0)
        y_max *= 1.1
        self.ax.set_ylim(0, y_max)
        if not self.ymin_entry.get().strip() and not self.ymax_entry.get().strip():
            self.ymin_entry.insert(0, "0")
            self.ymax_entry.insert(0, f"{y_max:.4g}")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.plot_canvas.draw()

    def _apply_y_axis(self):
        try:
            y_min = float(self.ymin_entry.get().strip())
            y_max = float(self.ymax_entry.get().strip())
            if y_max <= y_min:
                raise ValueError("Y max must be greater than Y min.")
            self.ax.set_ylim(y_min, y_max)
            self.plot_canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Y Axis Failed", str(e))

    def _auto_y_axis(self):
        y_max = max(float(np.nanmax(self.motion)) if len(self.motion) else 0.0, float(self.params["freeze_threshold"]), 1.0)
        y_max *= 1.1
        self.ymin_entry.delete(0, "end")
        self.ymin_entry.insert(0, "0")
        self.ymax_entry.delete(0, "end")
        self.ymax_entry.insert(0, f"{y_max:.4g}")
        self.ax.set_ylim(0, y_max)
        self.plot_canvas.draw_idle()

    def _update_motion_cursor(self):
        if self.current_line is None:
            return
        current_time = self.index / max(self.params["fps"], 1)
        self.current_line.set_xdata([current_time, current_time])
        self.plot_canvas.draw_idle()

    @staticmethod
    def _format_seconds(seconds):
        seconds = max(0, float(seconds))
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _update_seek_ui(self):
        if len(self.motion) == 0:
            return
        idx = min(max(int(self.index), 0), len(self.motion) - 1)
        if not self.seeking:
            self.seek_slider.set(idx)
        current = idx / max(self.params["fps"], 1)
        total = max(0, (len(self.motion) - 1) / max(self.params["fps"], 1))
        self.time_label.configure(text=f"{self._format_seconds(current)} / {self._format_seconds(total)}")

    def _on_seek_press(self, _event):
        self.seeking = True

    def _on_seek_drag(self, value):
        idx = int(round(float(value)))
        current = idx / max(self.params["fps"], 1)
        total = max(0, (len(self.motion) - 1) / max(self.params["fps"], 1))
        self.time_label.configure(text=f"{self._format_seconds(current)} / {self._format_seconds(total)}")
        if self.current_line is not None:
            self.current_line.set_xdata([current, current])
            self.plot_canvas.draw_idle()

    def _on_seek_release(self, _event):
        idx = int(round(float(self.seek_slider.get())))
        self._seek_to_index(idx)
        self.seeking = False

    def _seek_to_index(self, idx):
        if len(self.motion) == 0:
            return
        idx = min(max(int(idx), 0), len(self.motion) - 1)
        self.index = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.params["sample_start"] + idx)
        self._update_motion_cursor()
        self._update_seek_ui()

    def _draw_overlay(self, frame, idx):
        if self.params["crop"].get("enabled", False):
            crop = self.params["crop"]
            cv2.rectangle(
                frame,
                (int(crop["x0"]), int(crop["y0"])),
                (int(crop["x1"]), int(crop["y1"])),
                (255, 180, 0),
                2,
            )

        motion = self.motion[idx] if idx < len(self.motion) else 0
        is_freezing = idx < len(self.freezing) and self.freezing[idx] > 0
        label = "FREEZING" if is_freezing else "MOVING"
        color = (70, 200, 70) if is_freezing else (60, 120, 255)
        frame_no = self.params["sample_start"] + idx
        text = f"{label} | frame {frame_no} | motion={motion:.1f}"
        cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 680), 58), (0, 0, 0), -1)
        cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame

    def _tick(self):
        if self.playing:
            ret, frame = self.cap.read()
            if not ret or self.index >= len(self.motion):
                self._restart()
                ret, frame = self.cap.read()
            if ret:
                frame = self._draw_overlay(frame, self.index)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                max_width = 860
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), cv2.INTER_AREA)
                image = Image.fromarray(frame)
                self.photo = ImageTk.PhotoImage(image)
                self.video_label.configure(image=self.photo)
                self._update_motion_cursor()
                self._update_seek_ui()
                self.index += 1

        delay_ms = max(15, int(1000 / max(self.params["fps"], 1)))
        if self.window.winfo_exists():
            self.window.after(delay_ms, self._tick)


class FreezeConfigBuilderApp:
    def __init__(self, initial_config_path=None, root=None, embedded=False):
        self.embedded = embedded
        if root is None:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
            self.root.title("Freeze Analysis Config Builder")
            self.root.geometry("1040x860")
            self.root.minsize(920, 680)
        else:
            self.root = root
        self.crop_cfg = {"enabled": False, "x0": 0, "x1": 960, "y0": 0, "y1": 576}
        self.current_config_path = None

        self._build_ui()
        if initial_config_path:
            self._load_yaml_path(initial_config_path)

    def _build_ui(self):
        frame = ctk.CTkScrollableFrame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkButton(top_row, text="Load YAML", width=120, command=self._load_yaml).pack(side="left")
        ctk.CTkButton(top_row, text="Save YAML", width=120, command=self._save_yaml, fg_color="#1F6AA5").pack(side="left", padx=8)
        ctk.CTkButton(top_row, text="Initialize Project", width=160, command=self._initialize_project, fg_color="#2E8B57").pack(side="left", padx=8)

        ctk.CTkLabel(frame, text="Project Videos", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        video_row = ctk.CTkFrame(frame, fg_color="transparent")
        video_row.pack(fill="x", padx=12)
        self.video_dir_entry = ctk.CTkEntry(video_row)
        self.video_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(video_row, text="Browse", width=110, command=self._browse_video_dir).pack(side="left")

        cal_row = ctk.CTkFrame(frame, fg_color="transparent")
        cal_row.pack(fill="x", padx=12, pady=(8, 0))
        self.calibration_video_entry = ctk.CTkEntry(cal_row)
        self.calibration_video_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(cal_row, text="Calibration Video", width=160, command=self._browse_calibration_video).pack(side="left")

        project_section = ctk.CTkFrame(frame)
        project_section.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(project_section, text="Project Settings", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        row1 = ctk.CTkFrame(project_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=4)
        self.file_type_entry = self._labeled_entry(row1, "File Type", "mp4", 80)
        self.start_entry = self._labeled_entry(row1, "Start Frame", "0", 100)
        self.end_entry = self._labeled_entry(row1, "End Frame", "", 100)
        self.dsmpl_entry = self._labeled_entry(row1, "Downsample", "1.0", 100)

        crop_section = ctk.CTkFrame(frame)
        crop_section.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(crop_section, text="Crop", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        crop_row = ctk.CTkFrame(crop_section, fg_color="transparent")
        crop_row.pack(fill="x", padx=10, pady=(0, 8))
        self.crop_enabled_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(crop_row, text="Enable crop", variable=self.crop_enabled_var).pack(side="left", padx=(0, 14))
        ctk.CTkButton(crop_row, text="Select Crop From First Video", command=self._select_crop).pack(side="left")
        self.crop_label = ctk.CTkLabel(crop_row, text="x0=0, x1=960, y0=0, y1=576")
        self.crop_label.pack(side="left", padx=14)

        calibration_section = ctk.CTkFrame(frame)
        calibration_section.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(calibration_section, text="Calibration Parameters", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        row2 = ctk.CTkFrame(calibration_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=4)
        self.motion_cutoff_entry = self._labeled_entry(row2, "motion_cutoff", "", 120)
        self.cal_frames_entry = self._labeled_entry(row2, "cal_frames", "250", 100)
        self.cal_pixels_entry = self._labeled_entry(row2, "cal_pixels", "10000", 110)
        self.sigma_entry = self._labeled_entry(row2, "sigma", "1.0", 90)
        self.percentile_entry = self._labeled_entry(row2, "percentile", "99.99", 100)
        self.cutoff_multiplier_entry = self._labeled_entry(row2, "cutoff_multiplier", "2.8", 130)
        ctk.CTkButton(
            calibration_section,
            text="Auto Calibrate From Empty Video",
            command=self._auto_calibrate,
            fg_color="#2E8B57",
        ).pack(fill="x", padx=10, pady=(4, 8))

        freeze_section = ctk.CTkFrame(frame)
        freeze_section.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(freeze_section, text="Freeze Parameters", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        row3 = ctk.CTkFrame(freeze_section, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=(4, 8))
        self.freeze_threshold_entry = self._labeled_entry(row3, "freeze_threshold", "50", 130)
        self.min_duration_entry = self._labeled_entry(row3, "min_duration_seconds", "0.5", 150)
        self.accept_p_frames_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(row3, text="accept_p_frames", variable=self.accept_p_frames_var).pack(side="left", padx=(10, 16))
        self.save_frame_data_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row3, text="save_frame_data", variable=self.save_frame_data_var).pack(side="left")
        ctk.CTkButton(
            freeze_section,
            text="Preview / Tune Freeze Parameters",
            command=self._preview_freeze_sample,
            fg_color="#7A5AF8",
        ).pack(fill="x", padx=10, pady=(0, 8))

        bins_section = ctk.CTkFrame(frame)
        bins_section.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(bins_section, text="Summary Bins", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        bins_buttons = ctk.CTkFrame(bins_section, fg_color="transparent")
        bins_buttons.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkButton(bins_buttons, text="Insert Bin", width=110, command=self._insert_bin).pack(side="left")
        ctk.CTkButton(bins_buttons, text="Clear Bins", width=110, command=self._clear_bins).pack(side="left", padx=8)
        header = ctk.CTkFrame(bins_section, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(header, text="Name", width=180).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(header, text="Start (s)", width=120).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(header, text="End (s)", width=120).pack(side="left", padx=(0, 8))
        self.bins_table = ctk.CTkScrollableFrame(bins_section, height=220)
        self.bins_table.pack(fill="x", padx=10, pady=(0, 8))
        self.bin_rows = []

        self.status_box = ctk.CTkTextbox(frame, height=150)
        self.status_box.pack(fill="x", padx=12, pady=(6, 12))
        ctk.CTkButton(
            frame,
            text="Save YAML & Run Batch Analysis",
            command=self._save_and_run_batch_analysis,
            fg_color="#D97706",
        ).pack(fill="x", padx=12, pady=(0, 14))
        self._log("Ready. Fill parameters, optionally select crop, then save YAML.")

    def _labeled_entry(self, parent, label, default, width):
        ctk.CTkLabel(parent, text=label).pack(side="left", padx=(0, 6))
        entry = ctk.CTkEntry(parent, width=width)
        entry.insert(0, default)
        entry.pack(side="left", padx=(0, 14))
        return entry

    def _log(self, text):
        self.status_box.insert("end", text + "\n")
        self.status_box.see("end")

    def _browse_video_dir(self):
        path = filedialog.askdirectory(title="Select Behavior Video Directory")
        if path:
            self.video_dir_entry.delete(0, "end")
            self.video_dir_entry.insert(0, os.path.normpath(path))

    def _browse_calibration_video(self):
        path = filedialog.askopenfilename(title="Select Empty-Chamber Calibration Video")
        if path:
            self.calibration_video_entry.delete(0, "end")
            self.calibration_video_entry.insert(0, os.path.normpath(path))

    def _first_video_path(self):
        video_dir = os.path.normpath(self.video_dir_entry.get().strip())
        file_type = self.file_type_entry.get().strip().lstrip(".") or "mp4"
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        for name in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, name)
            if os.path.isfile(path) and name.lower().endswith("." + file_type.lower()):
                return path
        raise FileNotFoundError(f"No .{file_type} videos found in {video_dir}")

    def _select_crop(self):
        try:
            video_path = self._first_video_path()
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Could not read first frame: {video_path}")
            rect = cv2.selectROI("Select Freeze Crop, press Enter to confirm", frame, showCrosshair=True)
            cv2.destroyWindow("Select Freeze Crop, press Enter to confirm")
            x, y, w, h = [int(v) for v in rect]
            if w <= 0 or h <= 0:
                self._log("[INFO] Crop selection skipped.")
                return
            self.crop_cfg = {"enabled": True, "x0": x, "x1": x + w, "y0": y, "y1": y + h}
            self.crop_enabled_var.set(True)
            self._update_crop_label()
            self._log(f"[OK] Crop selected from {os.path.basename(video_path)}.")
        except Exception as e:
            messagebox.showerror("Crop Failed", str(e))
            self._log(f"[ERROR] Crop failed: {e}")

    def _update_crop_label(self):
        self.crop_label.configure(
            text=(
                f"x0={self.crop_cfg['x0']}, x1={self.crop_cfg['x1']}, "
                f"y0={self.crop_cfg['y0']}, y1={self.crop_cfg['y1']}"
            )
        )

    def _parse_optional_int(self, value):
        value = value.strip()
        return None if value == "" else int(float(value))

    def _parse_optional_float(self, value):
        value = value.strip()
        return None if value == "" else float(value)

    def _parse_bins(self):
        if not self.bin_rows:
            return None
        bins = {}
        for row in self.bin_rows:
            name = row["name"].get().strip()
            start_raw = row["start"].get().strip()
            end_raw = row["end"].get().strip()
            if not name and not start_raw and not end_raw:
                continue
            if not name:
                raise ValueError("Every bin needs a name.")
            if not start_raw or not end_raw:
                raise ValueError(f"Bin '{name}' needs both start and end seconds.")
            start = float(start_raw)
            end = float(end_raw)
            if end <= start:
                raise ValueError(f"Bin '{name}' end must be greater than start.")
            bins[name] = [start, end]
        if not bins:
            return None
        return bins

    def _add_bin_row(self, name="", start="", end=""):
        row_frame = ctk.CTkFrame(self.bins_table, fg_color="transparent")
        row_frame.pack(fill="x", pady=2)
        name_entry = ctk.CTkEntry(row_frame, width=180)
        name_entry.insert(0, str(name))
        name_entry.pack(side="left", padx=(0, 8))
        start_entry = ctk.CTkEntry(row_frame, width=120)
        start_entry.insert(0, str(start))
        start_entry.pack(side="left", padx=(0, 8))
        end_entry = ctk.CTkEntry(row_frame, width=120)
        end_entry.insert(0, str(end))
        end_entry.pack(side="left", padx=(0, 8))
        row = {"frame": row_frame, "name": name_entry, "start": start_entry, "end": end_entry}
        ctk.CTkButton(row_frame, text="Remove", width=80, command=lambda: self._remove_bin_row(row)).pack(side="left")
        self.bin_rows.append(row)
        return row

    def _remove_bin_row(self, row):
        if row in self.bin_rows:
            self.bin_rows.remove(row)
        row["frame"].destroy()

    def _next_bin_start(self):
        if not self.bin_rows:
            return 0.0
        last_end = self.bin_rows[-1]["end"].get().strip()
        return float(last_end) if last_end else 0.0

    def _insert_bin(self):
        try:
            start = self._next_bin_start()
            end = start + 60.0
            idx = len(self.bin_rows) + 1
            self._add_bin_row(f"bin_{idx}", f"{start:g}", f"{end:g}")
            self._log(f"[OK] Inserted bin_{idx}: {start:g}-{end:g}s.")
        except Exception as e:
            messagebox.showerror("Insert Bin Failed", str(e))
            self._log(f"[ERROR] Insert bin failed: {e}")

    def _clear_bins(self):
        for row in list(self.bin_rows):
            self._remove_bin_row(row)
        self._log("[OK] Summary bins cleared. Batch summary will use one 'all' bin.")

    def _find_default_calibration_video(self, video_dir):
        candidates = ["empty_box.mp4", "empty.mp4", "calibration.mp4"]
        for name in candidates:
            path = os.path.join(video_dir, name)
            if os.path.isfile(path):
                return path
        for name in sorted(os.listdir(video_dir)):
            lower = name.lower()
            if lower.endswith(".mp4") and ("empty" in lower or "cal" in lower):
                return os.path.join(video_dir, name)
        return ""

    def _initialize_project(self):
        try:
            video_dir = os.path.normpath(self.video_dir_entry.get().strip())
            if not video_dir:
                path = filedialog.askdirectory(title="Select Project Video Directory")
                if not path:
                    return
                video_dir = os.path.normpath(path)
                self._set_text(self.video_dir_entry, video_dir)
            if not os.path.isdir(video_dir):
                raise FileNotFoundError(f"Video directory not found: {video_dir}")

            if not self.calibration_video_entry.get().strip():
                self._set_text(self.calibration_video_entry, self._find_default_calibration_video(video_dir))

            analysis_dir = os.path.join(video_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            out_path = os.path.join(analysis_dir, "project_freeze_config.yml")
            yaml_obj = self._yaml_from_ui()
            yaml_obj["project"]["video_dir"] = video_dir
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            self.current_config_path = out_path
            self._log(f"[OK] Project initialized: {os.path.normpath(out_path)}")
            messagebox.showinfo("Project Initialized", f"Config created:\n{os.path.normpath(out_path)}")
        except Exception as e:
            messagebox.showerror("Initialize Project Failed", str(e))
            self._log(f"[ERROR] Initialize project failed: {e}")

    def _auto_calibrate(self):
        try:
            calibration_video = os.path.normpath(self.calibration_video_entry.get().strip())
            if not calibration_video:
                raise ValueError("Please choose an empty-chamber calibration video first.")
            if not os.path.isfile(calibration_video):
                raise FileNotFoundError(f"Calibration video not found: {calibration_video}")

            crop = dict(self.crop_cfg)
            crop["enabled"] = bool(self.crop_enabled_var.get())
            params = {
                "video_path": calibration_video,
                "start_frame": int(float(self.start_entry.get().strip() or "0")),
                "dsmpl": float(self.dsmpl_entry.get().strip() or "1.0"),
                "crop": crop,
                "cal_frames": int(float(self.cal_frames_entry.get().strip() or "250")),
                "cal_pixels": int(float(self.cal_pixels_entry.get().strip() or "10000")),
                "sigma": float(self.sigma_entry.get().strip() or "1.0"),
                "percentile": float(self.percentile_entry.get().strip() or "99.99"),
                "cutoff_multiplier": float(self.cutoff_multiplier_entry.get().strip() or "2.8"),
                "accept_p_frames": bool(self.accept_p_frames_var.get()),
            }
            self._log("[INFO] Auto calibration started. The GUI will update when it finishes.")
            threading.Thread(target=self._auto_calibrate_worker, args=(params,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Auto Calibration Failed", str(e))
            self._log(f"[ERROR] Auto calibration failed: {e}")

    def _auto_calibrate_worker(self, params):
        try:
            stats = auto_calibrate_motion_cutoff(**params)
            self.root.after(0, self._finish_auto_calibration, stats, None)
        except Exception as e:
            self.root.after(0, self._finish_auto_calibration, None, e)

    def _finish_auto_calibration(self, stats, error):
        if error is not None:
            messagebox.showerror("Auto Calibration Failed", str(error))
            self._log(f"[ERROR] Auto calibration failed: {error}")
            return

        cutoff = stats["motion_cutoff"]
        self._set_text(self.motion_cutoff_entry, f"{cutoff:.6g}")
        self._log(
            "[OK] Auto calibration complete: "
            f"motion_cutoff={cutoff:.6g}, "
            f"P{stats['percentile']}={stats['percentile_value']:.6g}, "
            f"multiplier={stats['cutoff_multiplier']:.3g}, "
            f"mean_diff={stats['average_pixel_difference']:.6g}, "
            f"zero_fraction={stats['zero_fraction']:.3f}, "
            f"frames={stats['frames_used']}, pixels={stats['pixels_sampled']}"
        )
        self._show_calibration_plot(stats)

    def _show_calibration_plot(self, stats):
        plot_window = ctk.CTkToplevel(self.root)
        plot_window.title("Freeze Calibration Plot")
        plot_window.geometry("820x560")
        plot_window.lift()

        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        edges = stats["hist_edges"]
        counts = stats["hist_counts"]
        widths = edges[1:] - edges[:-1]
        ax.bar(edges[:-1], counts, width=widths, align="edge", color="#4C78A8", alpha=0.8)
        ax.axvline(
            stats["percentile_value"],
            color="#F58518",
            linestyle="--",
            linewidth=2,
            label=f"P{stats['percentile']} = {stats['percentile_value']:.3g}",
        )
        ax.axvline(
            stats["motion_cutoff"],
            color="#E45756",
            linestyle="-",
            linewidth=2,
            label=f"motion_cutoff = {stats['motion_cutoff']:.3g}",
        )
        ax.set_title("Empty-Chamber Frame-to-Frame Pixel Difference")
        ax.set_xlabel("Absolute grayscale difference between adjacent frames")
        ax.set_ylabel("Sampled pixel count")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        summary = (
            f"method=percentile_multiplier   "
            f"P{stats['percentile']}={stats['percentile_value']:.4g}   "
            f"multiplier={stats['cutoff_multiplier']:.4g}   "
            f"zero_fraction={stats['zero_fraction']:.3f}   "
            f"frames={stats['frames_used']}   "
            f"pixels/frame={stats['pixels_sampled']}   "
            f"sigma={stats['sigma']}"
        )
        ctk.CTkLabel(plot_window, text=summary).pack(anchor="w", padx=12, pady=(0, 10))

    def _current_crop_for_analysis(self):
        crop = dict(self.crop_cfg)
        crop["enabled"] = bool(self.crop_enabled_var.get())
        return crop

    def _find_preview_video(self):
        video_dir = os.path.normpath(self.video_dir_entry.get().strip())
        file_type = self.file_type_entry.get().strip().lstrip(".") or "mp4"
        calibration_video = os.path.abspath(os.path.normpath(self.calibration_video_entry.get().strip() or ""))
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        suffix = "." + file_type.lower()
        for name in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, name)
            if not os.path.isfile(path) or not name.lower().endswith(suffix):
                continue
            if calibration_video and os.path.abspath(path) == calibration_video:
                continue
            return path
        raise FileNotFoundError(f"No behavior .{file_type} videos found in {video_dir}")

    @staticmethod
    def _video_fps_and_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (fps if fps and fps > 0 else 25.0), frame_count

    def _preview_freeze_sample(self):
        try:
            motion_cutoff = self._parse_optional_float(self.motion_cutoff_entry.get())
            if motion_cutoff is None:
                raise ValueError("Run auto calibration or enter motion_cutoff before previewing freeze parameters.")

            video_path = self._find_preview_video()
            fps, frame_count = self._video_fps_and_frames(video_path)
            configured_start = int(float(self.start_entry.get().strip() or "0"))
            configured_end = self._parse_optional_int(self.end_entry.get()) or frame_count
            sample_end = min(frame_count, configured_end)
            sample_start = max(configured_start, sample_end - int(round(fps * 60)))
            if sample_end <= sample_start:
                raise ValueError("Could not select a valid final 60-second preview segment.")

            params = {
                "video_path": video_path,
                "file_name": os.path.basename(video_path),
                "video_dir": os.path.dirname(video_path),
                "fps": fps,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "dsmpl": float(self.dsmpl_entry.get().strip() or "1.0"),
                "crop": self._current_crop_for_analysis(),
                "motion_cutoff": float(motion_cutoff),
                "sigma": float(self.sigma_entry.get().strip() or "1.0"),
                "freeze_threshold": float(self.freeze_threshold_entry.get().strip() or "50"),
                "min_duration_seconds": float(self.min_duration_entry.get().strip() or "0.5"),
            }
            self._log(
                "[INFO] Preview analysis started: "
                f"{params['file_name']} frames {sample_start}-{sample_end}."
            )
            threading.Thread(target=self._preview_worker, args=(params,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Preview Failed", str(e))
            self._log(f"[ERROR] Preview failed: {e}")

    def _preview_worker(self, params):
        try:
            from FreezeAnalysis_Functions import Measure_Freezing, Measure_Motion

            video_dict = {
                "dpath": params["video_dir"],
                "file": params["file_name"],
                "fpath": params["video_path"],
                "fps": params["fps"],
                "start": params["sample_start"],
                "end": params["sample_end"],
                "dsmpl": params["dsmpl"],
                "stretch": {"width": 1.0, "height": 1.0},
                "ftype": self.file_type_entry.get().strip().lstrip(".") or "mp4",
                "FileNames": [params["file_name"]],
                "cal_frms": int(float(self.cal_frames_entry.get().strip() or "250")),
                "crop": CropBox(params["crop"]) if params["crop"].get("enabled", False) else None,
            }
            motion = Measure_Motion(video_dict, params["motion_cutoff"], SIGMA=params["sigma"])
            min_duration_frames = int(round(params["min_duration_seconds"] * params["fps"]))
            freezing = Measure_Freezing(motion, params["freeze_threshold"], MinDuration=min_duration_frames)
            self.root.after(0, self._open_preview_player, params, motion, freezing, None)
        except Exception as e:
            self.root.after(0, self._open_preview_player, params, None, None, e)

    def _open_preview_player(self, params, motion, freezing, error):
        if error is not None:
            messagebox.showerror("Preview Failed", str(error))
            self._log(f"[ERROR] Preview failed: {error}")
            return
        self._log("[OK] Preview analysis ready. Opening player.")
        FreezePreviewPlayer(self, params, motion, freezing)

    def _yaml_from_ui(self):
        crop = dict(self.crop_cfg)
        crop["enabled"] = bool(self.crop_enabled_var.get())
        return {
            "project": {
                "video_dir": os.path.normpath(self.video_dir_entry.get().strip()),
                "file_type": self.file_type_entry.get().strip().lstrip(".") or "mp4",
                "start_frame": int(float(self.start_entry.get().strip() or "0")),
                "end_frame": self._parse_optional_int(self.end_entry.get()),
                "dsmpl": float(self.dsmpl_entry.get().strip() or "1.0"),
            },
            "crop": crop,
            "calibration": {
                "video_path": os.path.normpath(self.calibration_video_entry.get().strip()),
                "cal_frames": int(float(self.cal_frames_entry.get().strip() or "250")),
                "cal_pixels": int(float(self.cal_pixels_entry.get().strip() or "10000")),
                "sigma": float(self.sigma_entry.get().strip() or "1.0"),
                "percentile": float(self.percentile_entry.get().strip() or "99.99"),
                "cutoff_multiplier": float(self.cutoff_multiplier_entry.get().strip() or "2.8"),
                "motion_cutoff": self._parse_optional_float(self.motion_cutoff_entry.get()),
            },
            "freeze": {
                "freeze_threshold": float(self.freeze_threshold_entry.get().strip() or "50"),
                "min_duration_seconds": float(self.min_duration_entry.get().strip() or "0.5"),
            },
            "summary": {
                "bins": self._parse_bins(),
            },
            "run": {
                "accept_p_frames": bool(self.accept_p_frames_var.get()),
                "save_frame_data": bool(self.save_frame_data_var.get()),
            },
        }

    def _set_text(self, widget, value):
        widget.delete(0, "end")
        widget.insert(0, "" if value is None else str(value))

    def _load_yaml(self):
        try:
            path = filedialog.askopenfilename(title="Load Freeze YAML", filetypes=[("YAML", "*.yml *.yaml")])
            if not path:
                return
            self._load_yaml_path(path)
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))
            self._log(f"[ERROR] Load failed: {e}")

    def _load_yaml_path(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        project = data.get("project", {})
        crop = data.get("crop", {}) or {}
        calibration = data.get("calibration", {})
        freeze = data.get("freeze", {})
        summary = data.get("summary", {})
        run = data.get("run", {})

        self._set_text(self.video_dir_entry, project.get("video_dir", ""))
        self._set_text(self.file_type_entry, project.get("file_type", "mp4"))
        self._set_text(self.start_entry, project.get("start_frame", 0))
        self._set_text(self.end_entry, project.get("end_frame"))
        self._set_text(self.dsmpl_entry, project.get("dsmpl", 1.0))

        self._set_text(self.calibration_video_entry, calibration.get("video_path", ""))
        self._set_text(self.motion_cutoff_entry, calibration.get("motion_cutoff"))
        self._set_text(self.cal_frames_entry, calibration.get("cal_frames", 250))
        self._set_text(self.cal_pixels_entry, calibration.get("cal_pixels", 10000))
        self._set_text(self.sigma_entry, calibration.get("sigma", 1.0))
        self._set_text(self.percentile_entry, calibration.get("percentile", 99.99))
        self._set_text(self.cutoff_multiplier_entry, calibration.get("cutoff_multiplier", 2.8))

        self._set_text(self.freeze_threshold_entry, freeze.get("freeze_threshold", 50))
        self._set_text(self.min_duration_entry, freeze.get("min_duration_seconds", 0.5))
        self.accept_p_frames_var.set(bool(run.get("accept_p_frames", False)))
        self.save_frame_data_var.set(bool(run.get("save_frame_data", True)))

        self.crop_cfg = {
            "enabled": bool(crop.get("enabled", False)),
            "x0": int(crop.get("x0", 0)),
            "x1": int(crop.get("x1", 960)),
            "y0": int(crop.get("y0", 0)),
            "y1": int(crop.get("y1", 576)),
        }
        self.crop_enabled_var.set(self.crop_cfg["enabled"])
        self._update_crop_label()

        self._clear_bins()
        bins = summary.get("bins")
        if bins:
            for name, rng in bins.items():
                self._add_bin_row(name, rng[0], rng[1])

        self._log(f"[OK] YAML loaded: {os.path.normpath(path)}")
        self.current_config_path = os.path.normpath(path)

    def _save_yaml(self):
        try:
            yaml_obj = self._yaml_from_ui()
            out_path = self.current_config_path or filedialog.asksaveasfilename(
                    title="Save Freeze YAML",
                    defaultextension=".yml",
                    filetypes=[("YAML", "*.yml *.yaml")],
                    initialfile="project_freeze_config.yml",
                )
            if not out_path:
                return
            self._save_yaml_to_path(out_path, yaml_obj)
            messagebox.showinfo("Saved", f"Config saved:\n{os.path.normpath(out_path)}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))
            self._log(f"[ERROR] Save failed: {e}")

    def _default_config_path(self):
        video_dir = os.path.normpath(self.video_dir_entry.get().strip())
        if not video_dir:
            raise ValueError("Set Project Videos before saving or running analysis.")
        analysis_dir = os.path.join(video_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        return os.path.join(analysis_dir, "project_freeze_config.yml")

    def _save_yaml_to_path(self, out_path, yaml_obj=None):
        yaml_obj = yaml_obj or self._yaml_from_ui()
        out_path = os.path.normpath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        self.current_config_path = out_path
        self._log(f"[OK] YAML saved: {out_path}")
        return out_path

    def _save_and_run_batch_analysis(self):
        try:
            config_path = self.current_config_path or self._default_config_path()
            yaml_obj = self._yaml_from_ui()
            if yaml_obj.get("calibration", {}).get("motion_cutoff") is None:
                raise ValueError("Run auto calibration or enter motion_cutoff before batch analysis.")
            config_path = self._save_yaml_to_path(config_path, yaml_obj)
            self._log("[INFO] Batch analysis started. This may take a while.")
            threading.Thread(target=self._batch_analysis_worker, args=(config_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Batch Analysis Failed", str(e))
            self._log(f"[ERROR] Batch analysis failed: {e}")

    def _batch_analysis_worker(self, config_path):
        script_path = os.path.join(os.path.dirname(__file__), "RunFreezeAnalysisFromYAML.py")
        cmd = [sys.executable, script_path, "--config", config_path]
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            text=True,
            capture_output=True,
        )
        self.root.after(0, self._finish_batch_analysis, proc.returncode, proc.stdout, proc.stderr)

    def _finish_batch_analysis(self, returncode, stdout, stderr):
        if stdout:
            for line in stdout.strip().splitlines():
                self._log(line)
        if stderr:
            for line in stderr.strip().splitlines():
                self._log(line)
        if returncode == 0:
            self._log("[OK] Batch analysis finished.")
            messagebox.showinfo("Batch Analysis", "Batch analysis finished.")
        else:
            self._log(f"[ERROR] Batch analysis failed with exit code {returncode}.")
            messagebox.showerror("Batch Analysis Failed", f"Batch analysis failed with exit code {returncode}.")

    def run(self):
        if not self.embedded:
            self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or edit freeze-analysis YAML config.")
    parser.add_argument("--config", "-c", help="Optional YAML config to load at startup")
    args = parser.parse_args()
    FreezeConfigBuilderApp(initial_config_path=args.config).run()
