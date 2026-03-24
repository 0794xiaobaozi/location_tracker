#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modern GUI to create a project YAML config for location tracking.

This tool keeps `LocationTracking_Functions as lt` as the backend and uses its
OpenCV interactive selectors for:
- Crop Settings
- Analysis ROI (rectangle or polygon)
- Functional ROIs
- Scale definition
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import yaml

import LocationTracking_Functions as lt


class TrackingConfigBuilderApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Location Tracking Config Builder")
        self.root.geometry("980x760")

        self.video_dict = None
        self.scale_px_distance = None
        self.analysis_roi_type = tk.StringVar(value="rectangle")

        self._build_ui()

    def _build_ui(self):
        frame = ctk.CTkFrame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(frame, text="Project Video Folder", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        folder_row = ctk.CTkFrame(frame, fg_color="transparent")
        folder_row.pack(fill="x", padx=12)
        self.video_dir_entry = ctk.CTkEntry(folder_row)
        self.video_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(folder_row, text="Browse", width=110, command=self._browse_video_dir).pack(side="left")

        opts = ctk.CTkFrame(frame)
        opts.pack(fill="x", padx=12, pady=(10, 8))
        ctk.CTkLabel(opts, text="File Type").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.file_type_entry = ctk.CTkEntry(opts, width=80)
        self.file_type_entry.insert(0, "mp4")
        self.file_type_entry.grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ctk.CTkLabel(opts, text="Start Frame").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        self.start_entry = ctk.CTkEntry(opts, width=100)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=0, column=3, padx=8, pady=8, sticky="w")

        ctk.CTkLabel(opts, text="Downsample").grid(row=0, column=4, padx=8, pady=8, sticky="w")
        self.dsmpl_entry = ctk.CTkEntry(opts, width=100)
        self.dsmpl_entry.insert(0, "1.0")
        self.dsmpl_entry.grid(row=0, column=5, padx=8, pady=8, sticky="w")

        ctk.CTkButton(frame, text="1) Initialize Project", command=self._initialize_project, fg_color="#2E8B57").pack(fill="x", padx=12, pady=(8, 6))

        section = ctk.CTkFrame(frame)
        section.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(section, text="2) Interactive Selections", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        ctk.CTkButton(section, text="Select Crop Settings", command=self._select_crop).pack(fill="x", padx=10, pady=4)

        roi_type_row = ctk.CTkFrame(section, fg_color="transparent")
        roi_type_row.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(roi_type_row, text="Analysis ROI Type:").pack(side="left")
        ctk.CTkRadioButton(roi_type_row, text="Rectangle", variable=self.analysis_roi_type, value="rectangle").pack(side="left", padx=8)
        ctk.CTkRadioButton(roi_type_row, text="Polygon", variable=self.analysis_roi_type, value="polygon").pack(side="left", padx=8)
        ctk.CTkButton(section, text="Select Analysis ROI", command=self._select_analysis_roi).pack(fill="x", padx=10, pady=4)

        roi_names_row = ctk.CTkFrame(section, fg_color="transparent")
        roi_names_row.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(roi_names_row, text="Functional ROI Names (comma separated):").pack(side="left")
        self.region_names_entry = ctk.CTkEntry(roi_names_row)
        self.region_names_entry.insert(0, "Left,Right,Top,Bottom")
        self.region_names_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        ctk.CTkButton(section, text="Select Functional ROIs", command=self._select_functional_rois).pack(fill="x", padx=10, pady=4)

        scale_row = ctk.CTkFrame(section, fg_color="transparent")
        scale_row.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(scale_row, text="True Distance").pack(side="left")
        self.true_distance_entry = ctk.CTkEntry(scale_row, width=110)
        self.true_distance_entry.insert(0, "10")
        self.true_distance_entry.pack(side="left", padx=6)
        ctk.CTkLabel(scale_row, text="Unit").pack(side="left", padx=(8, 0))
        self.true_scale_entry = ctk.CTkEntry(scale_row, width=90)
        self.true_scale_entry.insert(0, "cm")
        self.true_scale_entry.pack(side="left", padx=6)
        ctk.CTkButton(section, text="Define Scale (2 points)", command=self._define_scale).pack(fill="x", padx=10, pady=4)

        ctk.CTkButton(frame, text="3) Save YAML Config", command=self._save_yaml, fg_color="#1F6AA5").pack(fill="x", padx=12, pady=(10, 6))

        tracking_section = ctk.CTkFrame(frame)
        tracking_section.pack(fill="x", padx=12, pady=(4, 6))
        ctk.CTkLabel(tracking_section, text="Tracking Parameters (same as notebook tracking_params)", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        row1 = ctk.CTkFrame(tracking_section, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(row1, text="loc_thresh").pack(side="left")
        self.loc_thresh_entry = ctk.CTkEntry(row1, width=90)
        self.loc_thresh_entry.insert(0, "99.5")
        self.loc_thresh_entry.pack(side="left", padx=(6, 16))

        self.use_window_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row1, text="use_window", variable=self.use_window_var).pack(side="left", padx=(0, 16))

        ctk.CTkLabel(row1, text="window_size").pack(side="left")
        self.window_size_entry = ctk.CTkEntry(row1, width=90)
        self.window_size_entry.insert(0, "100")
        self.window_size_entry.pack(side="left", padx=(6, 16))

        ctk.CTkLabel(row1, text="window_weight").pack(side="left")
        self.window_weight_entry = ctk.CTkEntry(row1, width=90)
        self.window_weight_entry.insert(0, "0.9")
        self.window_weight_entry.pack(side="left", padx=(6, 0))

        row2 = ctk.CTkFrame(tracking_section, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(row2, text="method").pack(side="left")
        self.method_option = ctk.CTkOptionMenu(row2, values=["abs", "light", "dark"])
        self.method_option.set("dark")
        self.method_option.pack(side="left", padx=(6, 16))

        self.rmv_wire_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row2, text="rmv_wire", variable=self.rmv_wire_var).pack(side="left", padx=(0, 16))

        ctk.CTkLabel(row2, text="wire_krn").pack(side="left")
        self.wire_krn_entry = ctk.CTkEntry(row2, width=90)
        self.wire_krn_entry.insert(0, "5")
        self.wire_krn_entry.pack(side="left", padx=(6, 0))

        self.status_box = ctk.CTkTextbox(frame, height=190)
        self.status_box.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self._log("Ready. 1) Set folder -> Initialize Project.")

    def _log(self, text: str):
        self.status_box.insert("end", text + "\n")
        self.status_box.see("end")

    def _browse_video_dir(self):
        path = filedialog.askdirectory(title="Select Video Directory")
        if path:
            self.video_dir_entry.delete(0, "end")
            self.video_dir_entry.insert(0, path)

    def _build_base_video_dict(self):
        dpath = os.path.normpath(self.video_dir_entry.get().strip())
        if not os.path.isdir(dpath):
            raise FileNotFoundError(f"Directory not found: {dpath}")

        ftype = self.file_type_entry.get().strip() or "mp4"
        start = int(float(self.start_entry.get().strip() or "0"))
        dsmpl = float(self.dsmpl_entry.get().strip() or "1.0")

        video_dict = {
            "dpath": dpath,
            "file": "",
            "ftype": ftype,
            "start": start,
            "end": None,
            "dsmpl": dsmpl,
            "stretch": {"width": 1.0, "height": 1.0},
            "region_names": None,
            "roi_stream": None,
            "crop": None,
            "mask": {"mask": None, "stream": None},
            "scale": None,
            "analysis_roi": None,
        }
        video_dict = lt.Batch_LoadFiles(video_dict)
        if len(video_dict.get("FileNames", [])) == 0:
            raise RuntimeError(f"No .{ftype} videos found in {dpath}")
        return video_dict

    def _ensure_initialized(self):
        if self.video_dict is None:
            raise RuntimeError("Please run 'Initialize Project' first.")

    def _ensure_reference(self, num_frames=50):
        self._ensure_initialized()
        if self.video_dict.get("reference") is None:
            self.video_dict["reference"], _ = lt.Reference(self.video_dict, fstfile=True, num_frames=num_frames)
            self._log("[INFO] Reference frame prepared.")

    def _initialize_project(self):
        try:
            self.video_dict = self._build_base_video_dict()
            first_file = self.video_dict["FileNames"][0]
            self.video_dict["file"] = first_file
            self.video_dict["fpath"] = os.path.join(self.video_dict["dpath"], first_file)
            self._log(f"[OK] Initialized. Videos: {len(self.video_dict['FileNames'])}, first: {first_file}")
        except Exception as e:
            messagebox.showerror("Initialize Failed", str(e))
            self._log(f"[ERROR] Initialize failed: {e}")

    def _select_crop(self):
        try:
            self._ensure_initialized()
            self.video_dict = lt.LoadAndCrop_cv2(self.video_dict, fstfile=True)
            self._log("[OK] Crop selected.")
        except Exception as e:
            messagebox.showerror("Crop Failed", str(e))
            self._log(f"[ERROR] Crop failed: {e}")

    def _select_analysis_roi(self):
        try:
            self._ensure_initialized()
            roi_type = self.analysis_roi_type.get()
            if roi_type == "polygon":
                self.video_dict["analysis_roi"] = lt.AnalysisROI_polygon_select_cv2(self.video_dict)
                self._log("[OK] Polygon analysis ROI selected.")
            else:
                self.video_dict["analysis_roi"] = lt.AnalysisROI_select_cv2(self.video_dict)
                self._log("[OK] Rectangle analysis ROI selected.")
        except Exception as e:
            messagebox.showerror("Analysis ROI Failed", str(e))
            self._log(f"[ERROR] Analysis ROI failed: {e}")

    def _select_functional_rois(self):
        try:
            self._ensure_initialized()
            names_raw = self.region_names_entry.get().strip()
            names = [x.strip() for x in names_raw.split(",") if x.strip()]
            if not names:
                raise ValueError("Please input at least one ROI name.")
            self.video_dict["region_names"] = names
            self._ensure_reference(num_frames=50)
            self.video_dict["roi_stream"] = lt.ROI_plot_cv2(self.video_dict)
            self._log(f"[OK] Functional ROIs selected: {', '.join(names)}")
        except Exception as e:
            messagebox.showerror("Functional ROI Failed", str(e))
            self._log(f"[ERROR] Functional ROI failed: {e}")

    def _define_scale(self):
        try:
            self._ensure_initialized()
            self._ensure_reference(num_frames=100)
            scale_dict = lt.DistanceTool_cv2(self.video_dict)
            px_distance = scale_dict.get("px_distance")
            if px_distance is None:
                self._log("[INFO] Scale skipped.")
                return
            true_distance = float(self.true_distance_entry.get().strip())
            true_scale = self.true_scale_entry.get().strip() or "cm"
            self.video_dict["scale"] = lt.setScale(true_distance, true_scale, scale_dict)
            self.video_dict["scale"]["factor"] = true_distance / px_distance if px_distance else 0
            self.scale_px_distance = px_distance
            self._log(
                f"[OK] Scale set: {px_distance:.3f}px = {true_distance} {true_scale} "
                f"(factor={self.video_dict['scale']['factor']:.6f} {true_scale}/px)"
            )
        except Exception as e:
            messagebox.showerror("Scale Failed", str(e))
            self._log(f"[ERROR] Scale failed: {e}")

    @staticmethod
    def _extract_crop_cfg(video_dict):
        crop = video_dict.get("crop")
        if crop is None or not hasattr(crop, "data"):
            return None
        data = crop.data
        return {
            "x0": int(data["x0"][0]),
            "x1": int(data["x1"][0]),
            "y0": int(data["y0"][0]),
            "y1": int(data["y1"][0]),
        }

    @staticmethod
    def _extract_analysis_cfg(video_dict):
        analysis_roi = video_dict.get("analysis_roi")
        if analysis_roi is None:
            return {"type": "none"}
        if isinstance(analysis_roi, tuple):
            x1, y1, x2, y2 = analysis_roi
            return {"type": "rectangle", "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        if isinstance(analysis_roi, dict):
            vertices = analysis_roi.get("vertices", [])
            return {"type": "polygon", "vertices": vertices}
        return {"type": "none"}

    @staticmethod
    def _extract_functional_cfg(video_dict):
        roi_stream = video_dict.get("roi_stream")
        region_names = video_dict.get("region_names")
        if roi_stream is None or not region_names:
            return {"regions": []}
        data = roi_stream.data if hasattr(roi_stream, "data") else {}
        xs = data.get("x", []) or data.get("xs", [])
        ys = data.get("y", []) or data.get("ys", [])
        regions = []
        for idx, name in enumerate(region_names):
            if idx >= len(xs) or idx >= len(ys):
                continue
            vertices = [[float(x), float(y)] for x, y in zip(xs[idx], ys[idx])]
            regions.append({"name": name, "vertices": vertices})
        return {"regions": regions}

    def _save_yaml(self):
        try:
            self._ensure_initialized()
            out_path = filedialog.asksaveasfilename(
                title="Save Tracking YAML",
                defaultextension=".yml",
                filetypes=[("YAML", "*.yml *.yaml")],
                initialfile="project_tracking_config.yml",
            )
            if not out_path:
                return

            project_cfg = {
                "video_dir": self.video_dict["dpath"],
                "file_type": self.video_dict["ftype"],
                "start_frame": int(self.video_dict.get("start", 0)),
                "end_frame": self.video_dict.get("end", None),
                "dsmpl": float(self.video_dict.get("dsmpl", 1.0)),
                "stretch_width": float(self.video_dict.get("stretch", {}).get("width", 1.0)),
                "stretch_height": float(self.video_dict.get("stretch", {}).get("height", 1.0)),
            }

            tracking_cfg = {
                "loc_thresh": float(self.loc_thresh_entry.get().strip() or "99.5"),
                "use_window": bool(self.use_window_var.get()),
                "window_size": int(float(self.window_size_entry.get().strip() or "100")),
                "window_weight": float(self.window_weight_entry.get().strip() or "0.9"),
                "method": self.method_option.get().strip() or "dark",
                "rmv_wire": bool(self.rmv_wire_var.get()),
                "wire_krn": int(float(self.wire_krn_entry.get().strip() or "5")),
            }

            scale_cfg = None
            if self.video_dict.get("scale"):
                s = self.video_dict["scale"]
                px = float(s.get("px_distance", 0) or 0)
                td = float(s.get("true_distance", 0) or 0)
                factor = float(s.get("factor", (td / px if px else 0)))
                scale_cfg = {
                    "px_distance": px,
                    "true_distance": td,
                    "true_scale": str(s.get("true_scale", "cm")),
                    "factor": factor,
                }

            yaml_obj = {
                "project": project_cfg,
                "crop": self._extract_crop_cfg(self.video_dict),
                "analysis_roi": self._extract_analysis_cfg(self.video_dict),
                "functional_roi": self._extract_functional_cfg(self.video_dict),
                "scale": scale_cfg,
                "tracking": tracking_cfg,
                "run": {
                    "parallel": True,
                    "n_processes": None,
                    "accept_p_frames": False,
                    "bin_dict": None,
                },
            }

            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

            self._log(f"[OK] YAML saved: {os.path.normpath(out_path)}")
            messagebox.showinfo("Saved", f"Config saved:\n{os.path.normpath(out_path)}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))
            self._log(f"[ERROR] Save failed: {e}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    TrackingConfigBuilderApp().run()

