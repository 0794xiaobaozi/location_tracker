#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main GUI for crop and freeze-analysis workflows."""

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from freeze.BuildFreezeConfigGUI import FreezeConfigBuilderApp


def ensure_standard_streams():
    if getattr(sys, "stdout", None) is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if getattr(sys, "stderr", None) is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def build_internal_command(internal_arg, extra_args):
    if getattr(sys, "frozen", False):
        return [sys.executable, internal_arg, *extra_args]
    return [sys.executable, os.path.abspath(__file__), internal_arg, *extra_args]


def subprocess_startup_kwargs():
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


def run_internal_command():
    if len(sys.argv) < 2:
        return False
    command = sys.argv[1]
    if command == "--internal-select-intervals":
        from crop.SelectVideoIntervals import main

        sys.argv = ["SelectVideoIntervals.py", *sys.argv[2:]]
        main()
        return True
    if command == "--internal-crop-videos":
        from crop.CropVideosFromIntervals import main

        sys.argv = ["CropVideosFromIntervals.py", *sys.argv[2:]]
        main()
        return True
    if command == "--internal-run-freeze":
        from freeze.RunFreezeAnalysisFromYAML import main

        sys.argv = ["RunFreezeAnalysisFromYAML.py", *sys.argv[2:]]
        main()
        return True
    return False


class CropWorkflowPanel:
    def __init__(self, parent):
        self.parent = parent
        self.frame = ctk.CTkFrame(parent)
        self.frame.pack(fill="both", expand=True)
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self.frame, text="Crop Workflow", font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=16, pady=(16, 8))

        row = ctk.CTkFrame(self.frame, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=8)
        self.video_dir_entry = ctk.CTkEntry(row)
        self.video_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=110, command=self._browse_video_dir).pack(side="left")

        options = ctk.CTkFrame(self.frame)
        options.pack(fill="x", padx=16, pady=8)
        self.crop_mode = tk.StringVar(value="auto")
        ctk.CTkRadioButton(options, text="Auto duration from START", variable=self.crop_mode, value="auto").pack(side="left", padx=10, pady=10)
        ctk.CTkRadioButton(options, text="Manual START and END", variable=self.crop_mode, value="manual").pack(side="left", padx=10, pady=10)

        duration_row = ctk.CTkFrame(self.frame)
        duration_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(duration_row, text="Auto duration").pack(side="left", padx=(10, 6), pady=10)
        self.duration_entry = ctk.CTkEntry(duration_row, width=90)
        self.duration_entry.insert(0, "5")
        self.duration_entry.pack(side="left", padx=(0, 8), pady=10)
        self.duration_unit = ctk.CTkOptionMenu(duration_row, values=["minutes", "seconds"])
        self.duration_unit.set("minutes")
        self.duration_unit.pack(side="left", padx=(0, 12), pady=10)
        ctk.CTkLabel(duration_row, text="Manual mode ignores duration and lets you set both START and END in the selector.").pack(side="left", padx=8)

        actions = ctk.CTkFrame(self.frame)
        actions.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(actions, text="1) Select Video Intervals", command=self._select_intervals, fg_color="#2E8B57").pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkButton(actions, text="2) Crop Videos From Intervals", command=self._crop_videos, fg_color="#1F6AA5").pack(fill="x", padx=10, pady=(6, 10))

        self.status_box = ctk.CTkTextbox(self.frame, height=260)
        self.status_box.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        self._log("Ready. Choose a video folder, select intervals, then crop videos.")

    def _log(self, text):
        self.status_box.insert("end", text + "\n")
        self.status_box.see("end")

    def _browse_video_dir(self):
        path = filedialog.askdirectory(title="Select Video Directory")
        if path:
            self.video_dir_entry.delete(0, "end")
            self.video_dir_entry.insert(0, os.path.normpath(path))

    def _video_dir(self):
        video_dir = os.path.normpath(self.video_dir_entry.get().strip())
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        return video_dir

    def _select_intervals(self):
        try:
            video_dir = self._video_dir()
            args = ["-d", video_dir, "--gui", "modern"]
            if self.crop_mode.get() == "auto":
                duration = float(self.duration_entry.get().strip())
                if duration <= 0:
                    raise ValueError("Auto duration must be greater than 0.")
                if self.duration_unit.get() == "minutes":
                    duration *= 60
                args.extend(["--auto-duration-seconds", f"{duration:g}"])
            cmd = build_internal_command("--internal-select-intervals", args)
            self._log("[INFO] Opening interval selector...")
            threading.Thread(target=self._run_command, args=(cmd,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Select Intervals Failed", str(e))
            self._log(f"[ERROR] Select intervals failed: {e}")

    def _crop_videos(self):
        try:
            video_dir = self._video_dir()
            cmd = build_internal_command("--internal-crop-videos", ["--directory", video_dir])
            self._log("[INFO] Cropping videos...")
            threading.Thread(target=self._run_command, args=(cmd,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Crop Failed", str(e))
            self._log(f"[ERROR] Crop failed: {e}")

    def _run_command(self, cmd):
        env = os.environ.copy()
        env["TRACKER_HIDE_SUBPROCESS_WINDOWS"] = "1"
        env["TRACKER_VIDEO_INFO_BACKEND"] = "opencv"
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            text=True,
            capture_output=True,
            env=env,
            **subprocess_startup_kwargs(),
        )
        self.parent.after(0, self._finish_command, proc.returncode, proc.stdout, proc.stderr)

    def _finish_command(self, returncode, stdout, stderr):
        for text in (stdout, stderr):
            if text:
                for line in text.strip().splitlines():
                    self._log(line)
        if returncode == 0:
            self._log("[OK] Command finished.")
        else:
            self._log(f"[ERROR] Command failed with exit code {returncode}.")


class TrackerGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Tracker Workflow GUI")
        self.root.geometry("1240x900")
        self.root.minsize(1040, 720)

        self.sidebar = ctk.CTkFrame(self.root, width=220)
        self.sidebar.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self.sidebar.pack_propagate(False)

        self.content = ctk.CTkFrame(self.root)
        self.content.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=12)

        ctk.CTkLabel(self.sidebar, text="Tracker", font=("Segoe UI", 22, "bold")).pack(anchor="w", padx=18, pady=(20, 18))
        self.crop_button = ctk.CTkButton(self.sidebar, text="Crop", command=lambda: self._show_panel("crop"))
        self.crop_button.pack(fill="x", padx=14, pady=6)
        self.freeze_button = ctk.CTkButton(self.sidebar, text="Freeze Analysis", command=lambda: self._show_panel("freeze"))
        self.freeze_button.pack(fill="x", padx=14, pady=6)

        self.panels = {}
        self._show_panel("crop")

    def _show_panel(self, name):
        self.crop_button.configure(fg_color="#1F6AA5" if name == "crop" else "transparent")
        self.freeze_button.configure(fg_color="#1F6AA5" if name == "freeze" else "transparent")

        for panel in self.panels.values():
            panel.pack_forget()

        if name not in self.panels:
            panel_frame = ctk.CTkFrame(self.content)
            if name == "crop":
                CropWorkflowPanel(panel_frame)
            elif name == "freeze":
                FreezeConfigBuilderApp(root=panel_frame, embedded=True)
            else:
                return
            self.panels[name] = panel_frame

        self.panels[name].pack(fill="both", expand=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ensure_standard_streams()
    if not run_internal_command():
        TrackerGUI().run()
