# Location Tracker

This repo has two parts:

1. **Crop workflow** (select interval -> export cropped videos)
2. **Analysis workflows** (run location tracking or freeze analysis on cropped videos)

---

## 1) Setup

```bash
pixi install -e location-tracker
```

Run commands with:

```bash
pixi run -e location-tracker python <script>.py ...
```

Avoid `py ...` to prevent interpreter mismatch.

Open the combined workflow GUI:

```bash
pixi run -e location-tracker python TrackerGUI.py
```

The left navigation switches between **Crop** and **Freeze Analysis**. The right panel shows the selected workflow.

Command examples below use paths **relative to the repository root**. The folder name `my_project` is a placeholder—use your own dataset directory name in both `video/...` and `video/cropped_video/...`.

---

## 2) Crop Workflow

Use these two scripts together.

### Step 2.1 Select video intervals

Script: `crop/SelectVideoIntervals.py`

```bash
pixi run -e location-tracker python crop/SelectVideoIntervals.py -d "video/my_project" --auto-5min --gui modern
```

Output:

- `video_intervals.json` in the video directory

Common options:

- `--gui modern|classic`
- `--auto-5min` / `--auto-10min`
- `--auto-duration-seconds <seconds>` (custom auto duration; select START only, END is START + duration)
- `--exclude <file>` (repeatable)
- `--output <path>`

In the combined GUI (`TrackerGUI.py`), Crop has two modes:

- **Auto duration from START**: choose START in the selector; END is calculated from the duration field and unit (`minutes` or `seconds`).
- **Manual START and END**: choose both START and END in the selector; the duration field is ignored.

Modern GUI shortcuts:

- `S`: set start
- `E`: set end (manual mode only)
- `Enter`: confirm current video
- `Q`: skip current video
- `Space`: play/pause
- `Left/Right`: -/+10 frames
- `Up/Down`: -/+100 frames
- `R`: reset interval

### Step 2.2 Crop videos from intervals

Script: `crop/CropVideosFromIntervals.py`

```bash
pixi run -e location-tracker python crop/CropVideosFromIntervals.py --directory "video/my_project"
```

Output:

- Cropped videos under `cropped_video` (same directory structure preserved)

Common options:

- `--directory` (repeatable)
- `--recursive`
- `--intervals-file`
- `--output-dir` (default: `cropped_video`)

---

## 3) A Location Tracking Workflow

Tracking supports two entry modes:

- **Simple CLI mode**: `tracking/RunLocationTrackingBatch.py`
- **YAML mode** (recommended for reproducibility): template/GUI + `tracking/RunLocationTrackingFromYAML.py`

### Option A: YAML tracking (recommended)

#### Step A1 Generate YAML template

Script: `tracking/CreateLocationTrackingYAMLTemplate.py`

```bash
pixi run -e location-tracker python tracking/CreateLocationTrackingYAMLTemplate.py --output "./project_tracking_config.yml" --video-dir "video/cropped_video/my_project"
```

#### Step A2 Edit YAML in GUI (optional)

Script: `tracking/BuildTrackingConfigGUI.py`

```bash
pixi run -e location-tracker python tracking/BuildTrackingConfigGUI.py
```

#### Step A3 Run tracking from YAML

Script: `tracking/RunLocationTrackingFromYAML.py`

```bash
pixi run -e location-tracker python tracking/RunLocationTrackingFromYAML.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

YAML run options:

- `run.parallel`
- `run.n_processes` (`null` = auto)
- `run.accept_p_frames`

### Option B: Simple CLI tracking

Script: `tracking/RunLocationTrackingBatch.py`

```bash
pixi run -e location-tracker python tracking/RunLocationTrackingBatch.py --directory "video/cropped_video/my_project" --ftype mp4 --loc-thresh 99.0 --parallel
```

Useful options:

- `--parallel`
- `--n-processes`
- `--loc-thresh`
- `--method abs|light|dark`
- `--use-window --window-size --window-weight`
- `--accept-p-frames`

### Tracking outputs

- Per video: `*_LocationOutput.csv`
- Batch summary: `BatchSummary.csv`

---

## 3) B Freeze Analysis Workflow

Freeze analysis is being built as a separate workflow from location tracking. The current framework follows ezTrack's freeze-analysis logic: calibrate a grayscale motion cutoff from an empty-chamber video, measure frame-to-frame motion in behavior videos, then score freezing when motion stays below a threshold for a minimum duration.

### Step B1 Generate freeze YAML template

Script: `freeze/CreateFreezeAnalysisYAMLTemplate.py`

```bash
pixi run -e location-tracker python freeze/CreateFreezeAnalysisYAMLTemplate.py --output "video/cropped_video/my_project/project_freeze_config.yml" --video-dir "video/cropped_video/my_project" --calibration-video "video/cropped_video/my_project/empty_chamber.mp4"
```

### Step B2 Edit YAML in GUI (optional)

Script: `freeze/BuildFreezeConfigGUI.py`

```bash
pixi run -e location-tracker python freeze/BuildFreezeConfigGUI.py
```

You can also open the same empty GUI from the freeze runner:

```bash
pixi run -e location-tracker python freeze/RunFreezeAnalysisFromYAML.py
```

Open an existing freeze YAML directly:

```bash
pixi run -e location-tracker python freeze/BuildFreezeConfigGUI.py --config "video/cropped_video/my_project/analysis/project_freeze_config.yml"
```

GUI workflow:

1. Set **Project Videos** to the folder containing the behavior videos.
2. Set **Calibration Video** to the empty-chamber video.
3. Click **Initialize Project** to create `analysis/project_freeze_config.yml` in the selected project folder.
4. Optional: click **Select Crop From First Video** and enable crop if cables, walls, or reflections should be excluded.
5. Click **Auto Calibrate From Empty Video**. The GUI fills `motion_cutoff` and opens a plot showing the empty-box pixel-difference distribution, the selected percentile, and the final cutoff.
6. Set initial `freeze_threshold` and `min_duration_seconds`.
7. Click **Preview / Tune Freeze Parameters**. The GUI picks a behavior video, extracts the final 60 seconds of the configured analysis range, computes motion/freezing, and opens a player with the current freeze/moving decision overlaid on the video. Adjust `freeze_threshold` and `min_duration_seconds` inside the preview window, click **Recompute**, then **Apply To Main GUI** when the overlay looks reasonable.
8. Add summary bins in the table if needed. **Insert Bin** adds one row next to the previous bin with a default 60-second duration; edit name/start/end by hand.
9. Click **Save YAML**, or click **Save YAML & Run Batch Analysis** to save the current GUI settings and immediately start batch analysis.

Key fields:

- `calibration.motion_cutoff`: estimated from the empty-chamber video by the GUI, or set manually.
- `calibration.percentile` and `calibration.cutoff_multiplier`: GUI auto-calibration uses `motion_cutoff = cutoff_multiplier * percentile(diff)`. Defaults are `99.99` and `2.8`.
- `freeze.freeze_threshold`: maximum motion pixels allowed for a frame to be considered freezing.
- `freeze.min_duration_seconds`: minimum continuous low-motion duration required.
- `Preview / Tune Freeze Parameters`: uses the final 60 seconds of a non-calibration behavior video so these two freeze parameters can be adjusted visually before batch analysis.
- `summary.bins`: named time windows in seconds for summary output. Empty bins means one `all` summary.
- `Save YAML & Run Batch Analysis`: saves the current GUI settings to the active YAML file, then runs `freeze/RunFreezeAnalysisFromYAML.py --config <that-yaml>`.

### Step B3 Run freeze analysis

Script: `freeze/RunFreezeAnalysisFromYAML.py`

Run analysis by passing a config. If no config is passed, this script opens the GUI instead.

```bash
pixi run -e location-tracker python freeze/RunFreezeAnalysisFromYAML.py --config "video/cropped_video/my_project/analysis/project_freeze_config.yml"
```

Calibration only:

```bash
pixi run -e location-tracker python freeze/RunFreezeAnalysisFromYAML.py --config "video/cropped_video/my_project/analysis/project_freeze_config.yml" --calibrate-only
```

In the GUI, `Auto Calibrate From Empty Video` estimates and fills `calibration.motion_cutoff` directly using percentile multiplier calibration. The `--calibrate-only` command prints the same calibration values without running batch analysis.

### Freeze outputs

- Per video: `*_FreezingOutput.csv`
- Batch summary: `FreezeBatchSummary.csv`

---

## 4) Result Visualization

### Trajectory figure images (one image per video)

Script: `visualization/GenerateTrajectoryImages.py`

Recommended command (load settings from project YAML):

```bash
pixi run -e location-tracker python visualization/GenerateTrajectoryImages.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

Output:

- Per video: `*_Trajectory.png`

Notes:

- Visualization scripts are organized under `visualization/`.
- `--config` auto-loads `video_dir`, `analysis_roi`, and `functional_roi`.
- Config `crop` is applied by default (YAML is treated as source of truth).
- Use `--skip-config-crop` if you want full-frame background without applying YAML crop.

### Tracking overlay videos (single or batch, both config-driven)

Script: `visualization/GenerateTrackingOverlayVideos.py`

Single video (still reads YAML for crop/video_dir):

```bash
pixi run -e location-tracker python visualization/GenerateTrackingOverlayVideos.py --config "video/cropped_video/my_project/project_tracking_config.yml" --video "1-5.mp4"
```

Batch (all videos from YAML `project.video_dir`):

```bash
pixi run -e location-tracker python visualization/GenerateTrackingOverlayVideos.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

Output:

- Single mode: `*_TrackingOverlay.mp4`
- Batch mode: one `*_TrackingOverlay.mp4` per video
- Both modes read the same YAML config by default

### EPM-specific statistics and figures

#### ROI statistics (config-driven)

Scripts:

- `visualization/GenerateROIEntryStatistics.py`
- `visualization/GenerateROIStatistics.py`

Generate entry statistics:

```bash
pixi run -e location-tracker python visualization/GenerateROIEntryStatistics.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

Generate time statistics:

```bash
pixi run -e location-tracker python visualization/GenerateROIStatistics.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

Outputs:

- `ROI_Entry_Statistics.csv`
- `ROI_Statistics_Detailed.csv`
- `ROI_Statistics_Summary.csv`

#### EPM bar charts (supports --config)

Script: `visualization/GenerateEPMBarCharts.py`

Use external group YAML:

```bash
pixi run -e location-tracker python visualization/GenerateEPMBarCharts.py --config "video/cropped_video/my_project/project_tracking_config.yml" --group-config "video/cropped_video/my_project/grouping.yml" --open-arms "Top,Bottom" --closed-arms "Left,Right"
```

Group YAML format (`grouping.yml`):

```yaml
groups:
  WT:
    - 1-12
    - 1-14
  pp3r1:
    - 1-1
    - 1-2
```

Rules:

- Top-level key must be `groups`.
- Exactly two groups are required for this bar chart.
- Video IDs can be either stem (`1-1`) or filename (`1-1.mp4`).
- Every video in ROI statistics must appear in this grouping file.
- You must provide both `--open-arms` and `--closed-arms`.
- `--open-arms` and `--closed-arms` must use ROI names that exist in your ROI stats columns.
- Open and closed arm ROI lists must not overlap.

Notes:

- Reads `ROI_Entry_Statistics.csv` and `ROI_Statistics_Detailed.csv` in `project.video_dir`.
- Output files: `EPM_BarCharts.png` and `EPM_BarCharts.pdf`.

#### EPM transformed heatmaps (YAML-driven)

Script: `visualization/GenerateTransformedTrajectoryHeatmap.py`

Batch mode:

```bash
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "video/cropped_video/my_project/project_tracking_config.yml"
```

Single video mode:

```bash
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "video/cropped_video/my_project/project_tracking_config.yml" --video "1-5.mp4"
```

Vertex modes:

- `--vertex-mode config`: use `epm_transform.original_vertices` in YAML.
- `--vertex-mode functional`: derive 12 vertices from `functional_roi` (`Left/Right/Top/Bottom` required).
- `--vertex-mode gui`: open modern GUI on a random project frame (after applying YAML crop) with all functional ROI overlays; a standard EPM reference panel is shown, and selected reference points are highlighted synchronously while picking.
- GUI point picking is restricted to functional ROI vertices (clicks snap to nearest functional vertex).
- GUI supports `Undo` to revert the last point selection.
- Hard requirement: this script requires every `functional_roi.regions[*].vertices` polygon to have exactly 4 points.

GUI example (save picked vertices back to YAML):

```bash
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "video/cropped_video/my_project/project_tracking_config.yml" --vertex-mode gui --save-picked-vertices
```

Required YAML section for `--vertex-mode config`:

```yaml
epm_transform:
  original_vertices:
    - [x0, y0]
    - [x1, y1]
    - [x2, y2]
    - [x3, y3]
    - [x4, y4]
    - [x5, y5]
    - [x6, y6]
    - [x7, y7]
    - [x8, y8]
    - [x9, y9]
    - [x10, y10]
    - [x11, y11]
  center_size_px: 119
  arm_length_ratio: 5.0
  canvas_size: 1409
  num_bins: 80
  sigma: 1.2
  colormap: viridis
  skip_seconds: 15
  fps_default: 25.0
```

Outputs per video:

- `*_EPM_Heatmap_Standard.png/.pdf`
- `*_TransformedTrajectory.csv`

---

## 5) Required files for YAML workflow

- `tracking/LocationTracking_Functions.py`
- `tracking/RunLocationTrackingFromYAML.py`
- `tracking/CreateLocationTrackingYAMLTemplate.py`
- `tracking/BuildTrackingConfigGUI.py`
- `freeze/FreezeAnalysis_Functions.py`
- `freeze/AutoFreezeCalibration.py`
- `freeze/RunFreezeAnalysisFromYAML.py`
- `freeze/CreateFreezeAnalysisYAMLTemplate.py`
- `freeze/BuildFreezeConfigGUI.py`
- `pixi.toml` / `pixi.lock`

---

## 6) Notes

- Large local data files are ignored via `.gitignore`.
- If GUI does not appear, test environment quickly:

```bash
pixi run -e location-tracker python -c "import customtkinter, cv2; print('ok')"
```

---

## 7) Citation and License

This project includes tracking logic adapted from `ezTrack`.

If you use this project in research, please also cite ezTrack:

Pennington ZT, Dong Z, Feng Y, Vetere LM, Page-Harley L, Shuman T, Cai DJ (2019).  
ezTrack: An open-source video analysis pipeline for the investigation of animal behavior.  
Scientific Reports, 9(1): 19979.

License:

- This project is distributed under the GNU GPLv3.
- See `LICENSE` for details.
