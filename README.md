# Location Tracker

This repo has two parts:

1. **Crop workflow** (select interval -> export cropped videos)
2. **Tracking workflow** (run location tracking on cropped videos)

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

---

## 2) Crop Workflow

Use these two scripts together.

### Step 2.1 Select video intervals

Script: `SelectVideoIntervals.py`

```bash
pixi run -e location-tracker python SelectVideoIntervals.py -d "F:\Neuro\ezTrack\LocationTracking\video\EPM_later" --auto-5min --gui modern
```

Output:

- `video_intervals.json` in the video directory

Common options:

- `--gui modern|classic`
- `--auto-5min` / `--auto-10min`
- `--exclude <file>` (repeatable)
- `--output <path>`

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

Script: `CropVideosFromIntervals.py`

```bash
pixi run -e location-tracker python CropVideosFromIntervals.py --directory "F:\Neuro\ezTrack\LocationTracking\video\EPM_later"
```

Output:

- Cropped videos under `cropped_video` (same directory structure preserved)

Common options:

- `--directory` (repeatable)
- `--recursive`
- `--intervals-file`
- `--output-dir` (default: `cropped_video`)

---

## 3) Tracking Workflow

Tracking supports two entry modes:

- **Simple CLI mode**: `RunLocationTrackingBatch.py`
- **YAML mode** (recommended for reproducibility): template/GUI + `RunLocationTrackingFromYAML.py`

### Option A: YAML tracking (recommended)

#### Step A1 Generate YAML template

Script: `CreateLocationTrackingYAMLTemplate.py`

```bash
pixi run -e location-tracker python CreateLocationTrackingYAMLTemplate.py --output ".\project_tracking_config.yml" --video-dir "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later"
```

#### Step A2 Edit YAML in GUI (optional)

Script: `BuildTrackingConfigGUI.py`

```bash
pixi run -e location-tracker python BuildTrackingConfigGUI.py
```

#### Step A3 Run tracking from YAML

Script: `RunLocationTrackingFromYAML.py`

```bash
pixi run -e location-tracker python RunLocationTrackingFromYAML.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

YAML run options:

- `run.parallel`
- `run.n_processes` (`null` = auto)
- `run.accept_p_frames`

### Option B: Simple CLI tracking

Script: `RunLocationTrackingBatch.py`

```bash
pixi run -e location-tracker python RunLocationTrackingBatch.py --directory "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later" --ftype mp4 --loc-thresh 99.0 --parallel
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

## 4) Result Visualization

### Trajectory figure images (one image per video)

Script: `visualization/GenerateTrajectoryImages.py`

Recommended command (load settings from project YAML):

```bash
pixi run -e location-tracker python visualization/GenerateTrajectoryImages.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
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
pixi run -e location-tracker python visualization/GenerateTrackingOverlayVideos.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml" --video "1-5.mp4"
```

Batch (all videos from YAML `project.video_dir`):

```bash
pixi run -e location-tracker python visualization/GenerateTrackingOverlayVideos.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
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
pixi run -e location-tracker python visualization/GenerateROIEntryStatistics.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

Generate time statistics:

```bash
pixi run -e location-tracker python visualization/GenerateROIStatistics.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

Outputs:

- `ROI_Entry_Statistics.csv`
- `ROI_Statistics_Detailed.csv`
- `ROI_Statistics_Summary.csv`

#### EPM bar charts (supports --config)

Script: `visualization/GenerateEPMBarCharts.py`

Use external group YAML:

```bash
pixi run -e location-tracker python visualization/GenerateEPMBarCharts.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml" --group-config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\grouping.yml" --open-arms "Top,Bottom" --closed-arms "Left,Right"
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
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

Single video mode:

```bash
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml" --video "1-5.mp4"
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
pixi run -e location-tracker python visualization/GenerateTransformedTrajectoryHeatmap.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml" --vertex-mode gui --save-picked-vertices
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

- `LocationTracking_Functions.py`
- `RunLocationTrackingFromYAML.py`
- `CreateLocationTrackingYAMLTemplate.py`
- `BuildTrackingConfigGUI.py`
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
