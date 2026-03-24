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

### Option A: Simple CLI tracking

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

### Option B: YAML tracking (recommended)

#### Step B1 Generate YAML template

Script: `CreateLocationTrackingYAMLTemplate.py`

```bash
pixi run -e location-tracker python CreateLocationTrackingYAMLTemplate.py --output ".\project_tracking_config.yml" --video-dir "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later"
```

#### Step B2 Edit YAML in GUI (optional)

Script: `BuildTrackingConfigGUI.py`

```bash
pixi run -e location-tracker python BuildTrackingConfigGUI.py
```

#### Step B3 Run tracking from YAML

Script: `RunLocationTrackingFromYAML.py`

```bash
pixi run -e location-tracker python RunLocationTrackingFromYAML.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

YAML run options:

- `run.parallel`
- `run.n_processes` (`null` = auto)
- `run.accept_p_frames`

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
