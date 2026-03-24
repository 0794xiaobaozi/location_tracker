# Location Tracker - Interval, Crop, and YAML Tracking

This repository publishes the following core tools:

1. `SelectVideoIntervals.py`: interactive interval selection for each video.
2. `CropVideosFromIntervals.py`: batch crop videos based on `video_intervals.json`.
3. `RunLocationTrackingBatch.py`: CLI batch location tracking (no notebook).
4. `CreateLocationTrackingYAMLTemplate.py`: generate a project YAML template.
5. `BuildTrackingConfigGUI.py`: GUI to create/edit tracking YAML.
6. `RunLocationTrackingFromYAML.py`: run full tracking directly from YAML.

---

## 1) Setup environment (pixi)

```bash
pixi install -e location-tracker
```

Use `pixi run ... python ...` (or `pixi shell -e location-tracker` then `python ...`).
Avoid `py ...` to prevent interpreter mismatch.

---

## 2) Function A: Select intervals

### What it does

- Scans a video directory.
- Lets you choose `start_frame` and `end_frame` (or auto-duration from start).
- Saves results to `video_intervals.json` in the same directory (unless `--output` is provided).

### Recommended command (modern GUI + auto 5 min)

```bash
pixi run -e location-tracker python SelectVideoIntervals.py -d "F:\Neuro\ezTrack\LocationTracking\video\EPM_later" --auto-5min --gui modern
```

### Common options

- `--gui modern|classic` GUI backend (default: `modern`)
- `--auto-5min` auto end = start + 5 minutes
- `--auto-10min` auto end = start + 10 minutes
- `--exclude <file>` skip specific video file (can repeat)
- `--output <path>` custom JSON output path

### Shortcuts (modern GUI)

- `S`: set start
- `E`: set end (manual mode only)
- `Enter`: confirm current video
- `Q`: skip current video
- `Space`: play/pause
- `Left/Right`: -/+10 frames
- `Up/Down`: -/+100 frames
- `R`: reset interval

---

## 3) Function B: Crop videos from selected intervals

### What it does

- Reads `video_intervals.json`.
- Exports cropped segments for all matched videos.
- Writes output to `cropped_video` folder while keeping directory structure.

### Recommended command

```bash
pixi run -e location-tracker python CropVideosFromIntervals.py --directory "F:\Neuro\ezTrack\LocationTracking\video\EPM_later"
```

### Useful options

- `--directory` process one or more directories (repeatable)
- `--recursive` recursively find all directories containing `video_intervals.json`
- `--intervals-file` specify a custom intervals JSON file
- `--output-dir` customize output folder name (default: `cropped_video`)

---

## 4) Typical workflow

1. Select intervals:
   `SelectVideoIntervals.py` -> generates `video_intervals.json`
2. Crop videos:
   `CropVideosFromIntervals.py` -> exports clipped videos to `cropped_video`
3. Track locations:
   `RunLocationTrackingBatch.py` -> generates `*_LocationOutput.csv` and `BatchSummary.csv`
4. Continue with downstream statistics/visualization.

---

## 5) Function C: Batch location tracking (CLI, no notebook)

### Recommended command

```bash
pixi run -e location-tracker python RunLocationTrackingBatch.py --directory "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later" --ftype mp4 --loc-thresh 99.0 --parallel
```

### Useful options

- `--parallel` enable multiprocessing
- `--n-processes` set process count (default: auto)
- `--loc-thresh` tracking threshold percentile
- `--method abs|light|dark`
- `--use-window --window-size --window-weight`
- `--accept-p-frames` allow p-frame videos

---

## 6) Function D: YAML template generation

### What it does

- Creates a project config file for tracking runs.
- Pre-fills project/video fields and all major tracking options.

### Recommended command

```bash
pixi run -e location-tracker python CreateLocationTrackingYAMLTemplate.py --output ".\project_tracking_config.yml" --video-dir "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later"
```

---

## 7) Function E: Build/Edit YAML via GUI

### What it does

- Opens a modern GUI for configuring crop, analysis ROI, functional ROIs, scale, and tracking params.
- Saves settings to a YAML file used by the runner script.

### Recommended command

```bash
pixi run -e location-tracker python BuildTrackingConfigGUI.py
```

---

## 8) Function F: Run tracking from YAML

### What it does

- Loads one YAML file and runs batch location tracking end-to-end.
- Outputs `*_LocationOutput.csv` for each video and `BatchSummary.csv` for aggregate statistics.

### Recommended command

```bash
pixi run -e location-tracker python RunLocationTrackingFromYAML.py --config "F:\Neuro\ezTrack\LocationTracking\video\cropped_video\EPM_later\project_tracking_config.yml"
```

### Typical YAML run options

- `run.parallel`: enable multiprocessing
- `run.n_processes`: process count (`null` = auto)
- `run.accept_p_frames`: allow/disallow p-frame videos

---

## 9) Files required for YAML workflow

If you publish YAML workflow scripts, also include these dependencies in the same repository:

- `LocationTracking_Functions.py` (core tracking library, required)
- `RunLocationTrackingFromYAML.py` (YAML runner, required)
- `CreateLocationTrackingYAMLTemplate.py` (template generator)
- `BuildTrackingConfigGUI.py` (YAML builder GUI)
- `pixi.toml` / `pixi.lock` (reproducible environment)

---

## Notes

- Large local data files are ignored via `.gitignore`.
- If GUI does not appear, verify environment and imports first:

```bash
pixi run -e location-tracker python -c "import customtkinter, cv2; print('ok')"
```
