# Location Tracker - Interval Selection and Cropping

This repository currently publishes two core tools:

1. `SelectVideoIntervals.py`: interactive interval selection for each video.
2. `CropVideosFromIntervals.py`: batch crop videos based on `video_intervals.json`.

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
3. Continue with downstream tracking/statistics on cropped outputs.

---

## Notes

- Large local data files are ignored via `.gitignore`.
- If GUI does not appear, verify environment and imports first:

```bash
pixi run -e location-tracker python -c "import customtkinter, cv2; print('ok')"
```
