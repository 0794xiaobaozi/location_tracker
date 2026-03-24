# Location Tracker - Select Interval Module

This repository contains the modernized interval selection workflow used before batch tracking.

## What is included

- `SelectVideoIntervals.py`: interval selection tool with modern GUI support.
- `CropVideosFromIntervals.py`: batch crop tool using `video_intervals.json`.
- `pixi.toml`: reproducible environment configuration.

## Main features

- Single-window modern GUI (`customtkinter`) for selecting intervals across multiple videos.
- Keyboard shortcuts:
  - `S` set start
  - `E` set end (manual mode)
  - `Enter` confirm
  - `Q` skip
  - `Space` play/pause
  - Arrow keys for navigation
- Auto interval modes:
  - `--auto-5min`
  - `--auto-10min`
- Progress timeline with highlighted selected interval.

## Environment setup (pixi)

```bash
pixi install -e location-tracker
```

## Run

```bash
pixi run -e location-tracker python SelectVideoIntervals.py -d "F:\Neuro\ezTrack\LocationTracking\video\EPM_later" --auto-5min --gui modern
```

```bash
pixi run -e location-tracker python CropVideosFromIntervals.py --directory "F:\Neuro\ezTrack\LocationTracking\video\EPM_later"
```

## Notes

- Use `python` inside pixi (`pixi run ...`) instead of `py` to avoid interpreter mismatch.
- Video/data outputs are ignored by git via `.gitignore`.
