"""
Shared dual-camera merge logic: folder pairing, ffmpeg filter, jobs JSON, batch merge.

Used by MergeDualCameraVideosGUI.py (writes plan) and MergeDualCameraVideos.py (runs ffmpeg).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2


def _stem_sort_key(filename: str) -> tuple:
    stem = Path(filename).stem
    key: list[Any] = []
    for part in stem.split("-"):
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def discover_pairs(dir_a: Path, dir_b: Path) -> list[tuple[str, str, str]]:
    """Same basename in both folders; sorted by filename stem."""
    if not dir_a.is_dir() or not dir_b.is_dir():
        return []
    names = sorted({p.name for p in dir_a.glob("*.mp4") if p.is_file()}, key=_stem_sort_key)
    out: list[tuple[str, str, str]] = []
    for name in names:
        pa, pb = dir_a / name, dir_b / name
        if pb.is_file():
            out.append((str(pa.resolve()), str(pb.resolve()), name))
    return out


def load_paths_config(config_path: Path) -> dict[str, str]:
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    for k in ("dir_video_a_cam3_left", "dir_video_b_cam2_right", "output_dir"):
        if k not in data or not str(data[k]).strip():
            raise KeyError(f"Missing or empty key: {k}")
    return {k: str(Path(str(data[k])).expanduser()) for k in data}


JOBS_VERSION = 1


def default_jobs_path(repo_root: Path | None = None) -> Path:
    base = repo_root if repo_root is not None else Path(__file__).resolve().parent
    return base / "dual_camera_merge_jobs.json"


def build_jobs_payload(
    dir_a: str,
    dir_b: str,
    output_dir: str,
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": JOBS_VERSION,
        "dir_video_a_cam3_left": dir_a,
        "dir_video_b_cam2_right": dir_b,
        "output_dir": output_dir,
        "clips": clips,
    }


def clips_from_alignments(
    pairs: list[tuple[str, str, str]],
    alignments: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _pa, _pb, name in pairs:
        sa, sb = alignments.get(name, (0, 0))
        out.append(
            {
                "filename": name,
                "start_frame_a": int(sa),
                "start_frame_b": int(sb),
            }
        )
    return out


def load_jobs_file(jobs_path: Path) -> dict[str, Any]:
    with open(jobs_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Jobs file must be a JSON object.")
    ver = data.get("version", 1)
    if ver != JOBS_VERSION:
        raise ValueError(f"Unsupported jobs version: {ver}")
    for k in ("dir_video_a_cam3_left", "dir_video_b_cam2_right", "output_dir", "clips"):
        if k not in data:
            raise KeyError(f"Missing key in jobs file: {k}")
    if not isinstance(data["clips"], list):
        raise ValueError("clips must be a list.")
    return data


def save_jobs_file(jobs_path: Path, payload: dict[str, Any]) -> None:
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def alignments_from_jobs(data: dict[str, Any]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for c in data.get("clips", []):
        if not isinstance(c, dict):
            continue
        fn = c.get("filename")
        if not fn:
            continue
        try:
            sa = int(c["start_frame_a"])
            sb = int(c["start_frame_b"])
        except (KeyError, TypeError, ValueError):
            continue
        out[str(fn)] = (sa, sb)
    return out


def probe_video(path: str) -> Optional[Tuple[int, float, int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n <= 0:
            return None
        return (n, fps, w, h)
    finally:
        cap.release()


def intersection_length(n_a: int, n_b: int, start_a: int, start_b: int) -> int:
    return max(0, min(n_a - start_a, n_b - start_b))


def build_filter_complex(
    start_a: int,
    start_b: int,
    n_a: int,
    n_b: int,
    transpose_left: int = 1,
    transpose_right: int = 2,
) -> str:
    t = intersection_length(n_a, n_b, start_a, start_b)
    if t <= 0:
        raise ValueError("Empty intersection: increase available frames or lower start indices.")
    end_a = start_a + t
    end_b = start_b + t
    return (
        f"[0:v]trim=start_frame={start_a}:end_frame={end_a},setpts=PTS-STARTPTS[va];"
        f"[1:v]trim=start_frame={start_b}:end_frame={end_b},setpts=PTS-STARTPTS[vb];"
        f"[va]transpose={transpose_left}[v0];"
        f"[vb]transpose={transpose_right}[v1];"
        f"[v0][v1]hstack=inputs=2[v]"
    )


def run_ffmpeg_merge(
    path_a: str,
    path_b: str,
    out_path: str,
    start_a: int,
    start_b: int,
    n_a: int,
    n_b: int,
) -> Tuple[bool, str]:
    try:
        fc = build_filter_complex(start_a, start_b, n_a, n_b)
    except ValueError as e:
        return False, str(e)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path_a,
        "-i",
        path_b,
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "ffmpeg not found in PATH."
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
        return False, err
    return True, ""


def merge_one_clip(
    dir_a: Path,
    dir_b: Path,
    output_dir: Path,
    filename: str,
    start_a: int,
    start_b: int,
) -> Tuple[bool, str]:
    pa = dir_a / filename
    pb = dir_b / filename
    out = output_dir / filename
    if not pa.is_file() or not pb.is_file():
        return False, f"missing input: {filename}"
    ia = probe_video(str(pa))
    ib = probe_video(str(pb))
    if not ia or not ib:
        return False, f"could not read video: {filename}"
    n_a, _, _, _ = ia
    n_b, _, _, _ = ib
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_ffmpeg_merge(str(pa), str(pb), str(out), start_a, start_b, n_a, n_b)


def merge_all_from_jobs(
    jobs_path: Path,
    *,
    only_filename: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """
    Returns process exit code: 0 if all merges OK, 1 if any failure or invalid jobs.
    """
    if not shutil.which("ffmpeg") and not dry_run:
        print("ffmpeg not found in PATH.", file=sys.stderr)
        return 1
    data = load_jobs_file(jobs_path)
    dir_a = Path(data["dir_video_a_cam3_left"]).expanduser()
    dir_b = Path(data["dir_video_b_cam2_right"]).expanduser()
    output_dir = Path(data["output_dir"]).expanduser()

    errors = 0
    for c in data["clips"]:
        if not isinstance(c, dict):
            continue
        fn = c.get("filename")
        if not fn:
            continue
        if only_filename and fn != only_filename:
            continue
        try:
            sa = int(c["start_frame_a"])
            sb = int(c["start_frame_b"])
        except (KeyError, TypeError, ValueError) as e:
            print(f"[skip] {fn}: bad alignment fields ({e})")
            errors += 1
            continue
        pa, pb = dir_a / fn, dir_b / fn
        if not pa.is_file() or not pb.is_file():
            print(f"[skip] {fn}: input file not found under configured folders")
            errors += 1
            continue
        ia = probe_video(str(pa))
        ib = probe_video(str(pb))
        if not ia or not ib:
            print(f"[skip] {fn}: could not probe video")
            errors += 1
            continue
        n_a, _, _, _ = ia
        n_b, _, _, _ = ib
        L = intersection_length(n_a, n_b, sa, sb)
        if L <= 0:
            print(f"[fail] {fn}: empty intersection (start_a={sa}, start_b={sb}, n_a={n_a}, n_b={n_b})")
            errors += 1
            continue
        out = output_dir / fn
        if dry_run:
            print(f"[dry-run] would merge {fn} -> {out}  (L={L} frames)")
            continue
        ok, err = merge_one_clip(dir_a, dir_b, output_dir, fn, sa, sb)
        if ok:
            print(f"[ok] {fn} -> {out}")
        else:
            print(f"[fail] {fn}: {err[:500]}")
            errors += 1
    return 0 if errors == 0 else 1
