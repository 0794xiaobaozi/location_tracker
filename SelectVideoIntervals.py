#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频区间选择脚本：为批量处理中的每个视频选择分析区间

这个脚本可以扫描一个目录中的所有视频，逐个播放并让用户选择分析的开始和结束帧，
然后将结果保存到JSON文件中，供批量处理脚本使用。

使用方法:
    python SelectVideoIntervals.py --directory <folder_path> [--output <output_file>] [--exclude <video_file> ...]

参数:
    --directory, -d: 包含视频文件的目录（必需）
    --output, -o: 输出JSON文件路径（可选，默认为目录中的video_intervals.json）
    --exclude, -e: 要排除的视频文件（可选，可多次指定）

输出:
    - 生成一个JSON文件，包含每个视频的分析区间（start_frame, end_frame）
    - 默认文件名：video_intervals.json
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import subprocess
from typing import Dict, Optional, List

# 不再需要配置加载功能


def find_video_files(directory: str, extensions: List[str] = None, exclude_files: List[str] = None) -> List[str]:
    """
    在目录中查找视频文件
    
    Args:
        directory: 目录路径
        extensions: 视频文件扩展名列表
        exclude_files: 要排除的文件名列表（相对于directory或绝对路径）
    
    Returns:
        视频文件路径列表
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
    
    if exclude_files is None:
        exclude_files = []
    
    # 将排除文件转换为绝对路径集合以便比较
    exclude_paths = set()
    for exclude_file in exclude_files:
        if os.path.isabs(exclude_file):
            exclude_paths.add(os.path.normpath(os.path.normcase(exclude_file)))
        else:
            exclude_paths.add(os.path.normpath(os.path.normcase(os.path.join(directory, exclude_file))))
    
    video_files = []
    seen_files = set()
    
    import glob
    for ext in extensions:
        pattern = os.path.join(directory, '**', f'*{ext}')
        found = glob.glob(pattern, recursive=True)
        for f in found:
            normalized = os.path.normpath(os.path.normcase(f))
            if normalized not in seen_files and normalized not in exclude_paths:
                seen_files.add(normalized)
                video_files.append(f)
        
        if ext != ext.upper():
            pattern = os.path.join(directory, '**', f'*{ext.upper()}')
            found = glob.glob(pattern, recursive=True)
            for f in found:
                normalized = os.path.normpath(os.path.normcase(f))
                if normalized not in seen_files and normalized not in exclude_paths:
                    seen_files.add(normalized)
                    video_files.append(f)
    
    return sorted(video_files)


def get_video_info_ffprobe(video_path: str) -> Optional[Dict]:
    """
    使用 ffprobe 获取视频信息（更准确）
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        包含视频信息的字典，如果失败则返回 None
    """
    try:
        # 获取视频时长（秒）
        cmd_duration = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
        duration = float(result_duration.stdout.strip())
        
        # 获取 FPS
        cmd_fps = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result_fps = subprocess.run(cmd_fps, capture_output=True, text=True, check=True)
        fps_str = result_fps.stdout.strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_str) if fps_str else 0
        
        # 获取分辨率
        cmd_size = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result_size = subprocess.run(cmd_size, capture_output=True, text=True, check=True)
        lines = result_size.stdout.strip().split('\n')
        width = int(lines[0]) if len(lines) > 0 and lines[0] else 0
        height = int(lines[1]) if len(lines) > 1 and lines[1] else 0
        
        total_frames = int(duration * fps) if fps > 0 else 0
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def validate_video_info(video_path: str, video_info: Dict) -> bool:
    """
    验证视频信息是否合理（基于文件大小）
    
    Args:
        video_path: 视频文件路径
        video_info: 视频信息字典
    
    Returns:
        如果信息合理返回 True，否则返回 False
    """
    try:
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        duration = video_info['duration']
        duration_minutes = duration / 60
        total_frames = video_info['total_frames']
        fps = video_info['fps']
        
        # 如果文件太小或时长为0，无法验证
        if file_size_mb < 1 or duration <= 0:
            return True
        
        # 方法1: 基于码率估算（适用于大多数视频）
        # 假设码率在 1-20 Mbps 之间
        min_bitrate_mbps = 1.0
        max_bitrate_mbps = 20.0
        min_duration_estimate = (file_size_mb * 8) / (max_bitrate_mbps * 60)  # 分钟
        max_duration_estimate = (file_size_mb * 8) / (min_bitrate_mbps * 60)  # 分钟
        
        # 方法2: 与其他同目录视频对比（如果可能）
        # 这里我们使用更严格的验证：如果检测时长超过估算最大时长的3倍，认为异常
        if duration_minutes > max_duration_estimate * 3:
            print(f"\n[WARNING] ⚠️  视频时长检测可能异常:")
            print(f"  文件: {os.path.basename(video_path)}")
            print(f"  检测到的时长: {format_time(duration)} ({duration_minutes:.1f} 分钟)")
            print(f"  文件大小: {file_size_mb:.2f} MB")
            print(f"  基于文件大小估算的合理时长范围: {min_duration_estimate:.1f} - {max_duration_estimate:.1f} 分钟")
            print(f"  检测到的总帧数: {total_frames:,} 帧")
            print(f"  帧率: {fps:.2f} fps")
            print(f"  码率估算: {file_size_mb * 8 / duration_minutes:.2f} Mbps (异常低，正常应为 5-15 Mbps)")
            print(f"\n[建议] 视频文件元数据可能损坏或异常。")
            print(f"       如果视频实际时长明显短于检测值，建议:")
            print(f"       1. 检查视频文件是否完整")
            print(f"       2. 尝试使用 ffprobe 验证: ffprobe \"{video_path}\"")
            print(f"       3. 如果确认异常，可以跳过该视频（按 'q'）")
            return False
        
        return True
    except Exception:
        return True  # 如果验证失败，假设信息合理


def get_video_info(video_path: str) -> Dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        包含视频信息的字典
    """
    # 首先尝试使用 ffprobe（更准确）
    ffprobe_info = get_video_info_ffprobe(video_path)
    if ffprobe_info:
        # 验证 ffprobe 的结果
        if not validate_video_info(video_path, ffprobe_info):
            # 如果验证失败，仍然返回信息，但已经打印了警告
            pass
        return ffprobe_info
    
    # 如果 ffprobe 不可用，使用 OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    # 验证 OpenCV 的结果
    validate_video_info(video_path, video_info)
    
    return video_info


def format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# 全局变量用于trackbar回调
_current_frame = [0]
_total_frames = [0]
_start_frame = [None]
_end_frame = [None]
_paused = [True]  # 默认暂停
_cap_ref = [None]
_frame_cache = {}  # 帧缓存：{frame_num: frame_data}
_cache_size = 50  # 缓存最近50帧
_last_seek_frame = [-1]  # 上次实际跳转的帧号
_trackbar_update_time = [0]  # trackbar更新时间戳（用于节流）
_modern_window_geometry = [None]  # 现代GUI窗口几何信息（复用大小和位置）

def on_trackbar(pos):
    """Trackbar回调函数 - 只更新目标帧，不立即跳转（由主循环处理）"""
    _current_frame[0] = pos
    _trackbar_update_time[0] = cv2.getTickCount()  # 记录更新时间

def on_mouse(event, x, y, flags, param):
    """鼠标回调函数 - 用于点击时间轴"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查是否点击在时间轴上
        frame_height, frame_width = param['frame_shape'][:2]
        timeline_y = frame_height - 30
        timeline_width = frame_width - 20
        timeline_x = 10
        
        if timeline_y - 20 <= y <= timeline_y + 20:
            # 点击在时间轴区域
            rel_x = x - timeline_x
            if 0 <= rel_x <= timeline_width:
                new_frame = int((rel_x / timeline_width) * _total_frames[0])
                new_frame = max(0, min(new_frame, _total_frames[0] - 1))
                _current_frame[0] = new_frame
                _trackbar_update_time[0] = cv2.getTickCount()  # 记录更新时间
                # 不立即跳转，由主循环处理（带节流）

def select_interval(video_path: str, video_info: Dict, auto_duration_secs: Optional[float] = None) -> Optional[Dict]:
    """
    播放视频并让用户选择分析区间
    
    Args:
        video_path: 视频文件路径
        video_info: 视频信息字典
        auto_duration_secs: 若提供，则只需选择start_frame，end_frame将自动设为start_frame+时长
    
    Returns:
        包含start_frame和end_frame的字典，如果跳过则返回None
    """
    global _current_frame, _total_frames, _start_frame, _end_frame, _paused, _cap_ref
    
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    
    # 初始化全局变量
    _total_frames[0] = total_frames
    _current_frame[0] = 0
    _start_frame[0] = None
    _end_frame[0] = None
    _paused[0] = True  # 默认暂停
    _frame_cache.clear()  # 清理帧缓存
    _last_seek_frame[0] = -1  # 重置上次跳转帧
    _trackbar_update_time[0] = cv2.getTickCount()  # 初始化更新时间
    
    print("\n" + "="*70)
    print(f"Video: {os.path.basename(video_path)}")
    print("="*70)
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {format_time(video_info['duration'])}")
    print("\nControls:")
    print("  - Trackbar: Drag to navigate to any frame")
    print("  - Click on timeline: Jump to clicked position")
    print("  - Arrow keys: Navigate frames (Left/Right: ±10 frames, Up/Down: ±100 frames)")
    print("  - 's': Set start frame (current position)")
    if auto_duration_secs is None:
        print("  - 'e': Set end frame (current position)")
    print("  - 'r': Reset (clear start and end)")
    print("  - 'Space': Play/Pause")
    print("  - 'q': Skip this video")
    print("  - 'Enter': Confirm and continue to next video")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None
    
    _cap_ref[0] = cap
    
    window_name = "Select Interval - Press 'q' to skip, Enter to confirm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 创建trackbar
    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, on_trackbar)
    
    # 获取第一帧以确定尺寸
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"[ERROR] Could not read first frame")
        return None
    
    frame_shape = first_frame.shape
    cv2.setMouseCallback(window_name, on_mouse, {'frame_shape': frame_shape})
    
    last_frame_num = -1
    last_display_frame = -1
    
    def get_frame(frame_num):
        """获取帧，优先使用缓存"""
        # 检查缓存
        if frame_num in _frame_cache:
            return _frame_cache[frame_num].copy()
        
        # 如果不在缓存中，读取帧
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 如果目标帧距离当前位置较远，直接跳转
        # 如果较近，可以顺序读取（更快）
        if abs(frame_num - current_pos) > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # 更新缓存
        if len(_frame_cache) >= _cache_size:
            # 删除最旧的帧（FIFO）
            oldest_frame = min(_frame_cache.keys())
            del _frame_cache[oldest_frame]
        
        _frame_cache[frame_num] = frame.copy()
        return frame
    
    while True:
        current_time = cv2.getTickCount()
        
        # 节流trackbar更新：如果用户正在拖动trackbar，延迟跳转
        # 只有在trackbar停止移动一段时间后才实际跳转
        should_seek = False
        if _trackbar_update_time[0] == 0:
            # 立即更新（方向键等）
            if _current_frame[0] != _last_seek_frame[0]:
                should_seek = True
                _last_seek_frame[0] = _current_frame[0]
            # 重置时间戳，避免影响后续节流
            _trackbar_update_time[0] = current_time
        else:
            time_since_trackbar_update = (current_time - _trackbar_update_time[0]) / cv2.getTickFrequency()
            
            # 如果trackbar最近更新过（0.3秒内），且目标帧与当前显示帧不同，等待
            # 否则，如果目标帧与上次跳转的帧不同，执行跳转
            if time_since_trackbar_update > 0.3:  # 用户停止拖动0.3秒后
                if _current_frame[0] != _last_seek_frame[0]:
                    should_seek = True
                    _last_seek_frame[0] = _current_frame[0]
            elif _current_frame[0] == last_display_frame:
                # 如果目标帧就是当前显示的帧，不需要跳转
                should_seek = False
            else:
                # 如果用户正在拖动，但目标帧变化很大，也跳转（避免等待太久）
                if abs(_current_frame[0] - last_display_frame) > 1000:
                    should_seek = True
                    _last_seek_frame[0] = _current_frame[0]
        
        # 更新trackbar位置显示
        if _current_frame[0] != last_frame_num:
            cv2.setTrackbarPos('Frame', window_name, _current_frame[0])
            last_frame_num = _current_frame[0]
        
        # 读取当前帧（使用缓存优化）
        if should_seek or last_display_frame != _current_frame[0]:
            frame = get_frame(_current_frame[0])
            if frame is None:
                _current_frame[0] = max(0, _current_frame[0] - 1)
                continue
            last_display_frame = _current_frame[0]
        else:
            # 如果不需要跳转，使用缓存的帧
            if last_display_frame in _frame_cache:
                frame = _frame_cache[last_display_frame].copy()
            else:
                frame = get_frame(_current_frame[0])
                if frame is None:
                    _current_frame[0] = max(0, _current_frame[0] - 1)
                    continue
                last_display_frame = _current_frame[0]
        
        # 在帧上绘制信息和控制界面
        frame_display = frame.copy()
        h, w = frame_display.shape[:2]
        
        # 绘制半透明信息面板
        overlay = frame_display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_display, 0.4, 0, frame_display)
        
        # 显示当前帧信息
        current_time = _current_frame[0] / fps if fps > 0 else 0
        progress = (_current_frame[0] / total_frames * 100) if total_frames > 0 else 0
        info_text = f"Frame: {_current_frame[0]}/{total_frames-1} | Time: {format_time(current_time)} | Progress: {progress:.1f}%"
        cv2.putText(frame_display, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示已选择的区间
        y_offset = 55
        if _start_frame[0] is not None:
            start_time = _start_frame[0] / fps if fps > 0 else 0
            start_text = f"[SET] START: Frame {_start_frame[0]} ({format_time(start_time)})"
            cv2.putText(frame_display, start_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
            if auto_duration_secs is not None:
                duration_frames = int(auto_duration_secs * fps) if fps > 0 else 0
                auto_end = min(total_frames - 1, _start_frame[0] + duration_frames - 1) if duration_frames > 0 else _start_frame[0]
                auto_end_time = auto_end / fps if fps > 0 else 0
                end_text = f"[AUTO] END: Frame {auto_end} ({format_time(auto_end_time)}) (+{auto_duration_secs/60:.1f} min)"
                cv2.putText(frame_display, end_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
        else:
            start_text = "[NOT SET] Press 'S' to set START frame"
            cv2.putText(frame_display, start_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y_offset += 30
        
        if _end_frame[0] is not None and auto_duration_secs is None:
            end_time = _end_frame[0] / fps if fps > 0 else 0
            end_text = f"[SET] END: Frame {_end_frame[0]} ({format_time(end_time)})"
            cv2.putText(frame_display, end_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        elif auto_duration_secs is None:
            end_text = "[NOT SET] Press 'E' to set END frame"
            cv2.putText(frame_display, end_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y_offset += 30
        else:
            end_text = f"[AUTO] END = START + {auto_duration_secs/60:.1f} min"
            cv2.putText(frame_display, end_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            y_offset += 30
        
        if _start_frame[0] is not None and _end_frame[0] is not None:
            interval_frames = _end_frame[0] - _start_frame[0] + 1
            interval_time = interval_frames / fps if fps > 0 else 0
            interval_text = f"INTERVAL: {interval_frames} frames ({format_time(interval_time)})"
            cv2.putText(frame_display, interval_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
            # 提示可以确认
            confirm_text = "Press 'ENTER' to confirm and continue"
            cv2.putText(frame_display, confirm_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示播放状态
        status_text = "[PAUSED]" if _paused[0] else "[PLAYING]"
        cv2.putText(frame_display, status_text, (w - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 在右侧显示控制提示
        controls_y = 55
        controls = [
            "Controls:",
            "S - Set START",
        ]
        if auto_duration_secs is None:
            controls.append("E - Set END")
        controls.extend([
            "R - Reset",
            "Space/P - Play/Pause",
            "Arrows - Navigate",
            "Q - Skip",
            "Enter - Confirm"
        ])
        for i, ctrl in enumerate(controls):
            cv2.putText(frame_display, ctrl, (w - 200, controls_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 绘制时间轴
        timeline_y = h - 50
        timeline_height = 40
        timeline_width = w - 20
        timeline_x = 10
        
        # 绘制时间轴背景
        cv2.rectangle(frame_display, 
                     (timeline_x, timeline_y), 
                     (timeline_x + timeline_width, timeline_y + timeline_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame_display, 
                     (timeline_x, timeline_y), 
                     (timeline_x + timeline_width, timeline_y + timeline_height),
                     (200, 200, 200), 2)
        
        # 绘制选中的区间（如果已选择）
        if _start_frame[0] is not None and _end_frame[0] is not None:
            start_pos = int(timeline_x + (_start_frame[0] / total_frames) * timeline_width)
            end_pos = int(timeline_x + (_end_frame[0] / total_frames) * timeline_width)
            # 绘制选中区间背景
            cv2.rectangle(frame_display,
                         (start_pos, timeline_y),
                         (end_pos, timeline_y + timeline_height),
                         (0, 255, 0), -1)
            # 绘制开始和结束标记
            cv2.line(frame_display,
                    (start_pos, timeline_y - 5),
                    (start_pos, timeline_y + timeline_height + 5),
                    (0, 0, 255), 3)
            cv2.line(frame_display,
                    (end_pos, timeline_y - 5),
                    (end_pos, timeline_y + timeline_height + 5),
                    (0, 0, 255), 3)
        
        # 绘制当前帧位置指示器
        current_pos = int(timeline_x + (_current_frame[0] / total_frames) * timeline_width)
        cv2.line(frame_display,
                (current_pos, timeline_y - 10),
                (current_pos, timeline_y + timeline_height + 10),
                (255, 255, 255), 3)
        # 绘制当前位置三角形
        triangle_points = np.array([
            [current_pos, timeline_y - 10],
            [current_pos - 8, timeline_y - 20],
            [current_pos + 8, timeline_y - 20]
        ], np.int32)
        cv2.fillPoly(frame_display, [triangle_points], (255, 255, 255))
        
        # 在时间轴上绘制刻度（每10%一个）
        for i in range(0, 11):
            tick_pos = int(timeline_x + (i / 10) * timeline_width)
            tick_label = f"{i * 10}%"
            cv2.line(frame_display,
                    (tick_pos, timeline_y + timeline_height),
                    (tick_pos, timeline_y + timeline_height + 5),
                    (150, 150, 150), 1)
            cv2.putText(frame_display, tick_label, (tick_pos - 15, timeline_y + timeline_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow(window_name, frame_display)
        
        # 处理键盘输入
        # 使用waitKeyEx来更好地处理方向键（Windows）
        if _paused[0]:
            key = cv2.waitKeyEx(30)  # 使用waitKeyEx获取完整键码
        else:
            key = cv2.waitKeyEx(int(1000 / fps / 2))
        
        # 如果按键为-1（无按键），继续循环
        if key == -1:
            # 自动播放逻辑
            if not _paused[0]:
                next_frame = (_current_frame[0] + 1) % total_frames
                if next_frame in _frame_cache or abs(next_frame - last_display_frame) <= 1:
                    _current_frame[0] = next_frame
                else:
                    ret, frame = cap.read()
                    if ret:
                        next_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        if next_frame >= 0:
                            _current_frame[0] = next_frame
                            if len(_frame_cache) >= _cache_size:
                                oldest_frame = min(_frame_cache.keys())
                                del _frame_cache[oldest_frame]
                            _frame_cache[_current_frame[0]] = frame.copy()
                            last_display_frame = _current_frame[0]
                    else:
                        _current_frame[0] = (_current_frame[0] + 1) % total_frames
            continue
        
        # 提取完整键码和低8位
        key_full = key & 0xFFFFFFFF  # 完整32位键码
        key_char = key & 0xFF  # 低8位用于普通字符键
        
        if key_char == ord('q'):
            # 跳过这个视频
            cv2.destroyWindow(window_name)
            cap.release()
            print("[SKIP] Video skipped")
            return None
        
        elif key_char == 13:  # Enter
            # 确认并继续
            if auto_duration_secs is not None:
                if _start_frame[0] is None:
                    print("[WARNING] Please set start frame before confirming")
                else:
                    duration_frames = int(auto_duration_secs * fps) if fps > 0 else 0
                    auto_end = min(total_frames - 1, _start_frame[0] + duration_frames - 1) if duration_frames > 0 else _start_frame[0]
                    cv2.destroyWindow(window_name)
                    cap.release()
                    result = {
                        'start_frame': int(_start_frame[0]),
                        'end_frame': int(auto_end)
                    }
                    print(f"[OK][AUTO] Interval selected: Frames {_start_frame[0]} to {auto_end} (+{auto_duration_secs/60:.1f} min)")
                    return result
            else:
                if _start_frame[0] is not None and _end_frame[0] is not None:
                    if _start_frame[0] > _end_frame[0]:
                        print("[WARNING] Start frame > End frame, swapping...")
                        _start_frame[0], _end_frame[0] = _end_frame[0], _start_frame[0]
                    cv2.destroyWindow(window_name)
                    cap.release()
                    result = {
                        'start_frame': int(_start_frame[0]),
                        'end_frame': int(_end_frame[0])
                    }
                    print(f"[OK] Interval selected: Frames {_start_frame[0]} to {_end_frame[0]}")
                    return result
                else:
                    print("[WARNING] Please set both start and end frames before confirming")
        
        elif key_char == ord('s') or key_char == ord('S'):
            # 设置开始帧
            _start_frame[0] = _current_frame[0]
            print(f"[SET] Start frame: {_start_frame[0]} ({format_time(_current_frame[0] / fps)})")
            if auto_duration_secs is not None:
                duration_frames = int(auto_duration_secs * fps) if fps > 0 else 0
                _end_frame[0] = min(total_frames - 1, _start_frame[0] + duration_frames - 1) if duration_frames > 0 else _start_frame[0]
                print(f"[AUTO] End frame set to {_end_frame[0]} (+{auto_duration_secs/60:.1f} min)")
        
        elif (key_char == ord('e') or key_char == ord('E')) and auto_duration_secs is None:
            # 设置结束帧
            _end_frame[0] = _current_frame[0]
            print(f"[SET] End frame: {_end_frame[0]} ({format_time(_current_frame[0] / fps)})")
        
        elif key_char == ord('r') or key_char == ord('R'):
            # 重置
            _start_frame[0] = None
            _end_frame[0] = None
            print("[RESET] Cleared start and end frames")
        
        elif key_char == ord(' ') or key_char == ord('p') or key_char == ord('P'):
            # 暂停/继续
            _paused[0] = not _paused[0]
            print(f"[{'PAUSED' if _paused[0] else 'PLAYING'}]")
        
        # 处理方向键 - 改进的检测逻辑
        # Windows方向键键码（多种可能的格式）
        # 左箭头: 0x250000 (2424832), 0x25000000
        # 右箭头: 0x270000 (2555904), 0x27000000
        # 上箭头: 0x260000 (2490368), 0x26000000
        # 下箭头: 0x280000 (2621440), 0x28000000
        
        # 提取方向键的高位部分（第16-23位）
        arrow_code = (key_full >> 16) & 0xFF
        
        # 左箭头 - 向后10帧
        if arrow_code == 0x25 or key_full == 2424832 or (key_full & 0xFFFF0000) == 0x25000000:
            _current_frame[0] = max(0, _current_frame[0] - 10)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ← 向后 10 帧 → 帧 {_current_frame[0]}")
        
        # 右箭头 - 向前10帧
        elif arrow_code == 0x27 or key_full == 2555904 or (key_full & 0xFFFF0000) == 0x27000000:
            _current_frame[0] = min(total_frames - 1, _current_frame[0] + 10)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] → 向前 10 帧 → 帧 {_current_frame[0]}")
        
        # 上箭头 - 向后100帧
        elif arrow_code == 0x26 or key_full == 2490368 or (key_full & 0xFFFF0000) == 0x26000000:
            _current_frame[0] = max(0, _current_frame[0] - 100)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ↑ 向后 100 帧 → 帧 {_current_frame[0]}")
        
        # 下箭头 - 向前100帧
        elif arrow_code == 0x28 or key_full == 2621440 or (key_full & 0xFFFF0000) == 0x28000000:
            _current_frame[0] = min(total_frames - 1, _current_frame[0] + 100)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ↓ 向前 100 帧 → 帧 {_current_frame[0]}")
        
        # Linux/其他系统的方向键处理（备用）
        elif key_char == 81 or key_char == 2:  # Left arrow (Linux)
            _current_frame[0] = max(0, _current_frame[0] - 10)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ← 向后 10 帧 → 帧 {_current_frame[0]}")
        
        elif key_char == 82 or key_char == 0:  # Up arrow (Linux)
            _current_frame[0] = max(0, _current_frame[0] - 100)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ↑ 向后 100 帧 → 帧 {_current_frame[0]}")
        
        elif key_char == 83 or key_char == 3:  # Right arrow (Linux)
            _current_frame[0] = min(total_frames - 1, _current_frame[0] + 10)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] → 向前 10 帧 → 帧 {_current_frame[0]}")
        
        elif key_char == 84 or key_char == 1:  # Down arrow (Linux)
            _current_frame[0] = min(total_frames - 1, _current_frame[0] + 100)
            _paused[0] = True
            _last_seek_frame[0] = _current_frame[0]
            _trackbar_update_time[0] = 0
            print(f"[NAV] ↓ 向前 100 帧 → 帧 {_current_frame[0]}")
        
        # 自动播放（如果未暂停）
        if not _paused[0]:
            # 顺序读取下一帧（比跳转快）
            next_frame = (_current_frame[0] + 1) % total_frames
            # 如果下一帧在缓存中或距离很近，直接读取
            if next_frame in _frame_cache or abs(next_frame - last_display_frame) <= 1:
                _current_frame[0] = next_frame
            else:
                # 否则顺序读取
                ret, frame = cap.read()
                if ret:
                    next_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if next_frame >= 0:
                        _current_frame[0] = next_frame
                        # 更新缓存
                        if len(_frame_cache) >= _cache_size:
                            oldest_frame = min(_frame_cache.keys())
                            del _frame_cache[oldest_frame]
                        _frame_cache[_current_frame[0]] = frame.copy()
                        last_display_frame = _current_frame[0]
                else:
                    _current_frame[0] = (_current_frame[0] + 1) % total_frames
    
    cap.release()
    cv2.destroyAllWindows()
    return None


class ModernIntervalSelector:
    """Single-window modern selector with keyboard shortcuts and interval highlight."""

    def __init__(self):
        import tkinter as tk
        import customtkinter as ctk
        from PIL import Image

        self.tk = tk
        self.ctk = ctk
        self.Image = Image
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0
        self.duration_frames = None
        self.auto_duration_secs = None

        self.state = {'current_frame': 0, 'start_frame': None, 'end_frame': None, 'playing': False, 'result': None}

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        if _modern_window_geometry[0]:
            self.root.geometry(_modern_window_geometry[0])
        else:
            self.root.geometry("1220x860")
        self.root.minsize(1000, 720)
        self._ensure_window_visible()
        # Bring window to front at startup; disable topmost shortly after.
        self.root.attributes("-topmost", True)
        self.root.after(1200, lambda: self.root.attributes("-topmost", False))

        self.header = ctk.CTkLabel(self.root, text="", anchor="w", font=("Segoe UI", 16, "bold"))
        self.header.pack(fill="x", padx=12, pady=(10, 8))

        # Keep bottom controls always visible; preview takes remaining space.
        self.bottom_panel = ctk.CTkFrame(self.root, fg_color="transparent")
        self.bottom_panel.pack(side="bottom", fill="x", padx=0, pady=0)

        self.preview_label = ctk.CTkLabel(self.root, text="")
        self.preview_label.pack(side="top", fill="both", expand=True, padx=12, pady=6)

        self.status_var = ctk.StringVar(value="")
        self.progress_var = ctk.StringVar(value="")
        self.hint_var = ctk.StringVar(value="")
        ctk.CTkLabel(self.bottom_panel, textvariable=self.progress_var, anchor="w").pack(fill="x", padx=12)
        ctk.CTkLabel(self.bottom_panel, textvariable=self.status_var, anchor="w").pack(fill="x", padx=12, pady=(4, 0))
        ctk.CTkLabel(self.bottom_panel, textvariable=self.hint_var, anchor="w", text_color="#B0B0B0").pack(fill="x", padx=12, pady=(0, 6))

        self.slider = ctk.CTkSlider(self.bottom_panel, from_=0, to=1, number_of_steps=1, command=self._on_slider)
        self.slider.pack(fill="x", padx=12, pady=(2, 2))

        self.timeline = tk.Canvas(self.bottom_panel, height=22, bg="#2A2A2A", highlightthickness=0)
        self.timeline.pack(fill="x", padx=12, pady=(0, 8))
        self.timeline.bind("<Configure>", lambda _e: self._redraw_timeline())

        controls = ctk.CTkFrame(self.bottom_panel)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(controls, text="<< 100", width=80, command=lambda: self._jump(-100)).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="< 10", width=70, command=lambda: self._jump(-10)).pack(side="left", padx=4, pady=8)
        self.play_btn = ctk.CTkButton(controls, text="Play", width=80, command=self._toggle_play)
        self.play_btn.pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="10 >", width=70, command=lambda: self._jump(10)).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="100 >>", width=80, command=lambda: self._jump(100)).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="Set START [S]", width=130, command=self._set_start).pack(side="left", padx=(16, 4), pady=8)
        self.end_btn = ctk.CTkButton(controls, text="Set END [E]", width=120, command=self._set_end)
        self.end_btn.pack(side="left", padx=4, pady=8)
        self.auto_hint_label = ctk.CTkLabel(controls, text="", text_color="#D0D0D0")
        self.auto_hint_label.pack(side="left", padx=8)
        ctk.CTkButton(controls, text="Reset [R]", width=100, command=self._reset_range).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="Skip [Q]", width=100, fg_color="#A34343", hover_color="#8A3737", command=self._skip).pack(side="right", padx=4, pady=8)
        ctk.CTkButton(controls, text="Confirm [Enter]", width=130, fg_color="#2E8B57", hover_color="#256F46", command=self._confirm).pack(side="right", padx=4, pady=8)

        self.done_var = tk.BooleanVar(value=False)
        self.root.bind("<Left>", lambda _e: self._jump(-10))
        self.root.bind("<Right>", lambda _e: self._jump(10))
        self.root.bind("<Up>", lambda _e: self._jump(-100))
        self.root.bind("<Down>", lambda _e: self._jump(100))
        self.root.bind("<s>", lambda _e: self._set_start())
        self.root.bind("<S>", lambda _e: self._set_start())
        self.root.bind("<e>", lambda _e: self._set_end())
        self.root.bind("<E>", lambda _e: self._set_end())
        self.root.bind("<r>", lambda _e: self._reset_range())
        self.root.bind("<R>", lambda _e: self._reset_range())
        self.root.bind("<q>", lambda _e: self._skip())
        self.root.bind("<Q>", lambda _e: self._skip())
        self.root.bind("<Return>", lambda _e: self._confirm())
        self.root.bind("<space>", lambda _e: self._toggle_play())
        self.root.protocol("WM_DELETE_WINDOW", self._skip)
        self.root.after(40, self._playback_tick)

    def _ensure_window_visible(self):
        """Clamp saved geometry to current screen to avoid off-screen windows."""
        try:
            geom = self.root.geometry()  # e.g. 1200x800+100+50
            size_part, pos_part = geom.split("+", 1)
            w_str, h_str = size_part.split("x", 1)
            x_str, y_str = pos_part.split("+", 1)
            w, h, x, y = int(w_str), int(h_str), int(x_str), int(y_str)
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            x = min(max(0, x), max(0, sw - min(w, sw)))
            y = min(max(0, y), max(0, sh - min(h, sh)))
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            # Fallback to a known-visible default.
            self.root.geometry("1220x860+50+50")

    def _clamp(self, idx: int) -> int:
        return max(0, min(self.total_frames - 1, int(idx)))

    def _read_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(1160 / max(1, w), 620 / max(1, h), 1.0)
        if scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame_rgb

    def _redraw_timeline(self):
        w = max(10, self.timeline.winfo_width())
        h = max(10, self.timeline.winfo_height())
        self.timeline.delete("all")
        self.timeline.create_rectangle(2, 4, w - 2, h - 4, fill="#444444", outline="#666666")
        s, e = self.state['start_frame'], self.state['end_frame']
        if s is not None and e is not None and self.total_frames > 1:
            x1 = 2 + int((s / (self.total_frames - 1)) * (w - 4))
            x2 = 2 + int((e / (self.total_frames - 1)) * (w - 4))
            self.timeline.create_rectangle(min(x1, x2), 4, max(x1, x2), h - 4, fill="#2E8B57", outline="")
        if self.total_frames > 1:
            cur = self.state['current_frame']
            xc = 2 + int((cur / (self.total_frames - 1)) * (w - 4))
            self.timeline.create_line(xc, 2, xc, h - 2, fill="#FFFFFF", width=2)

    def _refresh_text(self):
        cur = self.state['current_frame']
        cur_time = cur / self.fps if self.fps > 0 else 0.0
        self.progress_var.set(f"Frame {cur}/{max(0, self.total_frames - 1)} | Time {format_time(cur_time)}")
        s, e = self.state['start_frame'], self.state['end_frame']
        s_text = f"{s} ({format_time(s / self.fps)})" if (s is not None and self.fps > 0) else "[NOT SET]"
        e_text = f"{e} ({format_time(e / self.fps)})" if (e is not None and self.fps > 0) else "[NOT SET]"
        self.status_var.set(f"Start: {s_text} | End: {e_text}")
        self._redraw_timeline()

    def _update_frame(self, idx=None):
        if idx is None:
            idx = self.state['current_frame']
        idx = self._clamp(idx)
        self.state['current_frame'] = idx
        self.slider.set(idx)
        frame_rgb = self._read_frame(idx)
        if frame_rgb is not None:
            pil_image = self.Image.fromarray(frame_rgb)
            img = self.ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
            self.preview_label.configure(image=img, text="")
            self.preview_label.image = img
        self._refresh_text()

    def _on_slider(self, value):
        self.state['playing'] = False
        self.play_btn.configure(text="Play")
        self._update_frame(int(float(value)))

    def _jump(self, delta):
        self.state['playing'] = False
        self.play_btn.configure(text="Play")
        self._update_frame(self.state['current_frame'] + delta)

    def _set_start(self):
        self.state['start_frame'] = self.state['current_frame']
        if self.duration_frames is not None:
            self.state['end_frame'] = self._clamp(self.state['start_frame'] + self.duration_frames - 1)
        self._refresh_text()

    def _set_end(self):
        if self.auto_duration_secs is not None:
            return
        self.state['end_frame'] = self.state['current_frame']
        self._refresh_text()

    def _reset_range(self):
        self.state['start_frame'] = None
        self.state['end_frame'] = None
        self._refresh_text()

    def _confirm(self):
        s, e = self.state['start_frame'], self.state['end_frame']
        if s is None:
            self.hint_var.set("Please set START first.")
            return
        if self.auto_duration_secs is None and e is None:
            self.hint_var.set("Please set END first.")
            return
        if self.auto_duration_secs is not None:
            e = self._clamp(s + self.duration_frames - 1) if self.duration_frames else s
        if s > e:
            s, e = e, s
        self.state['result'] = {'start_frame': int(s), 'end_frame': int(e)}
        self.done_var.set(True)

    def _skip(self):
        self.state['result'] = None
        self.done_var.set(True)

    def _toggle_play(self):
        self.state['playing'] = not self.state['playing']
        self.play_btn.configure(text="Pause" if self.state['playing'] else "Play")

    def _playback_tick(self):
        if self.state['playing'] and self.total_frames > 0:
            nxt = self.state['current_frame'] + 1
            if nxt >= self.total_frames:
                self.state['playing'] = False
                self.play_btn.configure(text="Play")
            else:
                self._update_frame(nxt)
        interval_ms = max(15, int(1000 / self.fps)) if self.fps > 0 else 40
        self.root.after(interval_ms, self._playback_tick)

    def select_interval(self, video_path: str, video_info: Dict, auto_duration_secs: Optional[float] = None) -> Optional[Dict]:
        print(f"[INFO] Opening modern GUI for: {os.path.basename(video_path)}")
        self.total_frames = int(video_info['total_frames'])
        self.fps = float(video_info['fps']) if video_info['fps'] else 0.0
        self.auto_duration_secs = auto_duration_secs
        self.duration_frames = int(auto_duration_secs * self.fps) if (auto_duration_secs is not None and self.fps > 0) else None

        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None

        self.root.title(f"Select Interval - {os.path.basename(video_path)}")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)
        self.root.after(800, lambda: self.root.attributes("-topmost", False))
        self.header.configure(text=f"{os.path.basename(video_path)} | Frames: {self.total_frames} | FPS: {self.fps:.2f} | Duration: {format_time(video_info['duration'])}")
        self.hint_var.set("Shortcuts: S set start, E set end, Enter confirm, Q skip, Space play/pause, Arrows move")
        self.state = {'current_frame': 0, 'start_frame': None, 'end_frame': None, 'playing': False, 'result': None}

        self.slider.configure(from_=0, to=max(1, self.total_frames - 1), number_of_steps=max(1, self.total_frames - 1))
        if auto_duration_secs is None:
            self.end_btn.pack(side="left", padx=4, pady=8)
            self.auto_hint_label.configure(text="")
        else:
            self.end_btn.pack_forget()
            self.auto_hint_label.configure(text=f"END = START + {auto_duration_secs / 60:.1f} min")

        self.done_var.set(False)
        self._update_frame(0)
        self.root.wait_variable(self.done_var)
        _modern_window_geometry[0] = self.root.geometry()
        result = self.state['result']
        self.state['playing'] = False
        return result

    def close(self):
        if self.cap is not None:
            self.cap.release()
        _modern_window_geometry[0] = self.root.geometry()
        self.root.destroy()


_modern_selector_instance = [None]


def select_interval_modern(video_path: str, video_info: Dict, auto_duration_secs: Optional[float] = None) -> Optional[Dict]:
    """使用 customtkinter 现代 GUI 选择区间（单窗口复用）。"""
    try:
        if _modern_selector_instance[0] is None:
            _modern_selector_instance[0] = ModernIntervalSelector()
        return _modern_selector_instance[0].select_interval(video_path, video_info, auto_duration_secs)
    except ImportError:
        print("[ERROR] modern GUI 依赖缺失: customtkinter/Pillow")
        print("[TIP] 请在当前运行此脚本的同一终端执行:")
        print("      py -m pip install customtkinter pillow")
        # 尝试回退 classic；若当前 OpenCV 无 GUI 支持，会抛异常并由上层处理
        return select_interval(video_path, video_info, auto_duration_secs=auto_duration_secs)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Select analysis intervals for batch videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python SelectVideoIntervals.py --directory ./Videos
  
  # 指定输出文件
  python SelectVideoIntervals.py --directory ./Videos --output ./intervals.json
  
  # 排除特定视频文件
  python SelectVideoIntervals.py --directory ./Videos --exclude refer.mp4

  # 自动模式：从开头取5分钟
  python SelectVideoIntervals.py --directory ./Videos --auto-5min
        """
    )
    
    parser.add_argument('--directory', '-d',
                       required=True,
                       help='Directory containing video files (required)')
    parser.add_argument('--output', '-o',
                       help='Output JSON file path (default: video_intervals.json in the video directory)')
    parser.add_argument('--exclude', '-e',
                       action='append',
                       help='Video file(s) to exclude (can be specified multiple times)')
    parser.add_argument('--auto-5min',
                       action='store_true',
                       help='Interactive: choose start only, end auto = start + 5 minutes')
    parser.add_argument('--auto-10min',
                       action='store_true',
                       help='Interactive: choose start only, end auto = start + 10 minutes')
    parser.add_argument('--gui',
                       choices=['modern', 'classic'],
                       default='modern',
                       help='GUI backend: modern (customtkinter) or classic (OpenCV). Default: modern')
    
    args = parser.parse_args()
    
    # 获取目录
    directory = args.directory
    
    # 清理路径
    if isinstance(directory, str):
        directory = directory.strip()
        if directory.startswith('r"') and directory.endswith('"'):
            directory = directory[2:-1]
        elif directory.startswith("r'") and directory.endswith("'"):
            directory = directory[2:-1]
        elif directory.startswith('"') and directory.endswith('"'):
            directory = directory[1:-1]
        elif directory.startswith("'") and directory.endswith("'"):
            directory = directory[1:-1]
    
    # 处理相对路径
    if not os.path.isabs(directory):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_directory = os.path.abspath(os.path.join(script_dir, directory))
        if os.path.isdir(abs_directory):
            directory = abs_directory
        else:
            directory = os.path.abspath(directory)
    else:
        directory = os.path.normpath(directory)
    
    if not os.path.isdir(directory):
        print(f"[ERROR] Directory not found: {directory}")
        sys.exit(1)
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(directory, 'video_intervals.json')
    
    # 获取要排除的视频文件
    exclude_files = args.exclude if args.exclude else []
    
    # 查找视频文件
    print(f"\nScanning directory: {directory}")
    video_files = find_video_files(directory, exclude_files=exclude_files)
    
    if not video_files:
        print(f"[ERROR] No video files found in directory")
        if exclude_files:
            print(f"[INFO] (Excluded {len(exclude_files)} video file(s))")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} video file(s):")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    if exclude_files:
        print(f"\nExcluded {len(exclude_files)} video file(s):")
        for ef in exclude_files:
            print(f"  - {ef}")
    
    # 处理每个视频
    intervals = {}
    skipped = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        if not video_info:
            print(f"[ERROR] Could not read video info: {video_path}")
            skipped.append(os.path.basename(video_path))
            continue
        
        # 选择区间
        try:
            if args.auto_5min:
                if args.gui == 'modern':
                    interval = select_interval_modern(video_path, video_info, auto_duration_secs=300)
                else:
                    interval = select_interval(video_path, video_info, auto_duration_secs=300)
            elif args.auto_10min:
                if args.gui == 'modern':
                    interval = select_interval_modern(video_path, video_info, auto_duration_secs=600)
                else:
                    interval = select_interval(video_path, video_info, auto_duration_secs=600)
            else:
                if args.gui == 'modern':
                    interval = select_interval_modern(video_path, video_info)
                else:
                    interval = select_interval(video_path, video_info)
        except Exception as e:
            print(f"[ERROR] GUI 初始化失败: {e}")
            print("[TIP] 如果你在使用 modern GUI，请在同一终端执行:")
            print("      py -m pip install customtkinter pillow")
            print("[TIP] 如果你想强制 classic 模式:")
            print("      py SelectVideoIntervals.py -d <目录> --gui classic")
            if _modern_selector_instance[0] is not None:
                _modern_selector_instance[0].close()
                _modern_selector_instance[0] = None
            sys.exit(1)
        
        if interval:
            # 使用相对路径作为key（相对于目录）
            video_key = os.path.relpath(video_path, directory)
            intervals[video_key] = interval
        else:
            skipped.append(os.path.basename(video_path))

    if _modern_selector_instance[0] is not None:
        _modern_selector_instance[0].close()
        _modern_selector_instance[0] = None
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(intervals, f, indent=2, ensure_ascii=False)
        print("\n" + "="*70)
        print("Interval Selection Complete!")
        print("="*70)
        print(f"Output file: {output_file}")
        print(f"Processed: {len(intervals)} video(s)")
        if skipped:
            print(f"Skipped: {len(skipped)} video(s)")
            for v in skipped:
                print(f"  - {v}")
        print(f"\nYou can now use this intervals file in batch processing.")
    except Exception as e:
        print(f"\n[ERROR] Failed to save intervals file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

