#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成带有位置跟踪标记的视频

使用方法:
    python GenerateTrackingVideo.py --video path/to/video.mp4 --csv path/to/LocationOutput.csv
"""

import os
import sys
import argparse
import pandas as pd
import cv2
import numpy as np

# Ensure project root is importable when running from visualization/ subfolder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import LocationTracking_Functions as lt


def generate_tracking_video(video_path, csv_path, output_path=None, 
                            start_frame=0, end_frame=None, fps=30, 
                            resize=None, marker_size=10, show_trail=True, trail_length=30):
    """
    生成带有位置跟踪标记的视频
    
    Args:
        video_path: 原始视频路径
        csv_path: LocationOutput CSV 文件路径
        output_path: 输出视频路径（默认在原视频目录下）
        start_frame: 起始帧
        end_frame: 结束帧（None=到视频结尾）
        fps: 输出视频帧率
        resize: 调整大小 (width, height) 或 None
        marker_size: 标记点大小（像素）
        show_trail: 是否显示运动轨迹
        trail_length: 轨迹长度（帧数）
    """
    
    # 检查文件是否存在
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取位置数据
    print(f"读取位置数据: {csv_path}")
    location = pd.read_csv(csv_path)
    print(f"  总帧数: {len(location)}")
    print(f"  列: {list(location.columns)}")
    
    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = base_name + '_tracked.mp4'
    
    # 打开视频
    print(f"\n读取视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  原始分辨率: {width}x{height}")
    print(f"  原始帧率: {video_fps} fps")
    print(f"  总帧数: {total_frames}")
    
    # 设置结束帧
    if end_frame is None:
        end_frame = min(total_frames, len(location))
    else:
        end_frame = min(end_frame, total_frames, len(location))
    
    # 设置输出视频参数
    if resize:
        out_width, out_height = resize
    else:
        out_width, out_height = width, height
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    if not writer.isOpened():
        raise ValueError(f"无法创建输出视频: {output_path}")
    
    print(f"\n生成带标记视频...")
    print(f"  输出路径: {output_path}")
    print(f"  输出分辨率: {out_width}x{out_height}")
    print(f"  输出帧率: {fps} fps")
    print(f"  处理帧范围: {start_frame} - {end_frame}")
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 存储轨迹点
    trail_points = []
    
    # 逐帧处理
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] 无法读取帧 {frame_idx}，停止处理")
            break
        
        # 调整大小
        if resize:
            frame = cv2.resize(frame, (out_width, out_height))
        
        # 获取当前帧的位置
        if frame_idx < len(location):
            x = location.loc[frame_idx, 'X']
            y = location.loc[frame_idx, 'Y']
            
            # 调整坐标（如果调整了大小）
            if resize:
                x = int(x * out_width / width)
                y = int(y * out_height / height)
            else:
                x = int(x)
                y = int(y)
            
            # 添加到轨迹
            if not (np.isnan(x) or np.isnan(y)):
                trail_points.append((x, y))
                if len(trail_points) > trail_length:
                    trail_points.pop(0)
                
                # 绘制轨迹
                if show_trail and len(trail_points) > 1:
                    for i in range(len(trail_points) - 1):
                        alpha = (i + 1) / len(trail_points)  # 渐变透明度
                        thickness = max(1, int(alpha * 3))
                        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                        cv2.line(frame, trail_points[i], trail_points[i + 1], color, thickness)
                
                # 绘制当前位置标记
                cv2.circle(frame, (x, y), marker_size, (0, 0, 255), -1)  # 红色实心圆
                cv2.circle(frame, (x, y), marker_size + 2, (255, 255, 255), 2)  # 白色边框
                
                # 添加帧号和坐标信息
                info_text = f"Frame: {frame_idx} | Pos: ({x}, {y})"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 写入帧
        writer.write(frame)
        
        # 显示进度
        if (frame_idx - start_frame) % 100 == 0:
            progress = (frame_idx - start_frame + 1) / (end_frame - start_frame) * 100
            print(f"  进度: {progress:.1f}% ({frame_idx - start_frame + 1}/{end_frame - start_frame} 帧)")
    
    # 释放资源
    cap.release()
    writer.release()
    
    print(f"\n完成！视频已保存到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='生成带有位置跟踪标记的视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python GenerateTrackingVideo.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv
  
  # 指定输出路径
  python GenerateTrackingVideo.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --output tracked_3pl3.mp4
  
  # 只处理前1000帧，30fps
  python GenerateTrackingVideo.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --end 1000 --fps 30
  
  # 调整输出分辨率为 640x480
  python GenerateTrackingVideo.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --resize 640 480
        """
    )
    
    parser.add_argument('--video', '-v', required=True, help='原始视频路径')
    parser.add_argument('--csv', '-c', required=True, help='LocationOutput CSV 文件路径')
    parser.add_argument('--output', '-o', help='输出视频路径（默认：原视频名_tracked.mp4）')
    parser.add_argument('--start', type=int, default=0, help='起始帧（默认：0）')
    parser.add_argument('--end', type=int, default=None, help='结束帧（默认：到视频结尾）')
    parser.add_argument('--fps', type=int, default=30, help='输出帧率（默认：30）')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), 
                       help='调整输出分辨率（例如：--resize 640 480）')
    parser.add_argument('--marker-size', type=int, default=10, help='标记点大小（默认：10）')
    parser.add_argument('--no-trail', action='store_true', help='不显示运动轨迹')
    parser.add_argument('--trail-length', type=int, default=30, help='轨迹长度帧数（默认：30）')
    
    args = parser.parse_args()
    
    try:
        generate_tracking_video(
            video_path=args.video,
            csv_path=args.csv,
            output_path=args.output,
            start_frame=args.start,
            end_frame=args.end,
            fps=args.fps,
            resize=tuple(args.resize) if args.resize else None,
            marker_size=args.marker_size,
            show_trail=not args.no_trail,
            trail_length=args.trail_length
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

