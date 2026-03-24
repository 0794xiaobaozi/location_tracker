


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成带有位置跟踪标记的视频（独立版本，不依赖 LocationTracking_Functions）

使用方法:
    python GenerateTrackingVideo_Standalone.py --video path/to/video.mp4 --csv path/to/LocationOutput.csv
"""

import os
import sys
import argparse
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm


def generate_tracking_video(video_path, csv_path, output_path=None, 
                            start_frame=0, end_frame=None, fps=30, 
                            resize=None, marker_size=10, show_trail=True, trail_length=30,
                            show_info=True, crop_offset=(0, 0), apply_crop=None):
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
        show_info: 是否显示信息文字
        crop_offset: 坐标偏移量 (x_offset, y_offset)，如果CSV坐标是基于裁剪区域的
        apply_crop: 裁剪区域 (x1, y1, x2, y2)，如果需要裁剪视频帧
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
    
    # 计算实际的帧尺寸（考虑裁剪）
    if apply_crop:
        x1, y1, x2, y2 = apply_crop
        # 限制裁剪区域在视频范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        actual_width = x2 - x1
        actual_height = y2 - y1
        print(f"  裁剪区域: ({x1}, {y1}) 到 ({x2}, {y2})")
        print(f"  裁剪后尺寸: {actual_width}x{actual_height}")
        # 更新 apply_crop 为限制后的值
        apply_crop = (x1, y1, x2, y2)
    else:
        actual_width = width
        actual_height = height
    
    # 设置输出视频参数
    if resize:
        out_width, out_height = resize
    else:
        out_width, out_height = actual_width, actual_height
    
    # 创建视频写入器 - 尝试多种编码器
    # 优先使用 H.264，如果不行则使用 XVID 或 mp4v
    for codec in ['avc1', 'H264', 'X264', 'XVID', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        if writer.isOpened():
            print(f"  使用编码器: {codec}")
            break
    
    if not writer.isOpened():
        raise ValueError(f"无法创建输出视频: {output_path}，尝试了多种编码器均失败")
    
    print(f"\n生成带标记视频...")
    print(f"  输出路径: {output_path}")
    print(f"  输出分辨率: {out_width}x{out_height}")
    print(f"  输出帧率: {fps} fps")
    print(f"  处理帧范围: {start_frame} - {end_frame}")
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 存储轨迹点
    trail_points = []
    
    # 逐帧处理（使用 tqdm 显示进度条）
    for frame_idx in tqdm(range(start_frame, end_frame), desc="处理视频"):
        ret, frame = cap.read()
        if not ret:
            print(f"\n[WARNING] 无法读取帧 {frame_idx}，停止处理")
            break
        
        # 应用裁剪（如果指定）
        if apply_crop:
            x1, y1, x2, y2 = apply_crop
            frame = frame[y1:y2, x1:x2]
        
        # 调整大小
        if resize:
            frame = cv2.resize(frame, (out_width, out_height))
        
        # 获取当前帧的位置
        if frame_idx < len(location):
            x = location.loc[frame_idx, 'X']
            y = location.loc[frame_idx, 'Y']
            
            # 添加坐标偏移量（只在没有应用裁剪时使用）
            if not apply_crop:
                x = x + crop_offset[0]
                y = y + crop_offset[1]
            
            # 调整坐标（如果调整了大小）
            if resize:
                # 基于实际帧尺寸（裁剪后的尺寸）进行缩放
                x = int(x * out_width / actual_width)
                y = int(y * out_height / actual_height)
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
                if show_info:
                    # 半透明背景
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (5, 5), (400, 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    # 信息文字
                    info_text1 = f"Frame: {frame_idx}"
                    info_text2 = f"Position: ({x}, {y})"
                    cv2.putText(frame, info_text1, (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, info_text2, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 验证帧尺寸
        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            print(f"\n[ERROR] 帧尺寸不匹配！期望 {out_width}x{out_height}，实际 {frame.shape[1]}x{frame.shape[0]}")
            print(f"  帧 {frame_idx} 的形状: {frame.shape}")
            raise ValueError("帧尺寸不匹配")
        
        # 写入帧
        success = writer.write(frame)
        if not success and frame_idx == start_frame:
            print(f"\n[WARNING] 写入第一帧失败，帧信息:")
            print(f"  形状: {frame.shape}, dtype: {frame.dtype}, 尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 释放资源
    cap.release()
    writer.release()
    
    print(f"\n[OK] 完成！视频已保存到: {output_path}")
    print(f"  处理了 {end_frame - start_frame} 帧")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='生成带有位置跟踪标记的视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python GenerateTrackingVideo_Standalone.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv
  
  # 如果分析时进行了crop（例如从notebook的Cell 6），需要应用相同的crop
  python GenerateTrackingVideo_Standalone.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --apply-crop 124 1 954 604
  
  # 或者使用坐标偏移（在完整视频上绘制）
  python GenerateTrackingVideo_Standalone.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --crop-offset 124 1
  
  # 只处理前1000帧，30fps
  python GenerateTrackingVideo_Standalone.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --end 1000 --fps 30
  
  # 调整输出分辨率为 640x480
  python GenerateTrackingVideo_Standalone.py --video 3pl3.mp4 --csv 3pl3_LocationOutput.csv --resize 640 480
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
    parser.add_argument('--no-info', action='store_true', help='不显示帧号和坐标信息')
    parser.add_argument('--crop-offset', nargs=2, type=int, metavar=('X_OFFSET', 'Y_OFFSET'), 
                       default=[0, 0], help='坐标偏移量，如果CSV坐标是基于裁剪区域的（例如：--crop-offset 124 1）')
    parser.add_argument('--apply-crop', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                       help='裁剪视频帧（例如：--apply-crop 124 1 954 604）')
    
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
            trail_length=args.trail_length,
            show_info=not args.no_info,
            crop_offset=tuple(args.crop_offset),
            apply_crop=tuple(args.apply_crop) if args.apply_crop else None
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

