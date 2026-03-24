#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量视频截取脚本：根据 video_intervals.json 中的设置截取视频

这个脚本会读取指定目录中的 video_intervals.json 文件，根据其中设置的
开始和结束帧截取视频，并保存到 cropped_video 文件夹下，保持原有的目录结构。

支持批量处理多个目录，可以：
1. 多次指定 --directory 参数处理多个目录
2. 使用 --recursive 递归查找所有包含 video_intervals.json 的目录

使用方法:
    # 处理单个目录
    python CropVideosFromIntervals.py --directory <video_directory>
    
    # 批量处理多个目录
    python CropVideosFromIntervals.py --directory <dir1> --directory <dir2> ...
    
    # 递归查找并处理所有目录
    python CropVideosFromIntervals.py --recursive <base_directory>

参数:
    --directory, -d: 包含视频文件和 video_intervals.json 的目录（可多次指定）
    --recursive, -r: 递归查找所有包含 video_intervals.json 的目录
    --intervals-file, -i: video_intervals.json 文件路径（可选，仅对第一个目录有效）
    --output-dir, -o: 输出目录名称（可选，默认为 cropped_video）
"""

import os
import sys
import argparse
import json
import cv2
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm


def export_video_interval(video_path: str, start_frame: int, end_frame: int, output_path: str) -> bool:
    """
    从视频中导出指定帧范围的片段
    
    Args:
        video_path: 输入视频路径
        start_frame: 开始帧（0-based）
        end_frame: 结束帧（包含）
        output_path: 输出视频路径
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"  [错误] 视频文件不存在: {video_path}")
        return False
    
    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [错误] 无法打开视频文件: {video_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查帧范围是否有效
    if start_frame < 0:
        start_frame = 0
        print(f"  [警告] start_frame 小于 0，已调整为 0")
    if end_frame >= total_frames:
        end_frame = total_frames - 1
        print(f"  [警告] end_frame 超出范围，已调整为 {end_frame}")
    if start_frame > end_frame:
        print(f"  [错误] 无效的帧范围: start_frame ({start_frame}) > end_frame ({end_frame})")
        cap.release()
        return False
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置视频编码器（使用 H.264 编码以获得更好的兼容性）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        print(f"  [错误] 无法创建输出视频文件: {output_path}")
        return False
    
    # 跳转到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 读取并写入帧
    frame_count = 0
    current_frame = start_frame
    total_frames_to_export = end_frame - start_frame + 1
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            # 如果无法读取，尝试检查是否已经到达视频末尾
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if actual_pos >= total_frames:
                break
            else:
                current_frame += 1
                continue
        
        out.write(frame)
        frame_count += 1
        current_frame += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    if frame_count > 0:
        return True
    else:
        print(f"  [错误] 未能导出任何帧")
        return False


def find_video_intervals_file(directory: str, intervals_file: Optional[str] = None) -> Optional[str]:
    """
    查找 video_intervals.json 文件
    
    Args:
        directory: 目录路径
        intervals_file: 指定的 JSON 文件路径（可选）
    
    Returns:
        JSON 文件路径，如果未找到则返回 None
    """
    if intervals_file:
        if os.path.isfile(intervals_file):
            return intervals_file
        else:
            print(f"[警告] 指定的 intervals 文件不存在: {intervals_file}")
            return None
    
    # 在目录中查找 video_intervals.json
    default_path = os.path.join(directory, 'video_intervals.json')
    if os.path.isfile(default_path):
        return default_path
    
    return None


def load_intervals(intervals_file: str) -> Dict:
    """
    加载 video_intervals.json 文件
    
    Args:
        intervals_file: JSON 文件路径
    
    Returns:
        包含区间信息的字典
    """
    try:
        with open(intervals_file, 'r', encoding='utf-8') as f:
            intervals = json.load(f)
        return intervals
    except Exception as e:
        print(f"[错误] 无法读取 intervals 文件: {e}")
        return {}


def find_video_files(directory: str, extensions: list = None) -> list:
    """
    在目录中查找视频文件
    
    Args:
        directory: 目录路径
        extensions: 视频文件扩展名列表
    
    Returns:
        视频文件路径列表
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
    
    video_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        import glob
        found = glob.glob(pattern, recursive=False)
        video_files.extend(found)
        
        # 也查找大写扩展名
        if ext != ext.upper():
            pattern = os.path.join(directory, f'*{ext.upper()}')
            found = glob.glob(pattern, recursive=False)
            video_files.extend(found)
    
    return sorted(list(set(video_files)))


def find_directories_with_intervals(base_directory: str) -> list:
    """
    递归查找所有包含 video_intervals.json 的目录
    
    Args:
        base_directory: 基础目录路径
    
    Returns:
        包含 video_intervals.json 的目录列表
    """
    directories = []
    base_path = Path(base_directory)
    
    if not base_path.exists() or not base_path.is_dir():
        return directories
    
    # 检查当前目录
    intervals_file = base_path / 'video_intervals.json'
    if intervals_file.exists():
        directories.append(str(base_path))
    
    # 递归查找子目录
    for subdir in base_path.rglob('*'):
        if subdir.is_dir():
            intervals_file = subdir / 'video_intervals.json'
            if intervals_file.exists():
                directories.append(str(subdir))
    
    return sorted(directories)


def process_directory(directory: str, intervals_file: Optional[str] = None, 
                     output_dir_name: str = 'cropped_video') -> dict:
    """
    处理目录中的所有视频，根据 intervals 文件截取
    
    Args:
        directory: 视频目录路径
        intervals_file: intervals JSON 文件路径（可选）
        output_dir_name: 输出目录名称
    """
    # 规范化路径
    directory = os.path.normpath(os.path.abspath(directory))
    
    if not os.path.isdir(directory):
        print(f"[错误] 目录不存在: {directory}")
        sys.exit(1)
    
    # 查找 intervals 文件
    intervals_path = find_video_intervals_file(directory, intervals_file)
    if not intervals_path:
        print(f"[错误] 未找到 video_intervals.json 文件")
        print(f"  请在目录中创建 video_intervals.json 或使用 --intervals-file 指定")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"批量视频截取工具")
    print(f"{'='*70}")
    print(f"视频目录: {directory}")
    print(f"Intervals 文件: {intervals_path}")
    print(f"{'='*70}\n")
    
    # 加载 intervals
    intervals = load_intervals(intervals_path)
    if not intervals:
        print("[错误] intervals 文件为空或格式错误")
        sys.exit(1)
    
    print(f"已加载 {len(intervals)} 个视频的区间设置\n")
    
    # 查找视频文件
    video_files = find_video_files(directory)
    if not video_files:
        print(f"[错误] 在目录中未找到视频文件")
        sys.exit(1)
    
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    # 创建输出目录（保持完整的目录结构）
    # 例如：video/D2/right -> video/cropped_video/D2/right
    # 找到 video 目录的父目录（假设 video 是包含所有视频的根目录）
    # 如果目录路径中包含 'video'，则在 'video' 后插入 'cropped_video'
    # 否则，在目录的父目录下创建 cropped_video/目录名
    
    # 尝试找到 'video' 在路径中的位置
    path_parts = Path(directory).parts
    video_index = None
    for i, part in enumerate(path_parts):
        if 'video' in part.lower():
            video_index = i
            break
    
    if video_index is not None:
        # 在 video 目录后插入 cropped_video
        new_parts = list(path_parts[:video_index + 1]) + [output_dir_name] + list(path_parts[video_index + 1:])
        output_base = os.path.join(*new_parts)
    else:
        # 如果找不到 video 目录，则在父目录下创建 cropped_video/目录名
        base_dir = os.path.dirname(directory.rstrip(os.sep))
        dir_name = os.path.basename(directory.rstrip(os.sep))
        output_base = os.path.join(base_dir, output_dir_name, dir_name)
    
    os.makedirs(output_base, exist_ok=True)
    
    # 处理每个视频
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc="处理视频", unit="个"):
        video_name = os.path.basename(video_path)
        
        # 检查是否有对应的区间设置
        if video_name not in intervals:
            print(f"\n[跳过] {video_name}: 在 intervals 文件中未找到")
            skipped_count += 1
            continue
        
        interval = intervals[video_name]
        start_frame = interval.get('start_frame')
        end_frame = interval.get('end_frame')
        
        if start_frame is None or end_frame is None:
            print(f"\n[跳过] {video_name}: intervals 数据不完整")
            skipped_count += 1
            continue
        
        # 生成输出路径（保持相同的文件名）
        output_path = os.path.join(output_base, video_name)
        
        print(f"\n处理: {video_name}")
        print(f"  帧范围: {start_frame} - {end_frame} (共 {end_frame - start_frame + 1} 帧)")
        print(f"  输出: {output_path}")
        
        try:
            success = export_video_interval(video_path, start_frame, end_frame, output_path)
            if success:
                print(f"  [成功] 已保存到: {output_path}")
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")
            error_count += 1
    
    # 输出统计信息
    print(f"\n{'='*70}")
    print("目录处理完成！")
    print(f"{'='*70}")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skipped_count} 个")
    print(f"错误: {error_count} 个")
    print(f"\n输出目录: {output_base}")
    print(f"{'='*70}\n")
    
    # 返回统计信息
    return {
        'success': success_count,
        'skipped': skipped_count,
        'error': error_count,
        'output_dir': output_base
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Batch crop videos based on video_intervals.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法（处理单个目录）
  python CropVideosFromIntervals.py --directory ./video/D2/right
  
  # 批量处理多个目录
  python CropVideosFromIntervals.py --directory ./video/D2/right --directory ./video/D2/left
  
  # 递归查找并处理所有包含 video_intervals.json 的目录
  python CropVideosFromIntervals.py --recursive ./video
  
  # 指定 intervals 文件（仅对第一个目录有效）
  python CropVideosFromIntervals.py --directory ./video/D2/right --intervals-file ./custom_intervals.json
  
  # 指定输出目录名称
  python CropVideosFromIntervals.py --directory ./video/D2/right --output-dir my_cropped_videos
        """
    )
    
    parser.add_argument('--directory', '-d',
                       action='append',
                       dest='directories',
                       help='Directory containing video files and video_intervals.json (can be specified multiple times)')
    parser.add_argument('--recursive', '-r',
                       help='Recursively find and process all directories containing video_intervals.json')
    parser.add_argument('--intervals-file', '-i',
                       help='Path to video_intervals.json file (default: video_intervals.json in each directory)')
    parser.add_argument('--output-dir', '-o',
                       default='cropped_video',
                       help='Output directory name (default: cropped_video)')
    
    args = parser.parse_args()
    
    # 确定要处理的目录列表
    directories_to_process = []
    
    if args.recursive:
        # 递归查找所有包含 video_intervals.json 的目录
        print(f"\n递归查找包含 video_intervals.json 的目录...")
        print(f"基础目录: {args.recursive}\n")
        found_dirs = find_directories_with_intervals(args.recursive)
        if found_dirs:
            print(f"找到 {len(found_dirs)} 个目录:")
            for d in found_dirs:
                print(f"  - {d}")
            directories_to_process = found_dirs
        else:
            print(f"[警告] 在 {args.recursive} 中未找到包含 video_intervals.json 的目录")
            sys.exit(1)
    elif args.directories:
        # 使用指定的目录列表
        directories_to_process = args.directories
    else:
        print("[错误] 必须指定 --directory 或 --recursive")
        parser.print_help()
        sys.exit(1)
    
    # 批量处理所有目录
    print(f"\n{'='*70}")
    print(f"开始批量处理 {len(directories_to_process)} 个目录")
    print(f"{'='*70}\n")
    
    total_stats = {
        'success': 0,
        'skipped': 0,
        'error': 0,
        'directories_processed': 0,
        'directories_failed': 0
    }
    
    for i, directory in enumerate(directories_to_process, 1):
        print(f"\n{'#'*70}")
        print(f"处理目录 {i}/{len(directories_to_process)}: {directory}")
        print(f"{'#'*70}")
        
        try:
            # 对于每个目录，使用其自己的 intervals 文件（除非明确指定）
            intervals_file = args.intervals_file if i == 1 else None
            
            stats = process_directory(
                directory,
                intervals_file,
                args.output_dir
            )
            
            total_stats['success'] += stats['success']
            total_stats['skipped'] += stats['skipped']
            total_stats['error'] += stats['error']
            total_stats['directories_processed'] += 1
            
        except Exception as e:
            print(f"\n[错误] 处理目录失败: {directory}")
            print(f"  错误信息: {e}")
            total_stats['directories_failed'] += 1
            continue
    
    # 输出总体统计信息
    print(f"\n{'='*70}")
    print("批量处理完成！")
    print(f"{'='*70}")
    print(f"处理的目录数: {total_stats['directories_processed']}")
    print(f"失败的目录数: {total_stats['directories_failed']}")
    print(f"\n总计:")
    print(f"  成功: {total_stats['success']} 个视频")
    print(f"  跳过: {total_stats['skipped']} 个视频")
    print(f"  错误: {total_stats['error']} 个视频")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

