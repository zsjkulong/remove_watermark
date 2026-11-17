#!/usr/bin/env python3
"""
视频水印去除工具

通过提取多个关键帧并分析其梯度差异来自动检测和去除静态水印。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def normalize(x: np.ndarray) -> np.ndarray:
    """归一化数组到[0, 1]范围"""
    _min = np.min(x)
    _max = np.max(x)
    if _max - _min == 0:
        return np.zeros_like(x)
    return (x - _min) / (_max - _min)


def extract_keyframes(video_path: str, max_frames: int = 50) -> list:
    """提取视频的关键帧"""
    # 获取关键帧时间戳
    cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "warning",
        "-select_streams", "v", "-skip_frame", "nokey",
        "-show_frames", "-show_entries", "frame=pkt_dts_time",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        keyframes_time = [
            line.split("=")[1].strip()
            for line in result.stdout.split("\n")
            if line.startswith("pkt_dts_time=") and "N/A" not in line
        ]
        keyframes_time = sorted(set(keyframes_time))
        # 如果没有找到关键帧，降级到时间间隔提取
        if not keyframes_time:
            raise ValueError("No keyframes found")
        # 随机打乱并取max_frames个
        np.random.seed(42)  # 可复现的结果
        np.random.shuffle(keyframes_time)
        keyframes_time = keyframes_time[:max_frames]
    except (subprocess.CalledProcessError, ValueError):
        # 如果ffprobe失败或没有关键帧，使用时间间隔提取
        cmd = [
            "ffprobe", "-hide_banner", "-loglevel", "warning",
            "-show_entries", "format=duration", "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip().split(",")[0])
        interval = duration / max_frames
        keyframes_time = [f"{i * interval:.4f}" for i in range(1, max_frames + 1)]

    return keyframes_time


def extract_frames(video_path: str, keyframes_time: list, tmpdir: Path) -> int:
    """从视频中提取帧图片"""
    counter = 0
    for ts in keyframes_time:
        try:
            ts_str = str(float(ts))
        except ValueError:
            continue

        output_path = tmpdir / f"output_{counter}.png"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", ts_str, "-i", video_path, "-vframes", "1",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        counter += 1

    return counter


def extract_watermark_mask(tmpdir: Path) -> Path:
    """通过分析多帧图像提取水印蒙版

    只保留"在多帧中位置稳定、持续存在的高对比边缘"作为水印：
     - 对每帧先取绝对梯度再平均：水印边缘位置固定，平均后依然高；
      运动物体边缘位置飘移，平均后会被削弱。
     - 再用连通域过滤：去掉零散噪点，同时丢弃覆盖画面比例过大的
      "整片静止背景边缘"，避免蒙版过大导致整帧被模糊。
    """
    buff = []
    for p in sorted(tmpdir.glob("output_*.png")):
        buff.append(imageio.imread(p))

    if len(buff) < 2:
        raise ValueError(f"需要至少2帧图像来提取水印，实际获取了 {len(buff)} 帧")

    images = np.array(buff, dtype=np.float32)

    # 各像素在各帧的绝对梯度平均值：稳定边缘 → 高值，运动边缘 → 低值
    dx = np.abs(np.gradient(images, axis=1))
    dy = np.abs(np.gradient(images, axis=2))
    edge = (dx + dy).mean(axis=(0, 3))  # (H, W)

    # 归一化 + 轻微平滑，避免噪声导致的小孔
    edge = normalize(gaussian_filter(edge, sigma=1.2))
    binary = edge > 0.3  # 调高该值 → 蒙版更小

    # 连通域分析：只保留面积适中的分量
    label, n = ndimage.label(np.asarray(binary, dtype=np.uint8))
    if n > 0:
        sizes = ndimage.sum(binary, label, index=np.arange(1, n + 1))
        frame_area = binary.size
        min_area = frame_area * 0.0005   # 去掉零散噪点
        max_area = frame_area * 0.30     # 去掉横跨画面的背景边缘
        keep = (sizes >= min_area) & (sizes <= max_area)
        keep_map = np.zeros(n + 1, dtype=bool)
        keep_map[1:] = keep
        refined = keep_map[label]
    else:
        refined = binary

    mask = np.zeros_like(binary, dtype=np.uint8)
    mask[refined] = 255

    # 轻微膨胀，把半透明水印的羽化边缘也纳入修补
    if mask.any():
        mask = ndimage.binary_dilation(mask, iterations=2).astype(np.uint8) * 255

    mask_path = tmpdir / "mask.png"
    imageio.imsave(str(mask_path), mask)
    return mask_path


def remove_watermark(video_path: str, mask_path: Path, output_path: str):
    """使用ffmpeg的removelogo滤镜去除水印

    removelogo 滤镜参数是绝对路径时会因 Windows 盘符冒号被 FFmpeg 多层
    解析切分而报错。因此这里把 ffmpeg 的工作目录切到蒙版所在目录，
    仅传入文件名，彻底绕开盘符冒号问题。
    """
    mask_dir = mask_path.parent
    # 切换工作目录到蒙版目录会改变相对路径解析基准，
    # 因此须先把输入/输出路径转为绝对路径。
    video_abs = os.path.abspath(video_path)
    output_abs = os.path.abspath(output_path)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y", "-stats", "-i", video_abs,
        "-acodec", "copy",
        "-vf", f"removelogo={mask_path.name}",
        # libx264 视觉上更锐利、兼容性更好；crf 18 属视觉无损级（越小越清晰）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        output_abs
    ]
    subprocess.run(cmd, cwd=str(mask_dir), check=True)


def main():
    parser = argparse.ArgumentParser(
        description="视频水印去除工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python remove_watermark.py input.mp4
  python remove_watermark.py input.mp4 output_cleaned.mp4
  python remove_watermark.py input.mp4 -k 100
        """
    )
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", nargs="?", default=None,
                        help="输出视频文件路径（默认: 输入文件_cleaned.mp4）")
    parser.add_argument("-k", "--keyframes", type=int, default=50,
                        help="提取的关键帧数量（默认: 50）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="显示详细输出")

    args = parser.parse_args()

    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    # 生成输出文件名
    if args.output:
        output_path = args.output
    else:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = str(input_path.parent / f"{stem}_cleaned{suffix}")

    # 创建临时目录
    tmpdir = Path(tempfile.mkdtemp(prefix="watermark_remove_"))

    try:
        if args.verbose:
            print(f"处理视频: {input_path}")
            print(f"输出文件: {output_path}")
            print(f"关键帧数量: {args.keyframes}")

        # 提取关键帧
        if args.verbose:
            print("提取关键帧...")
        keyframes_time = extract_keyframes(str(input_path), args.keyframes)
        if args.verbose:
            print(f"找到 {len(keyframes_time)} 个关键帧")

        # 提取帧图片
        if args.verbose:
            print("提取帧图片...")
        frame_count = extract_frames(str(input_path), keyframes_time, tmpdir)
        if args.verbose:
            print(f"提取了 {frame_count} 帧")

        if frame_count < 2:
            print(f"错误: 提取了 {frame_count} 帧，需要至少2帧")
            sys.exit(1)

        # 提取水印蒙版
        if args.verbose:
            print("提取水印蒙版...")
        mask_path = extract_watermark_mask(tmpdir)
        if args.verbose:
            print(f"蒙版保存至: {mask_path}")

        # 去除水印
        if args.verbose:
            print("去除水印中...")
        remove_watermark(str(input_path), mask_path, output_path)
        if args.verbose:
            print(f"完成！输出文件: {output_path}")
        else:
            print(f"完成: {output_path}")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    finally:
        # 清理临时目录
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()