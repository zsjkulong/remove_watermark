#!/usr/bin/env python3
"""
按写死区域去除水印

只对左上角固定区域做 delogo 插值修补，其余画面完全不动。

用法:
  python remove_region.py 输入.mp4 [输出.mp4]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 写死的待去除区域坐标 (left, top, width, height)，单位为像素。
# 这是针对当前视频手工校正好的值，只修改左上角那一小块。
HARDCODED_REGION = (16, 17, 120, 55)


def main():
    parser = argparse.ArgumentParser(description="按写死区域去除水印")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", nargs="?", default=None,
                        help="输出视频文件路径（默认: 输入文件_cleaned.mp4）")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    output_path = args.output or str(
        input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    )

    x, y, w, h = HARDCODED_REGION
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y", "-stats", "-i", str(input_path),
        "-acodec", "copy",
        "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        output_path,
    ]
    print(f"去除区域(left,top,width,height): {HARDCODED_REGION}")
    subprocess.run(cmd, check=True)
    print(f"完成: {output_path}")


if __name__ == "__main__":
    main()