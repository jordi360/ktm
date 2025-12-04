#!/usr/bin/env python3
"""
Rename unwrapped frames with zero-padded numbers for correct ordering.
Usage: python tools/rename_frames.py <directory>
"""

import os
import sys
import re
from pathlib import Path


def rename_frames(directory):
    """Rename files from unwrap_N.png to unwrap_NN.png with zero-padding."""
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all unwrap_*.png files
    pattern = re.compile(r'unwrap_(\d+)\.png')
    files = []

    for file in directory.glob('unwrap_*.png'):
        match = pattern.match(file.name)
        if match:
            num = int(match.group(1))
            files.append((num, file))

    if not files:
        print(f"No unwrap_*.png files found in {directory}")
        return

    # Sort by number
    files.sort(key=lambda x: x[0])

    # Determine padding width
    max_num = max(num for num, _ in files)
    padding = len(str(max_num))

    print(f"Found {len(files)} files, renaming with {padding}-digit padding...")

    # Rename files (use temporary names first to avoid conflicts)
    temp_files = []
    for num, file in files:
        temp_name = file.parent / f"temp_{num:0{padding}d}.png"
        file.rename(temp_name)
        temp_files.append((num, temp_name))
        print(f"  {file.name} -> temp_{num:0{padding}d}.png")

    # Rename from temp to final names
    for num, temp_file in temp_files:
        final_name = temp_file.parent / f"unwrap_{num:0{padding}d}.png"
        temp_file.rename(final_name)

    print(f"\nDone! Files renamed from unwrap_1.png to unwrap_{max_num:0{padding}d}.png")
    print(f"Now run ffmpeg with pattern: unwrap_%0{padding}d.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/rename_frames.py <directory>")
        print("Example: python tools/rename_frames.py captures/E198/metal/unwrapped")
        sys.exit(1)

    rename_frames(sys.argv[1])
