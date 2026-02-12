"""
自动批量 OCR：扫描 img/ 下所有 png，从文件名提取 turns，调用 ocr.py 生成 JSON。
文件名格式：20260209_2031_T22.png  →  _T22 表示 turns=22
"""

import glob
import os
import re
import subprocess
import sys


def main() -> None:
    img_dir = "img"
    png_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if not png_files:
        print(f"[WARN] 在 {img_dir}/ 下未找到任何 .png 文件")
        return

    # 匹配 _T 后面的数字
    pattern = re.compile(r"_T(\d+)$")

    success = 0
    skipped = 0

    for path in png_files:
        basename = os.path.splitext(os.path.basename(path))[0]
        m = pattern.search(basename)
        if not m:
            print(f"[跳过] {path}  —— 文件名不含 _T** 后缀，无法提取 turns")
            skipped += 1
            continue

        turns = int(m.group(1))
        print(f"[处理] {basename}.png  turns={turns}")

        cmd = [
            sys.executable, "ocr.py",
            "--name", basename,
            "--turns", str(turns),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")

        if result.returncode == 0:
            success += 1
        else:
            print(f"  [ERROR] ocr.py 返回码 {result.returncode}")

    print(f"\n全部完成：成功 {success}，跳过 {skipped}，共 {len(png_files)} 张图片")


if __name__ == "__main__":
    main()
