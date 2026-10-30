# -*- coding: utf-8 -*-
"""生成应用图标：深青底 + 白色频谱柱（频率造型）。

用法：
    python tools/make_icon.py

输出：assets/icon.png（512×512，buildozer 会据此生成各密度图标）
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "icon.png"

SIZE = 512
SCALE = 4                      # 先画 4 倍再缩小，得到抗锯齿边缘
W = SIZE * SCALE

TOP = (11, 110, 110)           # #0B6E6E
BOTTOM = (14, 134, 134)        # #0E8686
BAR = (255, 255, 255)
BASE_LINE = (255, 255, 255, 110)


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def main() -> int:
    # 1. 垂直渐变底
    gradient = Image.new("RGB", (1, W))
    pixels = gradient.load()
    for y in range(W):
        ratio = y / (W - 1)
        pixels[0, y] = tuple(
            int(TOP[i] + (BOTTOM[i] - TOP[i]) * ratio) for i in range(3)
        )
    canvas = gradient.resize((W, W)).convert("RGBA")

    # 2. 频谱柱
    draw = ImageDraw.Draw(canvas, "RGBA")
    heights = [0.34, 0.56, 0.42, 0.80, 1.00, 0.62, 0.48, 0.72, 0.28]
    bar_w = 24 * SCALE
    gap = 12 * SCALE
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    start_x = (W - total) // 2
    baseline = 356 * SCALE
    max_height = 186 * SCALE
    radius = bar_w // 2
    for index, ratio in enumerate(heights):
        x0 = start_x + index * (bar_w + gap)
        x1 = x0 + bar_w
        top = baseline - int(max_height * ratio)
        draw.rounded_rectangle((x0, top, x1, baseline), radius=radius, fill=BAR + (255,))

    # 3. 底部基准线
    draw.rounded_rectangle(
        (start_x - 6 * SCALE, baseline + 10 * SCALE,
         start_x + total + 6 * SCALE, baseline + 14 * SCALE),
        radius=2 * SCALE,
        fill=BASE_LINE,
    )

    # 4. 圆角裁切
    canvas.putalpha(rounded_mask(W, 112 * SCALE))
    canvas = canvas.resize((SIZE, SIZE), Image.LANCZOS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("图标已生成：", OUT, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
