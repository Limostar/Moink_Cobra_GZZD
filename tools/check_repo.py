# -*- coding: utf-8 -*-
"""上传 GitHub 前自检（防呆）。

只依赖标准库，本地和 GitHub Actions 都能跑：
    python tools/check_repo.py

检查项：
    1. 打包必需的文件是否齐全（缺一个 APK 就会崩或黑屏）；
    2. buildozer.spec 的关键配置（图标、npz 后缀、依赖里不能出现 scipy/sklearn）；
    3. 图标必须是 512×512 PNG；
    4. 应用名称在 main.py 与 buildozer.spec 里是否一致；
    5. 是否混入了不该提交的构建产物（bin/、.buildozer/、__pycache__ 等）；
    6. 打印出应当提交的文件清单，供对照。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "main.py",
    "buildozer.spec",
    "requirements.txt",
    "constraints.txt",
    "bdapp/__init__.py",
    "bdapp/dsp.py",
    "bdapp/model.py",
    "bdapp/assets.py",
    "assets/models/params.json",
    "assets/models/rf_model.npz",
    "assets/models/filters.npz",
    "assets/data/demo.npz",
    "assets/fonts/NotoSansSC-Regular.ttf",
    "assets/icon.png",
    ".github/workflows/build-apk.yml",
    # numpy 编译补丁：缺了会在云端卡在 “no template named 'unordered_map'”
    "p4a-recipes/numpy/__init__.py",
    "p4a-recipes/numpy/ensure-unordered-map.patch",
]

FORBIDDEN_DIRS = ["bin", ".buildozer", "build", "dist", "__pycache__", "bdapp/__pycache__"]
FORBIDDEN_GLOBS = ["*.apk", "*.pyc", "*.spec.bak"]
SKIP_DIRS = FORBIDDEN_DIRS + [".git", ".kivy", ".idea", ".vscode"]

MAX_TOTAL_MB = 60.0


def read_png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(24)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    # 1. 必需文件
    for name in REQUIRED:
        path = ROOT / name
        if not path.exists():
            problems.append("缺少文件：%s" % name)
        elif path.stat().st_size == 0:
            problems.append("空文件：%s" % name)

    # 2. buildozer.spec
    spec_path = ROOT / "buildozer.spec"
    if spec_path.exists():
        spec = spec_path.read_text(encoding="utf-8", errors="ignore")
        checks = {
            "icon.filename": "icon.filename" in spec,
            "包名": "package.name" in spec,
            "包含 npz 后缀": "npz" in spec.split("source.include_exts", 1)[-1].split("\n", 1)[0],
            "依赖含 kivy": "kivy" in spec,
            "依赖含 numpy": "numpy" in spec,
        }
        # 只检查 requirements 这一行本身，避免被注释里的“不引入 scipy / sklearn”误伤
        requirements = ""
        for line in spec.splitlines():
            if line.strip().startswith("requirements"):
                requirements = line.split("=", 1)[1].strip()
                break
        checks["依赖不含 scipy"] = "scipy" not in requirements
        checks["依赖不含 scikit-learn"] = "sklearn" not in requirements and "scikit" not in requirements
        checks["已挂载本地 recipe"] = "p4a.local_recipes" in spec and "p4a-recipes" in spec
        for label, ok in checks.items():
            if not ok:
                problems.append("buildozer.spec 检查未通过：%s" % label)
        title = ""
        for line in spec.splitlines():
            if line.strip().startswith("title ="):
                title = line.split("=", 1)[1].strip()
                break
    else:
        title = ""

    # 3. 图标
    icon = ROOT / "assets" / "icon.png"
    if icon.exists():
        size = read_png_size(icon)
        if size is None:
            problems.append("assets/icon.png 不是有效的 PNG")
        elif size != (512, 512):
            problems.append("图标尺寸应为 512×512，当前 %dx%d" % size)

    # 4. 名称一致
    main_py = ROOT / "main.py"
    if main_py.exists():
        source = main_py.read_text(encoding="utf-8", errors="ignore")
        app_title = ""
        for line in source.splitlines():
            if line.strip().startswith("title ="):
                app_title = line.split("=", 1)[1].strip().strip("\"'")
                break
        if app_title and title and app_title != title:
            problems.append("应用名不一致：main.py=%s，buildozer.spec=%s" % (app_title, title))
        if "合安智诊" in source:
            problems.append("main.py 里仍有旧名称“合安智诊”")

    # 5. 构建产物
    for name in FORBIDDEN_DIRS:
        if (ROOT / name).exists():
            problems.append("不应提交的目录：%s/" % name)
    for pattern in FORBIDDEN_GLOBS:
        for hit in ROOT.rglob(pattern):
            if not hit.is_file():
                continue
            if "__pycache__" in str(hit):   # 目录本身已单独报错，这里不重复刷屏
                continue
            problems.append("不应提交的文件：%s" % hit.relative_to(ROOT))

    # 6. 体积
    total = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and "__pycache__" not in str(path):
            total += path.stat().st_size
    total_mb = total / 1024 / 1024
    if total_mb > MAX_TOTAL_MB:
        warnings.append("仓库体积 %.1f MB，偏大（建议检查是否混入构建产物）" % total_mb)

    # 输出
    print("=" * 62)
    print("待提交文件清单（相对仓库根目录）")
    print("=" * 62)
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(d + "/") or rel == d for d in SKIP_DIRS):
            continue
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        print("  %-46s %8.1f KB" % (rel, path.stat().st_size / 1024))
    print("-" * 62)
    print("合计 %.1f MB" % total_mb)

    print()
    if warnings:
        for item in warnings:
            print("[警告]", item)
    if problems:
        for item in problems:
            print("[错误]", item)
        print("\n自检未通过：先修掉上面的问题再 push。")
        return 1
    print("✅ 自检通过：目录结构、模型资源、图标与配置都符合要求，可以 push 到 GitHub。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
