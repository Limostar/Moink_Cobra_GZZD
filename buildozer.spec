[app]

# 应用名称（手机桌面显示的名字）
title = 智能诊断

# 应用图标：512×512 PNG，频谱（频率）造型
icon.filename = %(source.dir)s/assets/icon.png

# 包名：com.med.bearingdiag
package.name = bearingdiag
package.domain = com.med

# 源码目录
source.dir = .
# npz 必须显式加进来，否则模型与演示数据不会被打进 APK（默认清单里没有 npz）
source.include_exts = py,kv,ttf,otf,png,jpg,json,npz
source.exclude_dirs = build, dist, bin, .git, .github, .buildozer, __pycache__, tools, p4a-recipes

version = 1.0.0

# 只需要 Kivy 与 numpy：滤波、谱分析、随机森林推理全部用 numpy 实现，
# 不引入 scipy / sklearn，避免 Android 上的交叉编译问题
requirements = python3,kivy,numpy

p4a.branch = master

# 本地 recipe 覆盖：p4a 自带的 numpy 固定在 v2.3.0，该版本的 unique.cpp 少了
# #include <unordered_map>，在 NDK 的 libc++ 下会编译失败（详见 p4a-recipes/numpy）。
# 这里指向仓库自带的 recipe 目录，构建时会优先使用它。
p4a.local_recipes = %(source.dir)s/p4a-recipes

orientation = portrait
fullscreen = 0

# ----------------------------------------------------------------------
# Android 配置
# ----------------------------------------------------------------------
android.api = 33
android.minapi = 24
android.ndk = 25b

# 只编译 64 位架构，构建更快、包更小
android.archs = arm64-v8a

android.enable_androidx = True
android.accept_sdk_license = True
android.allow_backup = False

[buildozer]
log_level = 2
warn_on_root = 1
