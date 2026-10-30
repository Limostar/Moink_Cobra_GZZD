# -*- coding: utf-8 -*-
"""打包资源加载：模型参数、树结构、滤波核、演示数据。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .model import RandomForestModel

_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [_ROOT, Path.cwd(), Path.cwd().parent]


def asset_path(*parts: str) -> Path:
    """在若干候选根目录下找资源，兼容桌面调试与 Android 打包后的路径。"""
    for root in _CANDIDATES:
        candidate = Path(root).joinpath("assets", *parts)
        if candidate.exists():
            return candidate
    return _ROOT.joinpath("assets", *parts)


class Bundle:
    """一次性把用到的资源读进内存。"""

    def __init__(self) -> None:
        self.params = json.loads(asset_path("models", "params.json").read_text(encoding="utf-8"))
        with np.load(asset_path("models", "rf_model.npz")) as payload:
            self.forest = RandomForestModel({key: payload[key] for key in payload.files})
        self.kernels = {}
        with np.load(asset_path("models", "filters.npz")) as payload:
            for key in payload.files:
                if key != "meta":
                    self.kernels[key] = payload[key].astype(np.float64)
        with np.load(asset_path("data", "demo.npz"), allow_pickle=False) as payload:
            self.demo_signals = payload["signals"].astype(np.float64)
            self.demo_labels = payload["labels"].astype(int)
            self.demo_loads = payload["loads"].astype(float)
            self.demo_index = {
                (int(label), round(float(load), 1)): index
                for index, (label, load) in enumerate(zip(self.demo_labels, self.demo_loads))
            }

    # -- 便捷访问 -------------------------------------------------------- #
    @property
    def config(self) -> dict:
        return self.params["config"]

    @property
    def version(self) -> str:
        return self.params.get("version", "unknown")

    def demo_keys(self) -> list[tuple[int, float]]:
        return sorted(self.demo_index.keys())

    def demo_signal(self, label: int, load_hp: float) -> np.ndarray | None:
        index = self.demo_index.get((int(label), round(float(load_hp), 1)))
        return None if index is None else self.demo_signals[index]

    def demo_meta(self, label: int, load_hp: float) -> dict:
        for item in self.params["config"]["loads"]:
            if round(float(item["load_hp"]), 1) == round(float(load_hp), 1):
                return item
        return {"rpm": 1797.0, "load_hp": float(load_hp), "file": ""}

    @staticmethod
    def exists() -> bool:
        return all(
            asset_path("models", name).exists()
            for name in ("params.json", "rf_model.npz", "filters.npz")
        )
