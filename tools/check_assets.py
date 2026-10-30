# -*- coding: utf-8 -*-
"""资源自检：不需要 Kivy，也不需要安卓，直接跑一遍 16 条演示记录。

用法：
    python tools/check_assets.py

会加载 assets 下的模型与数据，对每种（状态 × 负载）组合做一次完整分析，
打印判定、置信度、健康指数与监测状态，任何一条判错就以非零码退出。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bdapp import dsp  # noqa: E402
from bdapp.assets import Bundle  # noqa: E402
from bdapp.model import analyze_record  # noqa: E402


def main() -> int:
    bundle = Bundle()
    extractor = dsp.FeatureExtractor(bundle.config, bundle.kernels)
    fs = float(bundle.config["signal"]["fs"])
    window = float(bundle.config["signal"]["window_seconds"])
    overlap = float(bundle.config["signal"]["overlap"])

    parity_path = ROOT / "assets" / "models" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.exists() else {}

    print("模型版本：", bundle.version)
    print("手机端测试集：Accuracy %.4f  Macro-F1 %.4f"
          % (float(parity.get("app_model_test_accuracy", float("nan"))),
             float(parity.get("app_model_macro_f1", float("nan")))))
    print("树遍历 vs sklearn 最大偏差：", parity.get("rf_probability_max_abs_diff"))
    print()

    failures = 0
    for label, load in bundle.demo_keys():
        signal = bundle.demo_signal(label, load)
        meta = bundle.demo_meta(label, load)
        rpm = float(meta.get("rpm", 1797.0))
        windows, _starts = dsp.segment_signal(signal, fs, window, overlap)
        result = analyze_record(windows, rpm, load, extractor, bundle.forest, bundle.params)
        truth = bundle.params["class_names"][str(label)]
        ok = result["class_name"] == truth
        failures += 0 if ok else 1
        print("%-6s %g hp  ->  %-6s 置信度 %.1f%%  健康 %5.1f %-4s 状态 %s %s"
              % (truth, load, result["class_name"], result["confidence"] * 100,
                 result["health"]["health_index"], result["health"]["health_grade"],
                 result["state"], "OK" if ok else "**错误**"))
    print()
    print("不合格组合：", failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
