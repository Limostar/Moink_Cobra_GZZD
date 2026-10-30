# -*- coding: utf-8 -*-
"""把诊断系统导出成手机 App 可直接读取的资源。

为什么手机端要单独训练一个模型：
    网页端用 scipy 的 sosfiltfilt / welch / hilbert 算特征，而 Android 上装 scipy
    需要交叉编译 Fortran/BLAS，失败率高、包体大。手机端因此用纯 numpy 复刻同一套
    信号处理（滤波核由 scipy 离线生成的等效冲激响应代替）。
    两者在窗口两端会被各自的边界处理影响（误差集中在头尾约 20 个采样点），
    所以手机端**用同一套 numpy 特征重新训练一个同协议模型**（同样的数据、同样的
    1/2 hp 训练 + 3 hp 测试划分），而不是硬套网页端权重——否则边缘差异会直接
    变成误判。训练逻辑完全复用网页端的 bearing_diag.pipeline.train_system。

产出：
    assets/models/rf_model.npz   随机森林树结构
    assets/models/params.json    基线、健康参数、决策阈值、配置、指标
    assets/models/filters.npz    三个零相位滤波核
    assets/models/parity.json    手机端模型测试精度 + 与网页端特征的口径差异
    assets/data/demo.npz         16 条 CWRU 演示记录（4 s/条）

用法：
    python tools/export_assets.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.io import loadmat

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

DEFAULT_PROJECT = Path(r"F:\Hurricane\课程设计-故障诊断\bearing_diagnosis_system\bearing_diagnosis_system")

FILE_MAP = {
    0: {0.0: 97, 1.0: 98, 2.0: 99, 3.0: 100},
    1: {0.0: 105, 1.0: 106, 2.0: 107, 3.0: 108},
    2: {0.0: 130, 1.0: 131, 2.0: 132, 3.0: 133},
    3: {0.0: 118, 1.0: 119, 2.0: 120, 3.0: 121},
}
LOAD_RPM = {0.0: 1797.0, 1.0: 1772.0, 2.0: 1750.0, 3.0: 1730.0}
LOAD_NOTE = {0.0: "验证集", 1.0: "训练集", 2.0: "训练集", 3.0: "独立测试集"}
SPLIT_BY_LOAD = {0.0: "validation", 1.0: "train", 2.0: "train", 3.0: "test"}

KERNEL_LENGTH = 12001
DEMO_SECONDS = 4.0
CLASS_LABELS = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}


def build_kernel(sos: np.ndarray, length: int = KERNEL_LENGTH) -> np.ndarray:
    impulse = np.zeros(length, dtype=np.float64)
    impulse[length // 2] = 1.0
    return scipy_signal.sosfiltfilt(sos, impulse).astype(np.float32)


def read_series(data_dir: Path, file_id: int) -> np.ndarray:
    mat = loadmat(str(data_dir / f"{file_id}.mat"))
    key = f"X{file_id:03d}_DE_time"
    return np.asarray(mat[key], dtype=np.float64).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 APK 资源并训练手机端模型")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        raise SystemExit("找不到网页端项目：%s" % project)
    sys.path.insert(0, str(project))

    from bearing_diag.config import AppConfig
    from bearing_diag.pipeline import train_system

    from bdapp import dsp
    from bdapp.model import analyze_record

    config = AppConfig()
    fs = int(config.signal.fs)
    data_dir = project / "data"
    model_dir = APP_ROOT / "assets" / "models"
    demo_dir = APP_ROOT / "assets" / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. 滤波核 ---------------- #
    sos_hp = scipy_signal.butter(4, config.signal.highpass_hz, btype="highpass", fs=fs, output="sos")
    cutoff = min(config.signal.lowpass_hz or 0.45 * fs, 0.45 * fs)
    sos_lp = scipy_signal.butter(6, cutoff, btype="lowpass", fs=fs, output="sos")
    sos_bp = scipy_signal.butter(
        4, list(config.signal.envelope_band_hz), btype="bandpass", fs=fs, output="sos"
    )
    kernels = {
        "highpass": build_kernel(sos_hp),
        "lowpass": build_kernel(sos_lp),
        "bandpass": build_kernel(sos_bp),
    }
    np.savez_compressed(model_dir / "filters.npz", **kernels)
    kernel64 = {key: value.astype(np.float64) for key, value in kernels.items()}

    # ---------------- 2. 配置骨架 ---------------- #
    app_config = {
        "bearing": {
            "name": config.bearing.name,
            "n_balls": config.bearing.n_balls,
            "ball_diameter_mm": config.bearing.ball_diameter_mm,
            "pitch_diameter_mm": config.bearing.pitch_diameter_mm,
            "contact_angle_deg": config.bearing.contact_angle_deg,
        },
        "signal": {
            "fs": config.signal.fs,
            "window_seconds": config.signal.window_seconds,
            "overlap": config.signal.overlap,
            "sensor_range_g": config.signal.sensor_range_g,
            "envelope_band_hz": list(config.signal.envelope_band_hz),
        },
        "features": {
            "selected_features": list(config.features.selected_features),
            "high_band_hz": list(config.features.high_band_hz),
            "fault_tolerance": config.features.fault_tolerance,
        },
        "alarm": {
            "threshold_sigma": config.alarm.threshold_sigma,
            "confirmations": config.alarm.confirmations,
            "release_confirmations": config.alarm.release_confirmations,
        },
        "health": {
            "healthy_min": config.health.healthy_min,
            "attention_min": config.health.attention_min,
            "degraded_min": config.health.degraded_min,
        },
        "loads": [
            {"load_hp": load, "rpm": LOAD_RPM[load], "file": f"{FILE_MAP[0][load]}.mat", "note": LOAD_NOTE[load]}
            for load in (0.0, 1.0, 2.0, 3.0)
        ],
    }
    extractor = dsp.FeatureExtractor(app_config, kernel64)

    # ---------------- 3. 用 numpy 特征重建特征表 ---------------- #
    rows = []
    demo_signals, demo_labels, demo_loads = [], [], []
    for label in range(4):
        for load in (0.0, 1.0, 2.0, 3.0):
            file_id = FILE_MAP[label][load]
            series = read_series(data_dir, file_id)
            record_id = f"CWRU-{file_id}"
            windows, starts = dsp.segment_signal(
                series, fs, config.signal.window_seconds, config.signal.overlap
            )
            length = int(round(args.seconds * fs))
            start = max(0, series.size // 3)
            chunk = series[start : start + length]
            if chunk.size < length:
                chunk = np.tile(series, int(np.ceil(length / series.size)))[:length]
            demo_signals.append(chunk.astype(np.float32))
            demo_labels.append(label)
            demo_loads.append(load)
            for index, window in enumerate(windows):
                result = extractor.extract(window, LOAD_RPM[load])
                row = {
                    "source": f"CWRU 实测 · {file_id}.mat",
                    "batch_id": f"load-{int(load)}",
                    "record_id": record_id,
                    "window_id": index,
                    "start_sample": int(starts[index]),
                    "fs": fs,
                    "speed_rpm": LOAD_RPM[load],
                    "load_hp": load,
                    "label": label,
                    "class_name": CLASS_LABELS[label],
                    "split": SPLIT_BY_LOAD[load],
                    "quality_valid": bool(result["quality"]["valid"]),
                    "unit": "g",
                }
                row.update(result["values"])
                rows.append(row)
    frame = pd.DataFrame(rows)
    print("特征表：%d 窗 × %d 列" % (frame.shape[0], frame.shape[1]))

    # ---------------- 4. 复用网页端训练流程 ---------------- #
    artifacts = train_system(frame, config, model_path=None)
    bundle = artifacts.model_bundle
    pipeline = bundle["models"]["随机森林"]
    forest = pipeline.named_steps["model"]
    imputer = pipeline.named_steps.get("imputer")
    evaluation = bundle.get("evaluation", {}).get("随机森林", {})

    feature_idx, thresholds, lefts, rights, values, roots = [], [], [], [], [], []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        left = tree.children_left.astype(np.int32).copy()
        right = tree.children_right.astype(np.int32).copy()
        # 每棵树的子节点索引是树内局部编号，拼成一个大数组后必须加上本树的起始偏移，
        # 否则遍历时会跳到别的树的节点上（叶子仍保留 -1）
        left[left >= 0] += offset
        right[right >= 0] += offset
        feature_idx.append(tree.feature.astype(np.int16))
        thresholds.append(tree.threshold.astype(np.float32))
        lefts.append(left)
        rights.append(right)
        values.append(tree.value[:, 0, :].astype(np.float32))
        roots.append(offset)
        offset += tree.node_count
    np.savez_compressed(
        model_dir / "rf_model.npz",
        feature=np.concatenate(feature_idx),
        threshold=np.concatenate(thresholds),
        left=np.concatenate(lefts),
        right=np.concatenate(rights),
        value=np.concatenate(values),
        roots=np.asarray(roots, dtype=np.int32),
    )
    print("随机森林：%d 棵树 / %d 个节点" % (len(forest.estimators_), offset))

    # 自检：手机端的树遍历必须与 sklearn 的概率输出一致
    from bdapp.model import RandomForestModel

    with np.load(model_dir / "rf_model.npz") as payload:
        forest_app = RandomForestModel({key: payload[key] for key in payload.files})
    selected = list(config.features.selected_features)
    probe = frame[selected].to_numpy(dtype=np.float64)[:128]
    reference_probs = pipeline.predict_proba(probe)
    app_probs = forest_app.predict_proba(probe)
    rf_diff = float(np.max(np.abs(reference_probs - app_probs)))
    print("树遍历 vs sklearn 概率最大偏差：%.3e" % rf_diff)
    if rf_diff > 1e-5:
        raise SystemExit("随机森林导出不一致，请检查节点偏移")

    # ---------------- 5. 基线 / 健康 / 参数 ---------------- #
    baselines = {"global": {}, "by_load": {}}
    for key, item in artifacts.baselines.global_baselines.items():
        baselines["global"][key] = {"median": item.median, "threshold": item.threshold, "std": item.std}
    for load, group in artifacts.baselines.by_load.items():
        baselines["by_load"][f"{round(float(load), 1)}"] = {
            key: {"median": item.median, "threshold": item.threshold, "std": item.std}
            for key, item in group.items()
        }

    health = artifacts.health
    health_payload = {
        "weights": {key: float(value) for key, value in health.weights.items()},
        "medians": {key: float(value) for key, value in health.medians.items()},
        "scales": {key: float(value) for key, value in health.scales.items()},
        "by_load": {
            f"{round(float(load), 1)}": {
                "medians": {k: float(v) for k, v in health.medians_by_load.get(load, {}).items()},
                "scales": {k: float(v) for k, v in health.scales_by_load.get(load, {}).items()},
            }
            for load in health.medians_by_load
        },
        "healthy_min": config.health.healthy_min,
        "attention_min": config.health.attention_min,
        "degraded_min": config.health.degraded_min,
    }

    params = {
        "version": bundle.get("version"),
        "trained_at": bundle.get("trained_at"),
        "source": "APK 端（numpy 特征，与网页端同数据同划分）",
        "class_names": {str(key): value for key, value in bundle.get("class_names", {}).items()},
        "config": app_config,
        "baselines": baselines,
        "health": health_payload,
        "metrics": {
            "accuracy": float(evaluation.get("accuracy", float("nan"))),
            "macro_f1": float(evaluation.get("macro_f1", float("nan"))),
            "train_size": int(bundle.get("train_size", 0)),
            "eval_size": int(bundle.get("eval_size", 0)),
            "imputer_statistics": None if imputer is None else np.asarray(imputer.statistics_).tolist(),
            "trees": len(forest.estimators_),
            "web_model_version": None,
        },
    }

    # ---------------- 6. 演示数据 ---------------- #
    np.savez_compressed(
        demo_dir / "demo.npz",
        signals=np.stack(demo_signals),
        labels=np.asarray(demo_labels, dtype=np.int16),
        loads=np.asarray(demo_loads, dtype=np.float32),
    )

    # ---------------- 7. 端到端自检：手机端链路预测 ---------------- #
    predicted, truth = [], []
    for index in range(len(demo_signals)):
        load = float(demo_loads[index])
        windows, _ = dsp.segment_signal(
            demo_signals[index].astype(np.float64), fs, config.signal.window_seconds, config.signal.overlap
        )
        result = analyze_record(windows, LOAD_RPM[load], load, extractor, forest_app, params)
        predicted.append(result["label"])
        truth.append(int(demo_labels[index]))
    accuracy = float(np.mean(np.asarray(predicted) == np.asarray(truth)))

    parity = {
        "app_model_test_accuracy": float(evaluation.get("accuracy", float("nan"))),
        "app_model_macro_f1": float(evaluation.get("macro_f1", float("nan"))),
        "rf_probability_max_abs_diff": rf_diff,
        "demo_accuracy": accuracy,
        "demo_predicted": predicted,
        "demo_truth": truth,
        "note": "手机端模型使用 numpy 特征在本机重训；与网页端 scipy 特征存在头尾边界差异，故不共用权重。",
    }
    (model_dir / "parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")
    params["metrics"]["web_model_version"] = None
    (model_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print("测试集 Accuracy：%.4f  Macro-F1：%.4f" % (evaluation.get("accuracy", float("nan")), evaluation.get("macro_f1", float("nan"))))
    print("16 条演示记录整体判对率：%.1f%%" % (accuracy * 100))
    print("演示预测：", predicted)
    print("真实标签：", truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
