# -*- coding: utf-8 -*-
"""模型与判据：随机森林推理、状态监测、健康评分、运维决策。

随机森林不用 sklearn，而是把 300 棵树导出成扁平数组，运行时用纯 numpy 遍历，
避免在 Android 上安装 scikit-learn（编译链太重、包体大）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

INDICATORS = [
    ("I1_RMS", "rms", "振动总能量"),
    ("I2_FAULT", "fault_frequency_ratio", "故障能量"),
    ("I3_KURTOSIS", "kurtosis", "冲击性"),
]


# --------------------------------------------------------------------------- #
# 随机森林
# --------------------------------------------------------------------------- #
class RandomForestModel:
    """加载导出的树结构并做 soft-voting 概率输出。"""

    def __init__(self, payload: dict):
        self.feature = payload["feature"].astype(np.int32)
        self.threshold = payload["threshold"].astype(np.float64)
        self.left = payload["left"].astype(np.int32)
        self.right = payload["right"].astype(np.int32)
        self.value = payload["value"].astype(np.float64)
        self.roots = payload["roots"].astype(np.int32)
        self.n_classes = int(self.value.shape[1])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        out = np.zeros((matrix.shape[0], self.n_classes), dtype=np.float64)
        for row_index in range(matrix.shape[0]):
            row = matrix[row_index]
            total = np.zeros(self.n_classes, dtype=np.float64)
            for root in self.roots:
                node = int(root)
                while True:
                    left = int(self.left[node])
                    if left < 0:
                        break
                    feature = int(self.feature[node])
                    node = left if row[feature] <= self.threshold[node] else int(self.right[node])
                total += self.value[node]
            summary = float(np.sum(total))
            out[row_index] = total / summary if summary > 0 else np.full(self.n_classes, 1.0 / self.n_classes)
        return out


# --------------------------------------------------------------------------- #
# 状态监测
# --------------------------------------------------------------------------- #
def indicator_values(feature_row: dict, baseline: dict) -> dict:
    values = {}
    for key, feature, _label in INDICATORS:
        raw = max(float(feature_row.get(feature, 0.0)), 1e-6)
        values[key] = float(math.log(raw) - baseline[key]["median"])
    return values


def indicator_thresholds(baseline: dict) -> dict:
    return {key: float(baseline[key]["threshold"]) for key, _f, _l in INDICATORS}


class AlarmStateMachine:
    """与网页端一致：连续 3 窗确认报警，连续 5 窗回落解除，带 5% 滞回。"""

    def __init__(self, confirmations: int = 3, release_confirmations: int = 5):
        self.confirmations = int(confirmations)
        self.release_confirmations = int(release_confirmations)
        self.exceed_counts: dict[str, int] = {}
        self.release_counts: dict[str, int] = {}
        self.active: set[str] = set()
        self.state = "正常"

    def update(self, indicators: dict, thresholds: dict, quality_valid: bool = True) -> dict:
        if not quality_valid:
            self.state = "数据异常"
            return {"state": self.state, "active": []}
        for key, value in indicators.items():
            limit = thresholds.get(key, float("inf"))
            if value > limit:
                self.exceed_counts[key] = self.exceed_counts.get(key, 0) + 1
                self.release_counts[key] = 0
                if self.exceed_counts[key] >= self.confirmations:
                    self.active.add(key)
            else:
                self.exceed_counts[key] = 0
                if key in self.active:
                    hysteresis = max(0.05, abs(limit) * 0.05)
                    if value < limit - hysteresis:
                        self.release_counts[key] = self.release_counts.get(key, 0) + 1
                    else:
                        self.release_counts[key] = 0
                    if self.release_counts[key] >= self.release_confirmations:
                        self.active.discard(key)
                        self.release_counts[key] = 0
        if self.active:
            self.state = "报警"
        elif any(count > 0 for count in self.exceed_counts.values()):
            self.state = "注意"
        else:
            self.state = "正常"
        return {"state": self.state, "active": sorted(self.active)}


# --------------------------------------------------------------------------- #
# 健康评分
# --------------------------------------------------------------------------- #
def health_grade(value: float, config: dict) -> str:
    if value >= float(config["healthy_min"]):
        return "健康"
    if value >= float(config["attention_min"]):
        return "注意"
    if value >= float(config["degraded_min"]):
        return "退化"
    return "严重"


def score_health(feature_row: dict, load_hp: float, health_cfg: dict) -> dict:
    """与网页端 HealthAssessor.score_row 一致：分负载位置 + 1.5σ 死区 + 斜坡。"""
    load_key = f"{round(float(load_hp), 1)}"
    per_load = health_cfg.get("by_load", {}).get(load_key, {})
    medians = per_load.get("medians", health_cfg["medians"])
    sigmas = per_load.get("scales", health_cfg["scales"])
    weights = health_cfg["weights"]

    total = 0.0
    anomalies = {}
    for feature, weight in weights.items():
        value = max(float(feature_row.get(feature, 0.0)), 0.0)
        median = float(medians.get(feature, 0.0))
        sigma = max(float(sigmas.get(feature, 1.0)), 1e-9)
        lower = median + 1.5 * sigma
        z = float(np.clip((value - lower) / (2.0 * sigma), 0.0, 1.0))
        anomalies[feature] = z
        total += float(weight) * z
    abnormality = 1.0 - math.exp(-total)
    health = 100.0 * (1.0 - abnormality)
    return {
        "health_index": health,
        "health_grade": health_grade(health, health_cfg),
        "abnormality": abnormality,
        "anomalies": anomalies,
    }


# --------------------------------------------------------------------------- #
# 运维决策
# --------------------------------------------------------------------------- #
def make_decision(
    quality_valid: bool,
    alarm_state: str,
    health_index: float,
    grade: str,
    fault_class: str,
    confidence: float,
    health_cfg: dict,
) -> dict:
    if not quality_valid:
        return {
            "rule_id": "D0-DATA",
            "urgency": "数据检查",
            "recommendation": "暂停轴承故障判断，检查传感器、线缆、采集通道和安装状态。",
            "rationale": "数据质量无效时，任何健康或故障结论都不可靠。",
            "color": "#D97706",
        }
    if health_index < float(health_cfg["degraded_min"]) and confidence >= 0.60 and fault_class != "正常":
        return {
            "rule_id": "D2-SEVERE",
            "urgency": "立即处置",
            "recommendation": "建议停机检查；重点检查轴承、润滑和安装状态，必要时更换轴承。",
            "rationale": "健康指标 %.1f（%s），诊断为%s，置信度 %.1f%%。"
            % (health_index, grade, fault_class, confidence * 100),
            "color": "#DC2626",
        }
    if alarm_state == "报警" or health_index < float(health_cfg["attention_min"]):
        return {
            "rule_id": "D3-DEGRADED",
            "urgency": "安排停机检查",
            "recommendation": "安排停机检查并复测振动包络谱，核对传感器安装状态和工况变化。",
            "rationale": "报警状态为%s，健康指标 %.1f，故障类别为%s。" % (alarm_state, health_index, fault_class),
            "color": "#EA580C",
        }
    if health_index < float(health_cfg["healthy_min"]) or alarm_state == "注意":
        return {
            "rule_id": "D4-ATTENTION",
            "urgency": "提高检查频次",
            "recommendation": "继续运行，但提高检查频次，关注健康指标和故障频率能量的变化趋势。",
            "rationale": "健康指标 %.1f，报警状态为%s，尚未达到停机条件。" % (health_index, alarm_state),
            "color": "#CA8A04",
        }
    return {
        "rule_id": "D5-NORMAL",
        "urgency": "继续监测",
        "recommendation": "保持当前运行和监测策略，按计划复测并保留历史记录。",
        "rationale": "数据质量正常、无确认报警、健康指标处于健康区间。",
        "color": "#15803D",
    }


# --------------------------------------------------------------------------- #
# 一次完整分析
# --------------------------------------------------------------------------- #
def analyze_record(
    windows: np.ndarray,
    rpm: float,
    load_hp: float,
    extractor,
    forest: RandomForestModel,
    params: dict,
) -> dict:
    """对一条记录的所有 1 s 窗口跑完整链路，返回聚合结果。"""
    config = params["config"]
    selected = list(config["features"]["selected_features"])
    baseline = params["baselines"]["by_load"].get(
        f"{round(float(load_hp), 1)}", params["baselines"]["global"]
    )
    thresholds = indicator_thresholds(baseline)
    machine = AlarmStateMachine(
        config["alarm"]["confirmations"], config["alarm"]["release_confirmations"]
    )

    rows = []
    matrix = []
    for window in windows:
        result = extractor.extract(window, rpm)
        values = result["values"]
        matrix.append([values[name] for name in selected])
        indicators = indicator_values(values, baseline)
        monitor = machine.update(indicators, thresholds, result["quality"]["valid"])
        health = score_health(values, load_hp, params["health"])
        rows.append(
            {
                "features": values,
                "quality": result["quality"],
                "indicators": indicators,
                "state": monitor["state"],
                "active": monitor["active"],
                "health": health,
                "fault_frequencies": result["fault_frequencies"],
                "ratio_by_class": result["ratio_by_class"],
                "harmonic_by_class": result["harmonic_by_class"],
                "dominant_envelope_hz": result["dominant_envelope_hz"],
                "spectra": {
                    "freqs": result["freqs"],
                    "psd": result["psd"],
                    "env_freqs": result["env_freqs"],
                    "env_psd": result["env_psd"],
                    "filtered": result["filtered"],
                },
            }
        )

    probabilities = forest.predict_proba(np.asarray(matrix, dtype=np.float64))
    mean_prob = probabilities.mean(axis=0) if probabilities.size else np.zeros(4)
    label = int(np.argmax(mean_prob))
    class_names = params["class_names"]
    class_of = lambda idx: class_names.get(str(idx), class_names.get(idx, "未知"))
    latest = rows[-1]
    decision = make_decision(
        quality_valid=bool(latest["quality"]["valid"]),
        alarm_state=latest["state"],
        health_index=float(latest["health"]["health_index"]),
        grade=latest["health"]["health_grade"],
        fault_class=class_of(label),
        confidence=float(mean_prob[label]),
        health_cfg=params["config"]["health"],
    )
    return {
        "rows": rows,
        "probabilities": mean_prob,
        "label": label,
        "class_name": class_of(label),
        "confidence": float(mean_prob[label]),
        "health": latest["health"],
        "state": latest["state"],
        "active_indicators": latest["active"],
        "quality": latest["quality"],
        "decision": decision,
        "thresholds": thresholds,
    }


def load_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
