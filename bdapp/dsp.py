# -*- coding: utf-8 -*-
"""信号处理：纯 numpy 实现，与网页端 scipy 版本逐项对齐。

为什么不用 scipy：
    python-for-android 上 scipy 需要交叉编译 Fortran/BLAS，失败率很高、包体也大。
    因此把用到的滤波、Welch 谱、Hilbert 包络、峭度全部用 numpy 重写。

如何保证与训练时一致：
    - 滤波器不是现设计的，而是把 scipy `sosfiltfilt` 的等效冲激响应（零相位）离线算好，
      打包进 assets/filters.npz，运行时只做 FFT 卷积；
    - Welch、Hilbert、峭度按 scipy 的公式逐条复刻；
    - 导出脚本 tools/export_assets.py 会做逐特征对拍，误差和标签一致率会写进报告。
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-15


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def as_float_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """等价于 scipy.signal.detrend(type='linear')。"""
    n = x.size
    if n < 2:
        return x - (x.mean() if n else 0.0)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean()
    denominator = np.sum((t - t_mean) ** 2)
    if denominator <= 0:
        return x - x_mean
    slope = np.sum((t - t_mean) * (x - x_mean)) / denominator
    return x - (slope * (t - t_mean) + x_mean)


def detrend_constant(x: np.ndarray) -> np.ndarray:
    return x - x.mean() if x.size else x


def apply_kernel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """零相位 FIR 卷积，输出与输入等长（等效于 scipy sosfiltfilt 的效果）。"""
    n = x.size
    if kernel is None or kernel.size < 2 or n < 2:
        return x.astype(np.float64, copy=True)
    nfft = 1 << int(np.ceil(np.log2(n + kernel.size - 1)))
    spectrum = np.fft.rfft(x, nfft) * np.fft.rfft(kernel, nfft)
    y = np.fft.irfft(spectrum, nfft)
    start = (kernel.size - 1) // 2
    return y[start : start + n]


def periodic_hann(n: int) -> np.ndarray:
    """周期 Hann 窗，与 scipy.signal.get_window('hann', n, fftbins=True) 一致。"""
    n = int(n)
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float64)
    index = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * index / n)


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert 解析信号的模，等价于 numpy.abs(scipy.signal.hilbert(x))。"""
    n = x.size
    if n < 2:
        return np.abs(x)
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(n, dtype=np.float64)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[1 : n // 2] = 2.0
        multiplier[n // 2] = 1.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch 功率谱密度，参数与 scipy.signal.welch(window='hann', detrend='constant',
    scaling='density', noverlap=nperseg//2) 对齐。"""
    x = as_float_1d(x)
    if x.size < 8:
        return np.array([0.0]), np.array([0.0])
    nperseg = min(int(nperseg or int(min(4096, max(64, x.size // 4)))), x.size)
    nperseg = max(8, nperseg)
    step = max(1, nperseg // 2)
    window = periodic_hann(nperseg)
    scale = 1.0 / (float(fs) * np.sum(window ** 2))
    starts = range(0, x.size - nperseg + 1, step)
    accumulator = np.zeros(nperseg // 2 + 1, dtype=np.float64)
    count = 0
    for start in starts:
        segment = x[start : start + nperseg]
        segment = segment - segment.mean()
        spectrum = np.fft.rfft(segment * window)
        accumulator += (spectrum.real ** 2 + spectrum.imag ** 2)
        count += 1
    if count == 0:
        segment = x[:nperseg] - x[:nperseg].mean()
        spectrum = np.fft.rfft(segment * window)
        accumulator += (spectrum.real ** 2 + spectrum.imag ** 2)
        count = 1
    psd = accumulator * (scale / count)
    if nperseg % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    return freqs, np.maximum(psd, 0.0)


def spectral_metrics(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    if psd.size == 0 or float(np.sum(psd)) <= 0:
        return 0.0, 0.0
    p = psd / np.sum(psd)
    centroid = float(np.sum(freqs * p))
    entropy = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
    max_entropy = math.log2(max(2, p.size))
    return centroid, entropy / max_entropy if max_entropy else 0.0


def kurtosis_excess_pearson(x: np.ndarray) -> float:
    """等价于 scipy.stats.kurtosis(x, fisher=False, bias=False)。"""
    n = x.size
    if n <= 3:
        return 3.0
    d = x - x.mean()
    m2 = float(np.mean(d ** 2))
    m4 = float(np.mean(d ** 4))
    if m2 <= 0:
        return 3.0
    value = ((n ** 2 - 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    value += 3.0
    return float(value) if np.isfinite(value) else 3.0


# --------------------------------------------------------------------------- #
# 质量检查与分段
# --------------------------------------------------------------------------- #
def longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if edges.size < 2:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


def check_quality(x: np.ndarray, fs: float, sensor_range_g: float = 50.0) -> dict:
    x = as_float_1d(x)
    finite_ratio = float(np.isfinite(x).mean()) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    peak_abs = float(np.max(np.abs(x))) if x.size else 0.0
    zero_run_seconds = longest_true_run(np.isclose(x, 0.0, atol=1e-12)) / fs if fs > 0 else 0.0
    clipping = float(np.mean(np.abs(x) >= 0.995 * sensor_range_g)) if sensor_range_g > 0 else 0.0
    status, message = "VALID", "采集链路正常"
    if x.size == 0:
        status, message = "NO_DATA", "没有有效采样点"
    elif finite_ratio < 0.995:
        status, message = "NONFINITE", "存在过多非有限值"
    elif rms < 1e-9:
        status, message = "FLATLINE", "信号近似恒值，可能断连"
    elif zero_run_seconds >= 0.05:
        status, message = "FLATLINE", "连续零值 %.3f s" % zero_run_seconds
    elif clipping > 0.001:
        status, message = "CLIPPED", "疑似超出传感器量程"
    return {
        "valid": status == "VALID",
        "status": status,
        "message": message,
        "rms": rms,
        "peak_abs": peak_abs,
        "zero_run_seconds": zero_run_seconds,
        "clipping_ratio": clipping,
    }


def segment_signal(x: np.ndarray, fs: float, window_seconds: float = 1.0, overlap: float = 0.5):
    x = as_float_1d(x)
    length = max(8, int(round(window_seconds * fs)))
    hop = max(1, int(round(length * (1.0 - float(np.clip(overlap, 0.0, 0.95))))))
    if x.size == 0:
        return np.empty((0, length)), np.empty(0, dtype=int)
    if x.size < length:
        pad = length - x.size
        left = pad // 2
        x = np.pad(x, (left, pad - left), mode="edge")
    starts = np.arange(0, x.size - length + 1, hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    return np.stack([x[s : s + length] for s in starts]), starts


# --------------------------------------------------------------------------- #
# 特征提取（与网页端 features.extract_features 对齐）
# --------------------------------------------------------------------------- #
def bearing_frequencies(bearing: dict, rpm: float) -> dict:
    fr = max(float(rpm), 1e-9) / 60.0
    n = float(bearing["n_balls"])
    d = float(bearing["ball_diameter_mm"])
    pitch = float(bearing["pitch_diameter_mm"])
    alpha = math.radians(float(bearing["contact_angle_deg"]))
    ratio = (d / pitch) * math.cos(alpha)
    return {
        "shaft": fr,
        "FTF": 0.5 * fr * (1.0 - ratio),
        "BPFO": 0.5 * n * fr * (1.0 - ratio),
        "BPFI": 0.5 * n * fr * (1.0 + ratio),
        "BSF": 0.5 * (pitch / d) * fr * (1.0 - ratio ** 2),
    }


def _power_in_band(freqs, psd, center, relative_tolerance=0.02) -> float:
    tolerance = max(1.5, abs(center) * relative_tolerance)
    mask = np.abs(freqs - center) <= tolerance
    return float(np.sum(psd[mask])) if np.any(mask) else 0.0


def _peak_ratio(freqs, psd, center, relative_tolerance=0.02) -> float:
    tolerance = max(1.5, abs(center) * relative_tolerance)
    mask = np.abs(freqs - center) <= tolerance
    if not np.any(mask):
        return 0.0
    noise = float(np.median(psd) + 1e-15)
    return float(np.max(psd[mask]) / noise)


class FeatureExtractor:
    """把滤波核、配置一次装好，之后对每个窗口调用 extract()。"""

    def __init__(self, config: dict, kernels: dict):
        self.config = config
        signal = config["signal"]
        self.fs = float(signal["fs"])
        self.load_cfg = config["loads"]
        self.kernel_hp = kernels.get("highpass")
        self.kernel_lp = kernels.get("lowpass")
        self.kernel_bp = kernels.get("bandpass")
        self.window_seconds = float(signal["window_seconds"])
        self.overlap = float(signal["overlap"])
        self.env_band = tuple(signal["envelope_band_hz"])
        self.high_band = tuple(config["features"]["high_band_hz"])
        self.tolerance = float(config["features"]["fault_tolerance"])
        self.sensor_range = float(signal["sensor_range_g"])
        self.bearing = config["bearing"]
        self.selected = list(config["features"]["selected_features"])

    # -- 预处理 ---------------------------------------------------------- #
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        y = detrend_linear(as_float_1d(x))
        if self.kernel_hp is not None:
            y = apply_kernel(y, self.kernel_hp)
        if self.kernel_lp is not None:
            y = apply_kernel(y, self.kernel_lp)
        return y

    def envelope(self, filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = apply_kernel(filtered, self.kernel_bp) if self.kernel_bp is not None else filtered
        env = detrend_constant(analytic_envelope(y))
        return welch_psd(env, self.fs)

    # -- 主入口 ---------------------------------------------------------- #
    def extract(self, raw_window: np.ndarray, rpm: float) -> dict:
        raw = as_float_1d(raw_window)
        quality = check_quality(raw, self.fs, self.sensor_range)
        filtered = self.preprocess(raw)
        freqs, psd = welch_psd(filtered, self.fs)
        env_freqs, env_psd = self.envelope(filtered)
        centroid, entropy = spectral_metrics(freqs, psd)

        rms = float(np.sqrt(np.mean(filtered ** 2))) if filtered.size else 0.0
        peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
        mean_abs = float(np.mean(np.abs(filtered))) if filtered.size else 0.0
        kurt = kurtosis_excess_pearson(filtered)
        crest = peak / rms if rms > 1e-12 else 0.0
        impulse = peak / mean_abs if mean_abs > 1e-12 else 0.0

        low, high = self.high_band
        mask = (freqs >= low) & (freqs <= min(high, 0.45 * self.fs))
        total = float(np.sum(psd) + 1e-15)
        high_ratio = float(np.sum(psd[mask]) / total)

        freq_map = bearing_frequencies(self.bearing, rpm)
        candidates = {
            "inner_race": freq_map["BPFI"],
            "outer_race": freq_map["BPFO"],
            "ball": freq_map["BSF"],
        }
        ratio_by_class, harmonic_by_class = {}, {}
        for name, center in candidates.items():
            harmonic_power = 0.0
            peak_ratios = []
            for harmonic in (1.0, 2.0, 3.0):
                target = harmonic * center
                if target >= 0.95 * float(np.max(env_freqs)):
                    peak_ratios.append(0.0)
                    continue
                harmonic_power += _power_in_band(env_freqs, env_psd, target, self.tolerance)
                peak_ratios.append(_peak_ratio(env_freqs, env_psd, target, self.tolerance))
            ratio_by_class[name] = harmonic_power / (float(np.sum(env_psd)) + 1e-15)
            harmonic_by_class[name] = float(
                np.mean(np.clip(np.asarray(peak_ratios) / 8.0, 0.0, 1.0))
            )

        values = {
            "rms": rms,
            "kurtosis": kurt,
            "crest_factor": crest,
            "impulse_factor": impulse,
            "high_band_ratio": high_ratio,
            "spectral_centroid": centroid,
            "spectral_entropy": entropy,
            "fault_frequency_ratio": max(ratio_by_class.values(), default=0.0),
            "harmonic_consistency": max(harmonic_by_class.values(), default=0.0),
        }
        dominant = float(env_freqs[int(np.argmax(env_psd))]) if env_psd.size else 0.0
        return {
            "values": values,
            "quality": quality,
            "fault_frequencies": freq_map,
            "ratio_by_class": ratio_by_class,
            "harmonic_by_class": harmonic_by_class,
            "dominant_envelope_hz": dominant,
            "filtered": filtered,
            "freqs": freqs,
            "psd": psd,
            "env_freqs": env_freqs,
            "env_psd": env_psd,
        }
