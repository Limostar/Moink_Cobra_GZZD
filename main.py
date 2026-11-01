# -*- coding: utf-8 -*-
"""智能诊断 · 轴承故障诊断（Android APK 版）

把课程设计里的网页端诊断系统搬到手机上，全部计算在本地完成：
    采集回放 → 质量检查 → 滤波/包络 → 9 项特征 → 随机森林四分类
    → 3 个状态指标与 3σ 阈值 → 健康评分 → 运维决策

五个页面：总览 / 采集 / 分析 / 监测 / 关于。
所有模型与演示数据都打包在 assets 里，断网也能用。
"""

from __future__ import annotations

import json
import os
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.logger import Logger
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex

from bdapp import dsp
from bdapp.assets import Bundle
from bdapp.model import analyze_record
from bdapp.model import RandomForestModel  # noqa: F401  (类型提示用)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# 主题
# --------------------------------------------------------------------------- #
BG = "#EEF4F6"
CARD = "#FFFFFF"
TEAL = "#0B6E6E"
TEAL_DARK = "#075454"
TEAL_LIGHT = "#D9EDED"
TEXT = "#12232B"
MUTED = "#6B7F8A"
LINE = "#DCE7EC"
OK = "#15803D"
WARN = "#CA8A04"
DANGER = "#DC2626"
INFO = "#2563EB"

CLASS_COLORS = {0: OK, 1: WARN, 2: DANGER, 3: INFO}
CLASS_EMOJI = {0: "●", 1: "◆", 2: "▲", 3: "■"}

FONT_NAME = "Roboto"
SYSTEM_FONTS = [
    os.path.join(APP_DIR, "assets", "fonts", "NotoSansSC-Regular.ttf"),
    "/system/fonts/NotoSansSC-Regular.otf",
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "/system/fonts/DroidSansFallbackFull.ttf",
    "/system/fonts/DroidSansFallback.ttf",
]


def register_fonts() -> str:
    """注册中文字体，找不到就用系统兜底。"""
    for path in SYSTEM_FONTS:
        if not path or not os.path.exists(path):
            continue
        try:
            LabelBase.register(name="CJK", fn_regular=path)
            return "CJK"
        except Exception:  # pragma: no cover - 仅在异常设备上触发
            Logger.exception("BearingApp: 字体注册失败 %s", path)
    return "Roboto"


FONT_NAME = register_fonts()


def col(hex_color: str):
    return get_color_from_hex(hex_color)


def label(text: str = "", size: float = 13, bold: bool = False, color: str = TEXT,
          halign: str = "left", valign: str = "middle", markup: bool = False,
          auto_height: bool = False, **kwargs) -> Label:
    widget = Label(
        text=text,
        font_name=FONT_NAME,
        font_size=sp(size),
        bold=bold,
        color=col(color),
        halign=halign,
        valign=valign,
        markup=markup,
        **kwargs,
    )
    if auto_height:
        widget.size_hint_y = None
        widget.bind(
            width=lambda inst, value: setattr(inst, "text_size", (value, None)),
            texture_size=lambda inst, value: setattr(inst, "height", value[1]),
        )
    else:
        widget.bind(size=lambda inst, _value: setattr(inst, "text_size", (inst.width, None)))
    return widget



def fmt(value: float, digits: int = 3) -> str:
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- #
# 基础控件
# --------------------------------------------------------------------------- #
class Card(BoxLayout):
    """白底圆角卡片。"""

    def __init__(self, **kwargs):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding", (dp(14), dp(12), dp(14), dp(12)))
        kwargs.setdefault("spacing", dp(6))
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*col(CARD))
            self._rect = RoundedRectangle(radius=[dp(16)], pos=self.pos, size=self.size)
            Color(*col(LINE))
            self._border = Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(16)), width=1)
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *_args):
        self._rect.pos = self.pos
        self._rect.size = self.size
        self._border.rounded_rectangle = (self.x, self.y, self.width, self.height, dp(16))

    def auto(self):
        """高度随内容自适应（内部子控件必须显式给高度）。"""
        self.size_hint_y = None
        self.bind(minimum_height=self.setter("height"))
        return self


class Chip(Button):
    """圆角选择按钮。"""

    def __init__(self, text: str, active: bool = False, **kwargs):
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("height", dp(38))
        kwargs.setdefault("font_name", FONT_NAME)
        kwargs.setdefault("font_size", sp(13))
        super().__init__(text=text, **kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.active = active
        self.color = col("#FFFFFF" if active else TEXT)
        with self.canvas.before:
            self._bg_color = Color(*(col(TEAL) if active else col("#FFFFFF")))
            self._bg = RoundedRectangle(radius=[dp(12)], pos=self.pos, size=self.size)
            self._edge_color = Color(*(col(TEAL) if active else col(LINE)))
            self._edge = Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(12)), width=1)
        self.bind(pos=self._sync, size=self._sync)
        self.bind(width=lambda *_: setattr(self, "text_size", (self.width - dp(8), None)))

    def _sync(self, *_args):
        self._bg.pos = self.pos
        self._bg.size = self.size
        self._edge.rounded_rectangle = (self.x, self.y, self.width, self.height, dp(12))

    def set_active(self, active: bool) -> None:
        self.active = active
        self._bg_color.rgba = col(TEAL) if active else col("#FFFFFF")
        self._edge_color.rgba = col(TEAL) if active else col(LINE)
        self.color = col("#FFFFFF" if active else TEXT)


class Bar(Widget):
    """水平比例条：value ∈ [0,1]，按比例填充。"""

    value = NumericProperty(0.0)

    def __init__(self, value: float = 0.0, color: str = TEAL, **kwargs):
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(8))
        super().__init__(**kwargs)
        self._color = color
        with self.canvas.before:
            Color(*col("#E4EDF1"))
            self._bg = RoundedRectangle(radius=[dp(4)], pos=self.pos, size=self.size)
            self._fill_color = Color(*col(color))
            self._fill = RoundedRectangle(radius=[dp(4)], pos=self.pos, size=(0, 0))
        self.bind(pos=self._sync, size=self._sync, value=lambda *_: self._sync())
        self.value = max(0.0, min(1.0, float(value)))
        self._sync()

    def _sync(self, *_args):
        self._bg.pos = self.pos
        self._bg.size = self.size
        ratio = max(0.0, min(1.0, float(self.value)))
        width = self.width * ratio
        height = self.height
        # 圆角不能超过条高或填充宽度的一半，否则圆角网格会退化
        self._fill.radius = [max(0.0, min(dp(4), height / 2.0, width / 2.0))]
        if width < 0.5:
            # 概率接近 0 时整段隐藏，避免极窄网格在手机 GPU 上画出杂点或错位色块
            self._fill_color.a = 0.0
            self._fill.pos = self.pos
            self._fill.size = (0.0, 0.0)
            return
        self._fill_color.a = 1.0
        self._fill.pos = self.pos
        self._fill.size = (width, height)

    def set_value(self, value: float, color: str | None = None) -> None:
        if color is not None:
            self._color = color
            self._fill_color.rgba = col(color)
        self.value = max(0.0, min(1.0, float(value)))
        self._sync()

    def refresh(self) -> None:
        """布局完成后可主动再对齐一次填充宽度。"""
        self._sync()


class MiniChart(Widget):
    """轻量折线图：把序列抽稀到 ~280 点再用 Line 画，手机上不卡。"""

    def __init__(self, color: str = TEAL, fill: bool = False, **kwargs):
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(120))
        super().__init__(**kwargs)
        self._color = color
        self._fill = fill
        self._xs = None
        self._ys = None
        self._markers = []
        self.bind(pos=lambda *_: self.redraw(), size=lambda *_: self.redraw())

    def set_data(self, xs, ys, markers=None) -> None:
        self._xs = None if xs is None else [float(v) for v in xs]
        self._ys = None if ys is None else [float(v) for v in ys]
        self._markers = markers or []
        self.redraw()

    def redraw(self) -> None:
        self.canvas.clear()
        with self.canvas:
            Color(*col("#F7FAFB"))
            RoundedRectangle(radius=[dp(10)], pos=self.pos, size=self.size)
            Color(*col(LINE))
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(10)), width=1)
            if not self._ys or len(self._ys) < 2:
                return
            values = self._ys
            step = max(1, len(values) // 280)
            series = values[::step]
            if len(series) < 2:
                series = values
            low, high = min(series), max(series)
            span = (high - low) or 1.0
            pad = dp(8)
            width = max(1.0, self.width - 2 * pad)
            height = max(1.0, self.height - 2 * pad)
            points = []
            for index, value in enumerate(series):
                x = self.x + pad + width * index / (len(series) - 1)
                y = self.y + pad + height * (value - low) / span
                points.extend([x, y])
            Color(*col(self._color))
            Line(points=points, width=1.3)
            for marker in self._markers:
                try:
                    position = int(marker)
                    if 0 <= position < len(series):
                        x = self.x + pad + width * position / (len(series) - 1)
                        Color(*col(DANGER))
                        Line(ellipse=(x - dp(3), self.y + pad - dp(3), dp(6), dp(6)))
                except (TypeError, ValueError):
                    continue


class NavButton(Button):
    def __init__(self, text: str, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("font_name", FONT_NAME)
        kwargs.setdefault("font_size", sp(12))
        super().__init__(text=text, **kwargs)

    def set_active(self, active: bool) -> None:
        self.color = col(TEAL if active else MUTED)
        self.bold = active


# --------------------------------------------------------------------------- #
# 应用状态
# --------------------------------------------------------------------------- #
class State:
    def __init__(self) -> None:
        self.bundle = Bundle()
        self.extractor = dsp.FeatureExtractor(self.bundle.config, self.bundle.kernels)
        self.label_true = 2
        self.load_hp = 0.0
        self.rpm = 1797.0
        self.signal = None
        self.windows = None
        self.analysis = None
        self.compute_ms = 0.0
        self.history = []
        self._history_path = os.path.join(App.get_running_app().user_data_dir, "history.json")
        self.load_history()

    # -- 数据 ------------------------------------------------------------ #
    def load_sample(self, label: int, load_hp: float) -> bool:
        signal = self.bundle.demo_signal(label, load_hp)
        if signal is None:
            return False
        meta = self.bundle.demo_meta(label, load_hp)
        self.label_true = int(label)
        self.load_hp = float(load_hp)
        self.rpm = float(meta.get("rpm", 1797.0))
        self.signal = signal
        self.windows, _starts = dsp.segment_signal(
            signal,
            float(self.bundle.config["signal"]["fs"]),
            float(self.bundle.config["signal"]["window_seconds"]),
            float(self.bundle.config["signal"]["overlap"]),
        )
        return True

    def set_rpm(self, rpm: float) -> None:
        """手动指定转速：只影响特征频率与指标，不改变信号与真实标签。"""
        self.rpm = float(max(300.0, min(6000.0, float(rpm))))

    def analyze(self) -> dict:
        started = time.perf_counter()
        self.analysis = analyze_record(
            self.windows, self.rpm, self.load_hp,
            self.extractor, self.bundle.forest, self.bundle.params,
        )
        self.compute_ms = (time.perf_counter() - started) * 1000.0
        return self.analysis

    # -- 历史 ------------------------------------------------------------ #
    def load_history(self) -> None:
        try:
            with open(self._history_path, "r", encoding="utf-8") as handle:
                self.history = json.load(handle)
        except (OSError, ValueError):
            self.history = []

    def push_history(self) -> None:
        if not self.analysis:
            return
        entry = {
            "time": time.strftime("%m-%d %H:%M:%S"),
            "label_true": self.label_true,
            "load_hp": self.load_hp,
            "class_name": self.analysis["class_name"],
            "confidence": round(float(self.analysis["confidence"]), 4),
            "health": round(float(self.analysis["health"]["health_index"]), 1),
            "state": self.analysis["state"],
        }
        self.history.append(entry)
        self.history = self.history[-20:]
        try:
            os.makedirs(os.path.dirname(self._history_path), exist_ok=True)
            with open(self._history_path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, ensure_ascii=False, indent=2)
        except OSError:
            Logger.exception("BearingApp: 历史记录写入失败")


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
class ScrollPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = BoxLayout(orientation="vertical")
        self.view = ScrollView(bar_width=dp(3), do_scroll_x=False)
        self.body = BoxLayout(
            orientation="vertical", spacing=dp(10),
            padding=(dp(12), dp(10), dp(12), dp(18)), size_hint_y=None,
        )
        self.body.bind(minimum_height=self.body.setter("height"))
        self.view.add_widget(self.body)
        self.root.add_widget(self.view)
        self.add_widget(self.root)

    def add(self, widget: Widget) -> Widget:
        self.body.add_widget(widget)
        return widget


class OverviewPage(ScrollPage):
    def __init__(self, state: State, **kwargs):
        super().__init__(name="overview", **kwargs)
        self.state = state

        card = Card().auto()
        self.verdict = label("—", size=26, bold=True, color=TEAL,
                             size_hint_y=None, height=dp(38))
        self.verdict_desc = label("等待采集", size=12, color=MUTED,
                                  size_hint_y=None, height=dp(20))
        card.add_widget(self.verdict)
        card.add_widget(self.verdict_desc)
        # 每条概率分两行：上行名称与百分比，下行进度条，避免三者挤在一行互相遮挡
        self.prob_bars = []
        for name in ("正常", "内圈故障", "外圈故障", "滚动体故障"):
            row = BoxLayout(orientation="vertical", size_hint_y=None,
                            height=dp(38), spacing=dp(3))
            head = BoxLayout(size_hint_y=None, height=dp(18), spacing=dp(6))
            head.add_widget(label(name, size=11.5, color=MUTED))
            value = label("0.0%", size=11.5, color=MUTED, halign="right",
                          size_hint_x=None, width=dp(54))
            head.add_widget(value)
            row.add_widget(head)
            bar = Bar(0.0, TEAL, size_hint_y=None, height=dp(9))
            row.add_widget(bar)
            self.prob_bars.append((bar, value))
            card.add_widget(row)
        self.add(card)

        metrics = BoxLayout(size_hint_y=None, height=dp(88), spacing=dp(8))
        self.health_tile = self._tile("健康指数", "—", "等待评估")
        self.state_tile = self._tile("监测状态", "—", "—")
        self.quality_tile = self._tile("数据质量", "—", "—")
        for tile in (self.health_tile[0], self.state_tile[0], self.quality_tile[0]):
            metrics.add_widget(tile)
        self.add(metrics)

        decision = Card().auto()
        decision.add_widget(label("运维决策", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.decision_urgency = label("—", size=16, bold=True, color=TEAL,
                                      size_hint_y=None, height=dp(28))
        decision.add_widget(self.decision_urgency)
        self.decision_text = label("—", size=12, color=TEXT, auto_height=True)
        decision.add_widget(self.decision_text)
        self.add(decision)

        info = Card().auto()
        info.add_widget(label("当前样本", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.sample_info = label("—", size=11.5, color=MUTED, auto_height=True)
        info.add_widget(self.sample_info)
        self.add(info)

    def _tile(self, title: str, value: str, note: str):
        card = Card(size_hint_y=None, padding=(dp(10), dp(8)), spacing=dp(2)).auto()
        card.add_widget(label(title, size=11, color=MUTED, size_hint_y=None, height=dp(16)))
        value_label = label(value, size=19, bold=True, size_hint_y=None, height=dp(28))
        note_label = label(note, size=10, color=MUTED, size_hint_y=None, height=dp(16))
        card.add_widget(value_label)
        card.add_widget(note_label)
        return card, value_label, note_label


    def update(self) -> None:
        analysis = self.state.analysis
        if not analysis:
            return
        name = analysis["class_name"]
        index = analysis["label"]
        self.verdict.text = name
        self.verdict.color = col(CLASS_COLORS.get(index, TEAL))
        self.verdict_desc.text = "置信度 %.1f%% · 真值 %s · %s" % (
            analysis["confidence"] * 100,
            self.state.bundle.params["class_names"][str(self.state.label_true)],
            "判定一致" if index == self.state.label_true else "判定不一致",
        )
        for i, (bar, value) in enumerate(self.prob_bars):
            prob = float(analysis["probabilities"][i])
            bar.set_value(prob, CLASS_COLORS.get(i, TEAL))
            value.text = "%.1f%%" % (prob * 100)
        # 布局可能比数据晚一帧（切页、卡片高度变化），下一帧再对齐一次
        Clock.schedule_once(lambda _dt: self._resync_bars(), 0)

        health = analysis["health"]
        self.health_tile[1].text = fmt(health["health_index"], 1)
        self.health_tile[1].color = col(
            OK if health["health_index"] >= 85 else WARN if health["health_index"] >= 65 else DANGER
        )
        self.health_tile[2].text = health["health_grade"]

        state_color = {"正常": OK, "注意": WARN, "报警": DANGER, "数据异常": MUTED}.get(analysis["state"], MUTED)
        self.state_tile[1].text = analysis["state"]
        self.state_tile[1].color = col(state_color)
        short = {"I1_RMS": "I1", "I2_FAULT": "I2", "I3_KURTOSIS": "I3"}
        active = [short.get(name, name) for name in analysis["active_indicators"]]
        self.state_tile[2].text = ("超限 " + " ".join(active)) if active else "无超限指标"

        quality = analysis["quality"]
        self.quality_tile[1].text = "有效" if quality["valid"] else "异常"
        self.quality_tile[1].color = col(OK if quality["valid"] else DANGER)
        self.quality_tile[2].text = quality["status"]

        decision = analysis["decision"]
        self.decision_urgency.text = "%s · %s" % (decision["urgency"], decision["rule_id"])
        self.decision_urgency.color = col(decision["color"])
        self.decision_text.text = "%s\n依据：%s" % (decision["recommendation"], decision["rationale"])

        self.sample_info.text = (
            "状态 %s ｜ 负载 %g hp ｜ 转速 %.0f r/min\n窗口 %d 个 ｜ 用时 %.0f ms ｜ 模型 %s"
            % (
                self.state.bundle.params["class_names"][str(self.state.label_true)],
                self.state.load_hp, self.state.rpm, len(self.state.windows),
                self.state.compute_ms, self.state.bundle.version,
            )
        )

    def _resync_bars(self) -> None:
        for bar, _value in self.prob_bars:
            bar.refresh()


class AcquirePage(ScrollPage):
    def __init__(self, state: State, on_done, **kwargs):
        super().__init__(name="acquire", **kwargs)
        self.state = state
        self.on_done = on_done

        card = Card().auto()
        card.add_widget(label("演示状态", size=12, color=MUTED, size_hint_y=None, height=dp(18)))
        # 四类故障排成两行两列的大按钮，避免一行挤四个把文字压扁
        grid = GridLayout(cols=2, size_hint_y=None, spacing=dp(8))
        grid.bind(minimum_height=grid.setter("height"))
        self.class_chips = []
        for index, name in enumerate(("正常", "内圈故障", "外圈故障", "滚动体故障")):
            chip = Chip(name, active=index == self.state.label_true,
                        size_hint=(1, None), height=dp(46), font_size=sp(14))
            chip.bind(on_release=lambda _inst, i=index: self._pick_class(i))
            self.class_chips.append(chip)
            grid.add_widget(chip)
        card.add_widget(grid)

        card.add_widget(label("负载", size=12, color=MUTED, size_hint_y=None, height=dp(18)))
        load_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.load_chips = []
        for load in (0.0, 1.0, 2.0, 3.0):
            chip = Chip("%g hp" % load, active=abs(load - self.state.load_hp) < 1e-6,
                        size_hint=(1, None), height=dp(44))
            chip.bind(on_release=lambda _inst, value=load: self._pick_load(value))
            self.load_chips.append((load, chip))
            load_row.add_widget(chip)
        card.add_widget(load_row)
        self.add(card)

        rpm_card = Card().auto()
        rpm_card.add_widget(label("自定义转速（r/min）", size=12, color=MUTED,
                                   size_hint_y=None, height=dp(18)))
        rpm_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.rpm_input = TextInput(
            text="%.6g" % self.state.rpm, multiline=False, input_type="number",
            input_filter="float", halign="center", font_name=FONT_NAME,
            font_size=sp(15), padding=(dp(8), dp(10)),
            background_normal="", background_active="", background_color=col(BG),
            foreground_color=col(TEXT), cursor_color=col(TEAL),
            hint_text="如 1772", hint_text_color=col(MUTED),
        )
        self.rpm_input.bind(on_text_validate=lambda *_: self._apply_rpm())
        rpm_row.add_widget(self.rpm_input)
        apply_button = Button(
            text="应用", font_name=FONT_NAME, font_size=sp(14),
            background_normal="", background_down="", background_color=col(TEAL),
            color=col("#FFFFFF"), size_hint_x=None, width=dp(76),
        )
        apply_button.bind(on_release=lambda *_: self._apply_rpm())
        rpm_row.add_widget(apply_button)
        rpm_card.add_widget(rpm_row)
        self.rpm_note = label("", size=11, color=MUTED, auto_height=True)
        rpm_card.add_widget(self.rpm_note)
        self.add(rpm_card)
        self._update_rpm_note()

        run_card = Card().auto()
        row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        self.start_button = Button(
            text="开始采集（回放全部窗口）", font_name=FONT_NAME, font_size=sp(13),
            background_normal="", background_down="", background_color=col(TEAL),
            color=col("#FFFFFF"),
        )
        self.start_button.bind(on_release=lambda *_: self.start())
        row.add_widget(self.start_button)
        run_card.add_widget(row)
        self.progress = Bar(0.0, TEAL, size_hint_y=None, height=dp(8))
        run_card.add_widget(self.progress)
        self.progress_text = label("等待开始", size=11, color=MUTED, size_hint_y=None, height=dp(18))
        run_card.add_widget(self.progress_text)
        self.add(run_card)

        wave_card = Card().auto()
        wave_card.add_widget(label("实时波形（1 s 窗口，抽样显示）", size=12, color=MUTED,
                                   size_hint_y=None, height=dp(18)))
        self.chart = MiniChart(TEAL, size_hint_y=None, height=dp(128))
        wave_card.add_widget(self.chart)
        self.add(wave_card)

        self._event = None
        self._step = 0

    # -- 交互 ------------------------------------------------------------ #
    def _pick_class(self, index: int) -> None:
        for i, chip in enumerate(self.class_chips):
            chip.set_active(i == index)
        self.state.load_sample(index, self.state.load_hp)
        self._recompute()

    def _pick_load(self, load: float) -> None:
        for value, chip in self.load_chips:
            chip.set_active(abs(value - load) < 1e-6)
        self.state.load_sample(self.state.label_true, load)
        self._recompute()

    def _apply_rpm(self) -> None:
        text = (self.rpm_input.text or "").strip()
        try:
            value = float(text)
        except ValueError:
            self.rpm_note.color = col(DANGER)
            self.rpm_note.text = "请输入数字，例如 1772"
            return
        if not 300.0 <= value <= 6000.0:
            self.rpm_note.color = col(DANGER)
            self.rpm_note.text = "转速需在 300～6000 r/min 之间"
            return
        self.state.set_rpm(value)
        self.state.analyze()
        self._update_rpm_note()
        self.on_done()

    def _update_rpm_note(self) -> None:
        self.rpm_note.color = col(MUTED)
        analysis = self.state.analysis
        if not analysis or not analysis.get("rows"):
            self.rpm_note.text = "当前 %.0f r/min（默认值随演示状态与负载变化）" % self.state.rpm
            return
        freqs = analysis["rows"][-1]["fault_frequencies"]
        self.rpm_note.text = (
            "当前 %.0f r/min ｜ BPFO %.1f ｜ BPFI %.1f ｜ BSF %.1f Hz"
            % (self.state.rpm, freqs["BPFO"], freqs["BPFI"], freqs["BSF"])
        )

    def _recompute(self) -> None:
        self.state.analyze()
        self.state.push_history()
        self.progress.set_value(1.0, TEAL)
        self.progress_text.text = "已用 %d 个窗口完成分析，耗时 %.0f ms" % (
            len(self.state.windows), self.state.compute_ms
        )
        self._draw_window(0)
        self.on_done()

    def start(self) -> None:
        if self._event:
            self._event.cancel()
        total = max(1, len(self.state.windows))
        self._step = 0
        self.progress.set_value(0.0, TEAL)
        self.progress_text.text = "采集中…"
        self._event = Clock.schedule_interval(lambda _dt: self._tick(total), 0.12)

    def _tick(self, total: int) -> bool:
        self._step += 1
        done = min(total, self._step)
        self.progress.set_value(done / total, TEAL)
        self.progress_text.text = "已读取 %d / %d 个窗口" % (done, total)
        self._draw_window(done - 1)
        if done >= total:
            if self._event:
                self._event.cancel()
                self._event = None
            self._recompute()
            return False
        return True

    def _draw_window(self, index: int) -> None:
        if self.state.windows is None or len(self.state.windows) == 0:
            return
        window = self.state.windows[min(max(index, 0), len(self.state.windows) - 1)]
        step = max(1, len(window) // 600)
        self.chart.set_data(range(0, len(window), step), window[::step])

    def sync_selection(self) -> None:
        for i, chip in enumerate(self.class_chips):
            chip.set_active(i == self.state.label_true)
        for value, chip in self.load_chips:
            chip.set_active(abs(value - self.state.load_hp) < 1e-6)
        self.rpm_input.text = "%.6g" % self.state.rpm
        self._update_rpm_note()


class AnalyzePage(ScrollPage):
    def __init__(self, state: State, **kwargs):
        super().__init__(name="analyze", **kwargs)
        self.state = state

        wave_card = Card().auto()
        wave_card.add_widget(label("时域波形（滤波后）", size=12, color=MUTED,
                                   size_hint_y=None, height=dp(18)))
        self.wave_chart = MiniChart(TEAL, size_hint_y=None, height=dp(124))
        wave_card.add_widget(self.wave_chart)
        self.add(wave_card)

        spec_card = Card().auto()
        spec_card.add_widget(label("频谱（Welch PSD）", size=12, color=MUTED,
                                   size_hint_y=None, height=dp(18)))
        self.spec_chart = MiniChart(INFO, size_hint_y=None, height=dp(124))
        spec_card.add_widget(self.spec_chart)
        self.add(spec_card)

        env_card = Card().auto()
        env_card.add_widget(label("包络谱（500–5500 Hz 解调）", size=12, color=MUTED,
                                  size_hint_y=None, height=dp(18)))
        self.env_chart = MiniChart(WARN, size_hint_y=None, height=dp(124))
        env_card.add_widget(self.env_chart)
        self.add(env_card)

        table = Card().auto()
        table.add_widget(label("候选特征（9 项）", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.feature_rows = []
        labels = [
            ("rms", "RMS"), ("kurtosis", "峭度"), ("crest_factor", "峰值因子"),
            ("impulse_factor", "脉冲因子"), ("high_band_ratio", "高频带能量比"),
            ("spectral_centroid", "谱质心"), ("spectral_entropy", "谱熵"),
            ("fault_frequency_ratio", "故障频率能量比"), ("harmonic_consistency", "故障谐波一致性"),
        ]
        for key, name in labels:
            row = BoxLayout(size_hint_y=None, height=dp(28))
            row.add_widget(label(name, size=11.5, color=MUTED))
            value = label("—", size=12, bold=True, halign="right")
            value.bind(size=lambda inst, _v: setattr(inst, "text_size", (inst.width, None)))
            row.add_widget(value)
            self.feature_rows.append((key, value))
            table.add_widget(row)
        self.add(table)

        freq_card = Card().auto()
        freq_card.add_widget(label("特征频率", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.freq_label = label("—", size=11.5, color=MUTED, auto_height=True)
        freq_card.add_widget(self.freq_label)
        self.add(freq_card)

    def update(self) -> None:
        analysis = self.state.analysis
        if not analysis:
            return
        latest = analysis["rows"][-1]
        spectra = latest["spectra"]
        filtered = spectra["filtered"]
        step = max(1, len(filtered) // 600)
        self.wave_chart.set_data(range(0, len(filtered), step), filtered[::step])
        freqs, psd = spectra["freqs"], spectra["psd"]
        if len(freqs) > 2:
            self.spec_chart.set_data(freqs, psd)
        env_f, env_p = spectra["env_freqs"], spectra["env_psd"]
        if len(env_f) > 2:
            self.env_chart.set_data(env_f, env_p)
        for key, value in self.feature_rows:
            value.text = fmt(latest["features"][key], 4)
        frequencies = latest["fault_frequencies"]
        self.freq_label.text = "BPFI %.1f Hz ｜ BPFO %.1f Hz ｜ BSF %.1f Hz ｜ 主轴 %.2f Hz" % (
            frequencies["BPFI"], frequencies["BPFO"], frequencies["BSF"], frequencies["shaft"]
        )


class MonitorPage(ScrollPage):
    def __init__(self, state: State, **kwargs):
        super().__init__(name="monitor", **kwargs)
        self.state = state

        card = Card().auto()
        card.add_widget(label("状态指标（对数偏移 vs 3σ 阈值）", size=13, bold=True,
                              size_hint_y=None, height=dp(22)))
        self.indicator_rows = []
        for key, title in (("I1_RMS", "振动总能量"), ("I2_FAULT", "故障能量"), ("I3_KURTOSIS", "冲击性")):
            row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(6))
            column = BoxLayout(orientation="vertical")
            column.add_widget(label(title, size=11.5))
            value = label("—", size=10.5, color=MUTED)
            column.add_widget(value)
            row.add_widget(column)
            bar_column = BoxLayout(orientation="vertical", spacing=dp(2))
            bar = Bar(0.0, TEAL, size_hint_y=None, height=dp(8))
            bar_column.add_widget(bar)
            limit = label("阈值 —", size=10, color=MUTED)
            bar_column.add_widget(limit)
            row.add_widget(bar_column)
            self.indicator_rows.append((key, bar, value, limit))
            card.add_widget(row)
        self.add(card)

        health_card = Card().auto()
        health_card.add_widget(label("健康评估", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.health_value = label("—", size=30, bold=True, color=TEAL, size_hint_y=None, height=dp(46))
        self.health_grade = label("—", size=11.5, color=MUTED, size_hint_y=None, height=dp(18))
        health_card.add_widget(self.health_value)
        health_card.add_widget(self.health_grade)
        self.health_bar = Bar(0.0, TEAL, size_hint_y=None, height=dp(8))
        health_card.add_widget(self.health_bar)
        self.add(health_card)

        trend_card = Card().auto()
        trend_card.add_widget(label("历史健康趋势（最近 20 次）", size=13, bold=True,
                                    size_hint_y=None, height=dp(22)))
        self.trend_chart = MiniChart(INFO, size_hint_y=None, height=dp(120))
        trend_card.add_widget(self.trend_chart)
        self.trend_note = label("—", size=10.5, color=MUTED, auto_height=True)
        trend_card.add_widget(self.trend_note)
        self.add(trend_card)

    def update(self) -> None:
        analysis = self.state.analysis
        if not analysis:
            return
        latest = analysis["rows"][-1]
        thresholds = analysis["thresholds"]
        for key, bar, value, limit in self.indicator_rows:
            current = float(latest["indicators"][key])
            threshold = float(thresholds.get(key, 0.0))
            exceeded = current > threshold
            ratio = 0.0 if threshold <= 0 else max(0.0, min(1.0, current / threshold))
            bar.set_value(ratio, DANGER if exceeded else TEAL)
            value.text = "当前 %+.3f" % current
            limit.text = "阈值 %+.3f %s" % (threshold, "· 超限" if exceeded else "")

        health = analysis["health"]
        self.health_value.text = fmt(health["health_index"], 1)
        grade_color = OK if health["health_index"] >= 85 else WARN if health["health_index"] >= 65 else DANGER
        self.health_value.color = col(grade_color)
        self.health_grade.text = "%s ｜ 异常度 %.3f ｜ 状态 %s" % (
            health["health_grade"], health["abnormality"], analysis["state"]
        )
        self.health_bar.set_value(max(0.0, min(1.0, health["health_index"] / 100.0)), grade_color)

        values = [float(item["health"]) for item in self.state.history]
        if values:
            self.trend_chart.set_data(range(len(values)), values)
            self.trend_note.text = "最近一次 %s（%s）：健康 %.1f，%s" % (
                self.state.history[-1]["time"], self.state.history[-1]["class_name"],
                self.state.history[-1]["health"], self.state.history[-1]["state"],
            )
        else:
            self.trend_note.text = "暂无历史记录"


class AboutPage(ScrollPage):
    def __init__(self, state: State, **kwargs):
        super().__init__(name="about", **kwargs)
        self.state = state

        intro = Card().auto()
        intro.add_widget(label("智能诊断 · 轴承故障诊断", size=18, bold=True, color=TEAL,
                               size_hint_y=None, height=dp(30)))
        intro.add_widget(label(
            "本 App 是课程设计《电机-轴承-转子智能运维监测系统》的移动端实现，"
            "把网页版的采集、特征、诊断、监测、健康与决策整条链路搬到手机上，"
            "所有计算在本地完成，无需联网。",
            size=11.5, color=MUTED, auto_height=True,
        ))
        self.add(intro)

        model_card = Card().auto()
        model_card.add_widget(label("模型与数据", size=13, bold=True, size_hint_y=None, height=dp(22)))
        self.model_label = label("—", size=11.5, color=MUTED, auto_height=True)
        model_card.add_widget(self.model_label)
        self.metrics_label = label("—", size=11.5, color=MUTED, auto_height=True)
        model_card.add_widget(self.metrics_label)
        self.add(model_card)

        method = Card().auto()
        method.add_widget(label("方法链路", size=13, bold=True, size_hint_y=None, height=dp(22)))
        method.add_widget(label(
            "1. 分段：1 s 窗口、50% 重叠；\n"
            "2. 预处理：去线性趋势 → 20 Hz 高通 → 5.4 kHz 低通（零相位）；\n"
            "3. 特征：RMS、峭度、峰值因子、脉冲因子、高频带能量比、谱质心、谱熵、"
            "故障频率能量比、故障谐波一致性；\n"
            "4. 诊断：300 棵随机森林，取窗口概率均值；\n"
            "5. 监测：3 个指标的对数偏移与分负载 3σ 阈值，连续 3 窗确认报警；\n"
            "6. 健康：分负载中位数 + 1.5σ 死区 + 斜坡，映射到 0–100；\n"
            "7. 决策：D0–D5 规则链给出处置建议。",
            size=11.5, color=MUTED, auto_height=True,
        ))
        self.add(method)

        tips = Card().auto()
        tips.add_widget(label("说明", size=13, bold=True, size_hint_y=None, height=dp(22)))
        tips.add_widget(label(
            "演示数据取自西储大学（CWRU）12k 驱动端实测记录，"
            "0/1/2 hp 参与训练或验证，3 hp 为独立测试集。"
            "界面上的“真值”标注来自记录本身，可用于对照模型是否有误判。",
            size=11.5, color=MUTED, auto_height=True,
        ))
        self.add(tips)

        self.version_label = label("", size=10.5, color=MUTED, halign="center",
                                   size_hint_y=None, height=dp(30))
        self.add(self.version_label)

    def update(self) -> None:
        params = self.state.bundle.params
        metrics = params.get("metrics", {})
        self.model_label.text = "版本 %s\n训练 %d 窗 ｜ 测试 %d 窗 ｜ 特征 %d 项 ｜ %d 棵树" % (
            params.get("version", "—"),
            int(metrics.get("train_size", 0)), int(metrics.get("eval_size", 0)),
            len(params["config"]["features"]["selected_features"]),
            int(metrics.get("trees", 0)),
        )
        self.metrics_label.text = "独立测试集：Accuracy %.2f%% ｜ Macro-F1 %.4f" % (
            float(metrics.get("accuracy", 0.0)) * 100, float(metrics.get("macro_f1", 0.0))
        )
        self.version_label.text = "智能诊断 C_V1.0 ｜ 数据源 CWRU 12k DE ｜ 完全离线运行"


# --------------------------------------------------------------------------- #
# 主程序
# --------------------------------------------------------------------------- #
class BearingApp(App):
    title = "智能诊断"

    def build(self):
        Window.clearcolor = col(BG)
        try:
            Window.softinput_mode = "below_target"
        except Exception:  # pragma: no cover
            pass
        self.state = State()
        self.state.load_sample(2, 0.0)
        self.state.analyze()

        root = BoxLayout(orientation="vertical")

        header = BoxLayout(
            orientation="vertical", size_hint_y=None, height=dp(92),
            padding=(dp(16), dp(12), dp(16), dp(8)), spacing=dp(2),
        )
        with header.canvas.before:
            Color(*col(TEAL))
            self._header_rect = RoundedRectangle(pos=header.pos, size=header.size,
                                                 radius=[0, 0, dp(18), dp(18)])
        header.bind(
            pos=lambda inst, _v: setattr(self._header_rect, "pos", inst.pos),
            size=lambda inst, _v: setattr(self._header_rect, "size", inst.size),
        )
        header.add_widget(label("智能诊断", size=20, bold=True, color="#FFFFFF"))
        self.header_sub = label("轴承故障诊断 · 移动端", size=11, color="#CDE8E8")
        header.add_widget(self.header_sub)
        root.add_widget(header)

        self.manager = ScreenManager()
        self.pages = {
            "overview": OverviewPage(self.state),
            "acquire": AcquirePage(self.state, self.refresh_all),
            "analyze": AnalyzePage(self.state),
            "monitor": MonitorPage(self.state),
            "about": AboutPage(self.state),
        }
        for page in self.pages.values():
            self.manager.add_widget(page)
        root.add_widget(self.manager)

        nav = BoxLayout(size_hint_y=None, height=dp(56), spacing=0)
        with nav.canvas.before:
            Color(*col("#FFFFFF"))
            self._nav_rect = RoundedRectangle(pos=nav.pos, size=nav.size)
            Color(*col(LINE))
            self._nav_line = Line(points=[0, 0, 0, 0], width=1)
        nav.bind(pos=self._sync_nav, size=self._sync_nav)
        self.nav_buttons = {}
        for key, text in (
            ("overview", "总览"), ("acquire", "采集"), ("analyze", "分析"),
            ("monitor", "监测"), ("about", "关于"),
        ):
            button = NavButton(text)
            button.bind(on_release=lambda _inst, name=key: self.switch(name))
            self.nav_buttons[key] = button
            nav.add_widget(button)
        root.add_widget(nav)

        self.switch("overview")
        self.refresh_all()
        return root

    def _sync_nav(self, inst, _value):
        self._nav_rect.pos = inst.pos
        self._nav_rect.size = inst.size
        self._nav_line.points = [inst.x, inst.top, inst.right, inst.top]

    def switch(self, name: str) -> None:
        self.manager.current = name
        for key, button in self.nav_buttons.items():
            button.set_active(key == name)

    def refresh_all(self) -> None:
        for page in self.pages.values():
            try:
                page.update()
            except AttributeError:
                pass
        if hasattr(self.pages["acquire"], "sync_selection"):
            self.pages["acquire"].sync_selection()
        analysis = self.state.analysis
        if analysis:
            self.header_sub.text = "当前 %s ｜ %g hp ｜ %.0f r/min" % (
                analysis["class_name"], self.state.load_hp, self.state.rpm
            )


if __name__ == "__main__":
    BearingApp().run()
