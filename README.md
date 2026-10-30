# 智能诊断 · 轴承故障诊断（Android APK 版）

把课程设计里的**网页版轴承故障诊断系统**完整搬到手机上：采集回放、预处理、9 项特征、
随机森林四分类、状态监测、健康评分、运维决策、历史记录，全部在手机本地计算，**断网可用**。

> 与网页版的关系：数据、模型结构、判据规则完全一致；区别只在工程实现——
> 网页版用 scipy + sklearn，手机版用**纯 numpy** 复刻同样的信号处理与树推理，
> 因此不需要在 Android 上交叉编译 scipy / sklearn。

---

## 1. 界面

底部五个标签：

| 页面 | 内容 |
|---|---|
| **总览** | 诊断结论 + 置信度、四类概率条、健康指数、监测状态、数据质量、运维决策、当前样本信息 |
| **采集** | 选择「状态 × 负载」四个 chip，一键回放采集（按窗口推进的进度条 + 实时波形） |
| **分析** | 滤波后时域波形、Welch 频谱、包络谱三张图 + 9 项候选特征 + 特征频率 |
| **监测** | 3 个状态指标（当前值 / 3σ 阈值 / 超限标记）、健康评分、历史健康趋势 |
| **关于** | 模型版本、训练与测试规模、方法链路、数据来源说明 |

配色沿用医疗科技风（主色 `#0B6E6E`）。**应用名：智能诊断**；**图标**为深青底 + 白色频谱柱
（`assets/icon.png`，512×512，由 `tools/make_icon.py` 生成，可随时重画）。

## 2. 目录结构

```
APK地址/
├── main.py                     Kivy 应用（5 个页面 + 自定义卡片/图表控件）
├── bdapp/
│   ├── dsp.py                  纯 numpy 信号处理与 9 项特征
│   ├── model.py                随机森林树遍历、监测状态机、健康、决策
│   └── assets.py               资源加载（模型/参数/滤波核/演示数据）
├── assets/
│   ├── icon.png                应用图标 512×512
│   ├── models/
│   │   ├── rf_model.npz        300 棵树扁平化后的节点数组
│   │   ├── params.json         基线、健康参数、阈值、配置、指标
│   │   ├── filters.npz         3 个零相位滤波核（高通/低通/带通）
│   │   └── parity.json         手机端模型精度 + 与 sklearn 的一致性
│   ├── data/demo.npz           16 条 CWRU 演示记录（4 s/条）
│   └── fonts/NotoSansSC-Regular.ttf
├── tools/
│   ├── export_assets.py        PC 端导出/重训脚本（需要 scipy/sklearn）
│   ├── make_icon.py            重新生成图标
│   ├── check_assets.py         资源自检（不依赖 Kivy）
│   └── check_repo.py           上传前自检（防呆，只用标准库）
├── buildozer.spec
├── requirements.txt
├── constraints.txt
├── .github/workflows/build-apk.yml
└── README.md
```

## 3. 实测指标

资源由 `tools/export_assets.py` 生成，结果写在 `assets/models/parity.json`：

| 项目 | 数值 |
|---|---|
| 手机端模型测试集（3 hp，训练未见过的负载） | **Accuracy 1.0000 / Macro-F1 1.0000** |
| 16 条演示记录（4 类 × 4 负载）整体判对率 | **16 / 16** |
| 树遍历 vs sklearn `predict_proba` 最大偏差 | 1.4e-10 |
| 单条记录（7 窗）分析耗时 | 约 40 ms（PC 上实测） |

## 4. 为什么要导出、而不是直接用网页版模型

网页端特征由 `scipy.signal.sosfiltfilt / welch / hilbert` 计算。Android 上安装 scipy 需要
交叉编译 Fortran/BLAS，失败率高、包体大。手机端因此：

1. 把三个滤波器的**等效零相位冲激响应**离线算好（`filters.npz`），运行时只做 FFT 卷积；
2. Welch、Hilbert、峭度、包络按 scipy 的公式用 numpy 逐条复刻；
3. 由于两端在窗口**头尾约 20 个采样点**的边界处理不同（峰值因子、峭度对此敏感），
   手机端**用同一套 numpy 特征重新训练一个同协议模型**（同样 272 训练窗 / 136 测试窗、
   同样按负载划分），而不是硬套网页端权重。

所以两端的模型版本号不同（手机端由导出脚本重新生成），这是有意为之。

## 5. 重新生成资源 / 图标

```powershell
cd APK地址
python tools\export_assets.py     # 重训并导出模型、基线、演示数据（需要 scipy/sklearn）
python tools\make_icon.py         # 重新画图标（需要 pillow）
python tools\check_assets.py      # 不需要窗口，验证 16 条演示记录是否全部判对
```

`--project` 可指定网页端项目路径（默认指向本机的 `bearing_diagnosis_system`）。

## 6. 本地运行

```powershell
pip install -r requirements.txt      # kivy + numpy + pillow
python main.py                       # 桌面窗口预览（与手机同一套 UI）
```

## 7. 上传 GitHub 的目录结构（防呆）

### 7.1 仓库根目录必须长这样

```
<仓库根>/                      ← 直接就是 APK地址 里的内容，不要再套一层文件夹
├── main.py                    ✅ 必须
├── buildozer.spec             ✅ 必须（工作流靠它编译）
├── requirements.txt           ✅ 必须
├── constraints.txt            ✅ 必须（工作流用 PIP_CONSTRAINT 引用）
├── README.md                  ✅ 建议
├── .gitignore                 ✅ 必须（挡住构建产物）
├── .github/workflows/build-apk.yml   ✅ 必须（否则 Actions 里没有可运行的工作流）
├── bdapp/                     ✅ 必须（4 个 .py 全要）
├── assets/icon.png            ✅ 必须（缺了会用默认绿色机器人图标）
├── assets/models/*.npz/json   ✅ 必须（缺了打开就崩）
├── assets/data/demo.npz       ✅ 必须（1.7 MB，演示数据）
├── assets/fonts/*.ttf         ✅ 必须（否则中文显示成方框）
├── tools/                     ✅ 建议（不影响打包，但方便以后重训/排查）
└── ...
```

### 7.2 三步上传

```powershell
cd "F:\Hurricane\课程设计-故障诊断\APK地址"
python tools\check_repo.py            # ① 先自检（见 7.4）
git init
git add .
git commit -m "智能诊断 APK v1.0"
git branch -M main
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git push -u origin main               # ② 推上去，Actions 自动开始构建
```

③ 打开仓库 **Actions → 构建 APK → 最新运行 → Artifacts**，下载
`zhinengzhenduan-bearing-diagnosis-apk`，解压得到 `bin/*.apk`，传到手机安装。

### 7.3 绝对不要提交

| 不要提交 | 原因 |
|---|---|
| `bin/`、`*.apk` | 构建产物，会把仓库撑爆 |
| `.buildozer/`、`build/`、`dist/` | 本机缓存与中间产物，几个 GB |
| `__pycache__/`、`*.pyc` | 字节码，跨 Python 版本会冲突 |
| `.kivy/` | 本机 Kivy 缓存 |
| 网页端整个工程、`.mat` 原始数据 | 手机端只需要导出的 npz，原始数据 60 MB 没必要 |

`.gitignore` 已经把这些都写进去了；**前提是你在 `APK地址` 目录里执行 `git init`**，
不要在它的父目录初始化（否则会把 `bearing_diagnosis_system`、`神经网络方法` 一起推上去）。

### 7.4 防呆自检（本地与云端都会跑）

```powershell
python tools\check_repo.py
```

它会检查并给出结论：

- 14 个必需文件是否齐全、是否为空文件；
- `buildozer.spec` 里是否配了 `icon.filename`、`source.include_exts` 是否含 **npz**、
  `requirements` 是否只有 `python3,kivy,numpy`（**不含 scipy/sklearn**）；
- `assets/icon.png` 是否为 512×512 的合法 PNG；
- `main.py` 的 `title` 与 `buildozer.spec` 的 `title` 是否一致（防止改了名字只改一处）；
- 是否混入了 `bin/`、`.buildozer/`、`__pycache__` 等不该提交的目录；
- 打印**待提交文件清单**与总体积（当前约 18.8 MB，其中字体 17 MB）。

GitHub Actions 的第 1.5 步会执行同一个脚本，**不通过就直接失败**，不会白跑 1 小时才发现缺文件。

### 7.5 最常见的四个翻车点

| 现象 | 原因 | 处理 |
|---|---|---|
| Actions 里没有工作流 | `.github/workflows/` 没提交，或文件名不是 `.yml` | 确认该路径下有 `build-apk.yml` |
| APK 能装但一打开就崩 | `assets/models/*.npz` 没进仓库，或 `include_exts` 漏了 `npz` | 跑 `check_repo.py` |
| 中文全变方框 | 字体文件没提交（`assets/fonts/`） | 补上 17 MB 的 ttf |
| 桌面图标还是绿色机器人 | `icon.filename` 未配置或 `assets/icon.png` 缺失 | 跑 `python tools/make_icon.py` |

## 8. 已知限制

1. **只做演示数据回放**，没有做手机文件选择导入（需要 plyer/SAF 权限，属后续扩展）；
2. 演示记录为每条 4 s（7 个窗口），比网页版的 12 s 少，置信度会略低但仍稳定；
3. 手机端模型与网页端模型不是同一份权重（原因见第 4 节），两侧数值不会逐位一致；
4. 单条记录分析约 40 ms，但首次启动加载 `demo.npz`（1.7 MB）会有约 1 s 的读盘时间。

## 9. 说明

本目录为新建工程，**未改动**桌面上的蓝牙 App（`ble-assistant`）任何文件，
只借鉴了它的构建脚本结构与中文字体资源。
