# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

双目立体视觉测距系统（Windows + USB 双目摄像头，1280x480 左右拼接画面）。核心工作流是"空气 → 空气+平板玻璃 → 水下"三阶段渐进标定，最终在下水状态下实时测量深度（单位 cm）。

无构建系统、无测试框架、无 lint 配置。所有脚本直接 `python <脚本名>.py` 运行，运行前请阅读脚本头部注释中的"先运行"依赖说明。

依赖：`opencv-contrib-python`（WLS 视差滤波需要 contrib 版的 `cv2.ximgproc`，装普通 opencv-python 会自动退回原始 SGBM）、`numpy`、`scipy`。

## 标定流水线（文件间的数据流）

```
matlab.mat (MATLAB stereoParameters)
  → transposition.m（MATLAB 侧运行，无损导出，注意转置 MATLAB IntrinsicMatrix/RotationOfCamera2）
  → stereo_params_plain.mat（Python 唯一的空气参数来源，禁止从 txt 复制参数）
  → 立体校正.py（生成空气校正参数 stereo_params_updated.npz）
  → rect.py（空气+玻璃标定 → stereo_params_updated_with_glass.npz）
  → rect_water.py（水下标定 → stereo_params_updated_underwater.npz）
```

测深入口（都复用 `stereo_depth_core.py` 的 `StereoDepthRunner`）：
- `测量深度.py`：空气测距
- `测量深度_玻璃.py`：加玻璃测距（玻璃须在标定时的同一位置）
- `measure_depth_underwater.py`：水下测距，**不重新标定**；默认读根目录水下 npz 最新版，也可命令行指定历史 run 时间戳或任意 npz 路径（`--list` 列出可选 run）；折射深度修正由 npz 的 `depth_scale`/`depth_offset_cm` 驱动；s 键验证采样样本存根目录 `depth_samples/underwater_<时间戳>/`；d 键采集数据 → 同会话 `sampledata/{RGB,Depth,Depth_color}/`
- `collect_depth_data.py`：下水数据采集入口（d 键整帧采集，s 键验证采样在此入口禁用）→ 根目录 `sampledata/underwater_<时间戳>/{RGB,Depth,Depth_color}/` 三件套，同一编号对应同一帧（RGB=对齐校正左图，Depth=**原始深度图 uint16 PNG，数值=深度毫米即 cm×10，无效像素 0，已折射修正**，Depth_color=彩色深度图）；参数解析复用 measure_depth_underwater.py（run 时间戳/npz 路径/`--list`）；d 键采集逻辑为 rect_water.py 的 `SampleDataStereoDepthRunner`（两入口共用，经 `on_key` 扩展点接入）
- `基于WSL的立体匹配.py`：同一核心，只是视差窗口名不同
- `rect.py` / `rect_water.py` 末尾在 `RUN_DEPTH_AFTER_CALIB=True` 时会直接进入测深

辅助/实验脚本：
- `采集数据.py`：采集左右图对到 `left/`、`right/`、`pair/`（已 gitignore）
- `1.py`：论文"三步法"水下版本（焦距 ×1.33 初值、允许重估焦距）——与 `rect_water.py` 思路不同，属实验性
- `2.py`：Snell 折射 + 光线追踪的物理建模实验
- `test.py`：测距诊断脚本（注意其 `PARAM_FILE` 仍是旧机器的硬编码绝对路径，如需使用先改路径）

## 两个主入口脚本（重点）

### rect.py — 空气+玻璃标定
1. 读 `stereo_params_plain.mat` 拿空气 K/dist/R/T；
2. 实时采集（按 `s` 缓存并落盘原图，`q` 结束后统一检测角点，避免采集卡顿）；
3. **固定焦距 fx/fy 和光心 cx/cy**，只重估玻璃引起的左右畸变 dist（5 参数 [k1,k2,p1,p2,k3]）；
4. `RECALIBRATE_STEREO_RT=True` 时用 `CALIB_FIX_INTRINSIC` 重估双目 R/T；
5. `stereoRectify` 生成 map1x/map1y/map2x/map2y/Q，保存 `stereo_params_updated_with_glass.npz`（同时存 `swap_lr`、`swap_capture_halves` 等一致性标记）；
6. 可选直接启动测深。

### rect_water.py — 水下标定
与 rect.py 同构，差异：初值来自玻璃 npz（无 MATLAB 单位转换与 swap_lr 逻辑）；CLAHE clipLimit=2.5；`RECTIFY_ALPHA=0`（裁黑边求视差稳定）；标定迭代数更多；**不做 1.33 经验缩放**（MLE 直接重估水下 dist 再重估 R/T）。采集/归档模式与 rect.py 统一（见下方规则）。

### 标定历史归档（两脚本共用机制）

每次运行把本次标定的全部产物写入 `calibration_history/<glass|underwater>/<启动时间戳>/`：`images/{left,right,pair}/`（原图，gitignore）、角点 npz（含 `left_files`/`right_files` 溯源键）、参数 npz 快照、`calibration_report.md`（配置快照、采集统计、各步 RMS、最终参数）。**只增不覆盖**。根目录下的 npz 文件保持"最新版"语义继续覆盖，测深入口按固定路径读它们；回滚到某次历史标定 = 把该 run 目录下的参数 npz 复制回根目录（或改 `PARAM_FILE`）。

## 硬性约定（改动前必须理解）

1. **单位链**：`SQUARE_SIZE=1.0` 表示 cm；角点世界坐标为 cm，因此标定出的 T 和 Q 均为 cm。MATLAB 空气 T 单位由 `MATLAB_T_UNIT`/`TRANSLATION_INPUT_UNIT`（当前 "mm"）转成 cm。Q 已经是 cm，**测距代码中不得再乘任何 UNIT_SCALE**。
2. **SWAP_CAPTURE_HALVES=True**：当前摄像头 1280x480 拼接图的物理左右半边是反的，切半图后需交换。此开关在 采集数据.py / 立体校正流程 / rect.py / rect_water.py / 各测深入口 及 npz 中的 `swap_capture_halves` 键之间必须保持一致，否则画面错位、测距错误。
3. **SWAP_LR ≠ SWAP_CAPTURE_HALVES**：`SWAP_LR`（立体校正.py）交换的是 MATLAB 标定的 Camera1/Camera2 顺序（R_new=R.T, T_new=-R.T@T，K/dist 同步交换）；`rect.py` 的 `SWAP_AIR_PARAMS_LR=None` 会从空气 npz 自动读取 `swap_lr`，二者不要手动绑定。
4. **固定 K 的标定策略**：玻璃/水下阶段只重估畸变（`CALIB_FIX_FOCAL_LENGTH | CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_K4/K5/K6`），焦距光心继承上一阶段。这是项目核心方法论，勿随意放开。
5. **npz 契约**：测深核心要求 npz 必含 `map1x/map1y/map2x/map2y/Q/image_size`，缺一即抛 KeyError。新增字段可以，删字段不行。
6. **相机**：`cv2.CAP_DSHOW`（Windows 专用），单目尺寸 640x480（`IMG_SIZE=(640,480)`，OpenCV 宽高序）。当前机器双目摄像头枚举为 `CAMERA_ID=1`（rect.py / rect_water.py / 测量深度_水下.py）；`测量深度.py` / `测量深度_玻璃.py` / `基于WSL的立体匹配.py` 仍硬编码 0，换入口运行前先核对。
7. **棋盘格**：`PATTERN_SIZE=(8,11)` 是**内角点数**（横向×纵向），不是格子数；检测时先 `findChessboardCornersSB`（EXHAUSTIVE|ACCURACY），失败用传统方法+`cornerSubPix` 兜底；灰度先过 CLAHE 增强对抗玻璃反光/水下低对比；**检测后必须过 `canonicalize_corner_order` 统一角点链方向**（8×11 网格 180° 旋转对称，左右视图起点不一致会破坏双目对应、拖垮基线）。
8. **中文路径**：脚本和图像目录含中文，写图必须走 `imwrite_unicode`（`cv2.imencode` + `tofile`），不要直接 `cv2.imwrite`。

## 后续规则记录（持续追加）

- 2026-09-07：初始化仓库（本地 git 管理，无远程）。`left/`、`right/`、`pair/`、`underwater_calibration_images/` 图像数据不入库；标定参数 npz/mat 和角点 npz 保留入库以便回滚到已验证的标定结果。
- 2026-09-08：rect.py 与 rect_water.py 的采集/归档模式已统一：采集阶段只缓存+落盘原图（不做实时角点检测），`q` 后统一批量检测、左右都成功才计入标定；原图均按程序启动时间戳归档到 `<glass|underwater>_calibration_images/<时间戳>/{left,right,pair}/`，目录不入库。
- 2026-09-08：**标定历史归档机制**：每次标定完整产物（原图/角点/参数/报告）写入 `calibration_history/<stage>/<时间戳>/`，只增不覆盖（起因：水下标定曾被一张角点 180° 翻转的坏图污染且无法溯源到具体图片）。根目录 npz 仍是"最新版"语义；角点 npz 新增 `left_files`/`right_files` 键关联原图文件名；`calibration_history/**/images/` 不入库，其余入库。
- 2026-09-08：**历史数据手工规整**：过去的 run 全部并入 `calibration_history/`（glass×3、underwater×2，含被污染的 21:35 水下 run 及其事后诊断报告、从 git 初始提交找回的已废弃玻璃版）；旧目录 `glass_calibration_images/`、`underwater_calibration_images/` 已删除，当天第一条规则中的原图归档路径由本机制取代。run 索引见 `calibration_history/README.md`。
- 2026-09-08：**调用 subagent（Agent 工具）并行执行时，并行数不得超过 3**——模型 API 有限速，超并行会触发 429。大任务按 3+2 分波执行（例：论文翻译即分两波）。
- 2026-09-08：**测深阶段 s 键采样**：rect.py / rect_water.py 标定后进入测深时按 s，把当前帧原始左右图 + 该点测量值（disp/X/Y/Z/D）+ tkinter 弹窗输入的真实深度保存为一条验证样本，落在本次 run 目录 `depth_samples/`（`left_XXX.png`、`right_XXX.png` + `depth_samples.json` 追加记录，含误差字段）。无待测点或视差无效仅警告不保存；弹窗取消则整条放弃。实现为两脚本各自的 `SamplingStereoDepthRunner`（覆写 run() 加 s 分支），`stereo_depth_core.py` 保持不动。
- 2026-09-08：**水下测深三重修复**（015940 run 离线修复验证，13-39cm 十样本 RMS 0.94cm）：
  1. **角点 180° 翻转根治**：检测后 `canonicalize_corner_order` 统一角点链方向；双目拟合升级为"拟合→逐图误差门控→坏组尝试倒序找回→不可修剔除→重拟合"迭代（`stereo_calibrate_fixed_intrinsic` 返回 gate_stats 并写入报告，基线偏离物理参照 >10% 告警）；
  2. **深度规范修正**：固定空气 K 的水下标定输出折射虚拟深度 Z_v = c + Z_真实/n_w（近轴精确，数据上无法与真实深度规范区分——焦距扫描证实数据内无焦距信号，论文 §3.3 的 ×1.33 是物理先验而非拟合结果）。水下 npz 写入 `depth_scale=1.3333`/`depth_offset_cm≈1.1`（rect_water.py 配置 `WATER_REFRACTIVE_INDEX`/`WATER_PORT_OFFSET_CM`），`stereo_depth_core.measure_depth` 读取并按 Z_真实=scale×(Z_raw−offset) 还原；空气/玻璃 npz 无此二键，恒等不受影响；
  3. **numDisparities 128→192**：水下可测下限 12.2→8.1cm（空气 9→6.0cm）。
  附：`1.py` 的自由焦距路线实测不稳定（尺度简并，fx 漂移 ×1.5、RMS 恶化），勿作为修复手段。
- 2026-09-08：**新增独立水下测深入口 `measure_depth_underwater.py`**——不重新标定，默认读根目录水下 npz，可命令行指定历史 run（如 `python measure_depth_underwater.py 20260908_015940`）或任意 npz 路径，`--list` 列出可用 run。**今后新建脚本文件一律用英文名**（用户要求）；存量中文脚本名（测量深度.py 等）未经用户指示不主动改。
