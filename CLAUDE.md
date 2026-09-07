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
2. 实时采集（按 `s` 缓存原图，`q` 结束后统一检测角点，避免采集卡顿）；
3. **固定焦距 fx/fy 和光心 cx/cy**，只重估玻璃引起的左右畸变 dist（5 参数 [k1,k2,p1,p2,k3]）；
4. `RECALIBRATE_STEREO_RT=True` 时用 `CALIB_FIX_INTRINSIC` 重估双目 R/T；
5. `stereoRectify` 生成 map1x/map1y/map2x/map2y/Q，保存 `stereo_params_updated_with_glass.npz`（同时存 `swap_lr`、`swap_capture_halves` 等一致性标记）；
6. 可选直接启动测深。

### rect_water.py — 水下标定
与 rect.py 同构，差异：初值来自玻璃 npz；每次启动按时间戳在 `underwater_calibration_images/<时间戳>/` 落盘原图；按 `s` 时实时检测角点，左右都成功才计入标定；**不做 1.33 经验缩放**（MLE 直接重估水下 dist 再重估 R/T）。

## 硬性约定（改动前必须理解）

1. **单位链**：`SQUARE_SIZE=1.0` 表示 cm；角点世界坐标为 cm，因此标定出的 T 和 Q 均为 cm。MATLAB 空气 T 单位由 `MATLAB_T_UNIT`/`TRANSLATION_INPUT_UNIT`（当前 "mm"）转成 cm。Q 已经是 cm，**测距代码中不得再乘任何 UNIT_SCALE**。
2. **SWAP_CAPTURE_HALVES=True**：当前摄像头 1280x480 拼接图的物理左右半边是反的，切半图后需交换。此开关在 采集数据.py / 立体校正流程 / rect.py / rect_water.py / 各测深入口 及 npz 中的 `swap_capture_halves` 键之间必须保持一致，否则画面错位、测距错误。
3. **SWAP_LR ≠ SWAP_CAPTURE_HALVES**：`SWAP_LR`（立体校正.py）交换的是 MATLAB 标定的 Camera1/Camera2 顺序（R_new=R.T, T_new=-R.T@T，K/dist 同步交换）；`rect.py` 的 `SWAP_AIR_PARAMS_LR=None` 会从空气 npz 自动读取 `swap_lr`，二者不要手动绑定。
4. **固定 K 的标定策略**：玻璃/水下阶段只重估畸变（`CALIB_FIX_FOCAL_LENGTH | CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_K4/K5/K6`），焦距光心继承上一阶段。这是项目核心方法论，勿随意放开。
5. **npz 契约**：测深核心要求 npz 必含 `map1x/map1y/map2x/map2y/Q/image_size`，缺一即抛 KeyError。新增字段可以，删字段不行。
6. **相机**：`CAMERA_ID=0`，`cv2.CAP_DSHOW`（Windows 专用），单目尺寸 640x480（`IMG_SIZE=(640,480)`，OpenCV 宽高序）。
7. **棋盘格**：`PATTERN_SIZE=(8,11)` 是**内角点数**（横向×纵向），不是格子数；检测时先 `findChessboardCornersSB`（EXHAUSTIVE|ACCURACY），失败用传统方法+`cornerSubPix` 兜底；灰度先过 CLAHE 增强对抗玻璃反光/水下低对比。
8. **中文路径**：脚本和图像目录含中文，写图必须走 `imwrite_unicode`（`cv2.imencode` + `tofile`），不要直接 `cv2.imwrite`。

## 后续规则记录（持续追加）

- 2026-09-07：初始化仓库（本地 git 管理，无远程）。`left/`、`right/`、`pair/`、`underwater_calibration_images/` 图像数据不入库；标定参数 npz/mat 和角点 npz 保留入库以便回滚到已验证的标定结果。
