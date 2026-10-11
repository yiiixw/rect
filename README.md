# 双目视觉水下测深系统

基于 Windows + USB 双目摄像头（1280×480 左右拼接帧，单目 640×480）的立体视觉测深系统。
核心思路是**三阶段渐进标定**：空气 → 空气+平板玻璃 → 水下，每一阶段固定上一阶段已验证的
内参，只重估新介质引入的畸变与双目外参，最终在水下状态实时点击测深（单位 cm）。

实测精度：水灰箱环境下 10–40cm 实时验证样本 RMS **0.52cm**、最大误差 0.89cm
（20260908_024805 水下标定 run，优于卷尺手测精度）。

## 标定流水线

```
MATLAB stereoCalibrate
  → transposition.m（MATLAB 侧无损导出，注意转置 IntrinsicMatrix/RotationOfCamera2）
  → stereo_params_plain.mat          # 空气参数唯一来源，禁止从 txt 手抄
  → 立体校正.py                       # → stereo_params_updated.npz（空气校正 map/Q）
  → rect.py                          # 空气+玻璃标定 → stereo_params_updated_with_glass.npz
  → rect_water.py                    # 水下标定       → stereo_params_updated_underwater.npz
```

## 主要脚本

| 脚本 | 用途 |
|---|---|
| `stereo_depth_core.py` | 测深核心 `StereoDepthRunner`：读 npz → remap 校正 → SGBM+WLS 视差 → reprojectImageTo3D，点击取 9×9 窗口中位数 |
| `rect.py` | 空气+玻璃标定入口（固定 K 只重估畸变 + 可选重估 R/T），标定后可直接进入测深 |
| `rect_water.py` | 水下标定入口（同构 rect.py），含 s 键深度验证采样、d 键整帧采集（`SampleDataStereoDepthRunner`） |
| `measure_depth_underwater.py` | 水下纯测距：不重新标定，读已有水下 npz；可指定历史 run 或任意 npz（`--list` 列出） |
| `collect_depth_data.py` | 水下数据采集：d 键整帧存 RGB / Depth(uint16 毫米) / Depth_color 三件套 |
| `测量深度.py` / `测量深度_玻璃.py` | 空气 / 加玻璃测距入口 |
| `立体校正.py` | 从 MATLAB mat 生成空气校正参数 npz |
| `采集数据.py` | 采集左右图对（MATLAB 标定用） |
| `1.py` / `2.py` | 实验脚本：论文"三步法"水下版、Snell 折射光线追踪 |
| `transposition.m` | MATLAB 侧 stereoParameters 导出脚本 |

## 核心方法论

1. **固定 K 的渐进标定**：玻璃/水下阶段固定焦距与光心
   （`CALIB_FIX_FOCAL_LENGTH | CALIB_FIX_PRINCIPAL_POINT`），只重估 5 参数畸变；
   再以 `CALIB_FIX_INTRINSIC` 重估双目 R/T。避免介质变化与内参耦合导致的病态解。
2. **角点 180° 起点歧义防护**：8×11 内角点网格 180° 旋转对称，左右视图角点链起点
   不一致会把 `stereoCalibrate` 整体拖歪（曾致基线 2.9→1.9cm）。所有角点检测后过
   `canonicalize_corner_order` 统一方向，双目拟合另带"逐图误差门控 → 坏组倒序找回 →
   剔除 → 重拟合"迭代。
3. **水下折射深度规范修正**：固定空气 K 的水下标定输出的是折射"虚拟深度"
   `Z_v = c + Z_真实/n_w`（近轴精确）。水下 npz 写入 `depth_scale=1.3333` /
   `depth_offset_cm≈1.1`（相机光心到玻璃外表面的等效空气光程），测深端按
   `Z_真实 = scale × (Z_v − offset)` 自动还原；空气/玻璃 npz 无此二键，恒等不受影响。
4. **单位链**：`SQUARE_SIZE=1.0`（cm）→ 角点世界坐标、T、Q 全为 cm；MATLAB 的 mm
   外参在导入时换算，测距代码不再乘任何比例。

## 参数与归档

- **参数按相机分目录**：`inner-param/盒子/`（旧相机）与 `inner-param/rebotfish1/`
  （新相机）各持有一套完整参数链（matlab.mat → 空气/玻璃/水下 npz + 角点 npz）。
  切换相机 = 把对应脚本里的路径指向对应目录。
- **标定历史归档**：每次标定把全部产物（角点 npz 含原图溯源键、参数 npz 快照、
  `calibration_report.md` 标定报告）写入 `calibration_history/<阶段>/<启动时间戳>/`，
  **只增不覆盖**；根目录 npz 保持"最新版"语义。回滚 = 把历史 run 的参数 npz 复制回去。
- 采集原图（`left/ right/ pair/`、各 `images/`）与测深采集数据（`sampledata/`）
  体积大，不入库（.gitignore）。

## 运行环境

- Windows + DirectShow 摄像头（`cv2.CAP_DSHOW`），双目拼接 1280×480
- Python 依赖：`opencv-contrib-python`（WLS 滤波需要 contrib 的 `cv2.ximgproc`，
  缺失时自动退回原始 SGBM）、`numpy`、`scipy`
- 棋盘格：8×11 内角点，方格边长 1cm

```bash
pip install opencv-contrib-python numpy scipy
python rect.py        # 玻璃标定（需先完成 MATLAB 空气标定并运行 transposition.m + 立体校正.py）
python rect_water.py  # 水下标定（玻璃/外壳/相机相对位置不得再变动）
python measure_depth_underwater.py  # 日常水下测距
```

各脚本运行前请先阅读其头部注释中的按键说明与依赖顺序。

## 文档

- `CLAUDE.md`：面向开发/维护的完整项目约定（数据流、硬性约定、变更记录）
- `水下校准模型.pdf` 及其中文翻译/解读：水下折射标定的方法论参考。
  **论文版权归原作者所有**，此处仅作个人研究存档。

## License

[MIT](LICENSE)
