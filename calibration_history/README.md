# 标定历史归档

每次标定（rect.py / rect_water.py）把本次完整产物写入 `<glass|underwater>/<启动时间戳>/`，**只增不覆盖**：

```
<stage>/<时间戳>/
├── images/{left,right,pair}/*.png   # 采集原图（gitignore，不入库）
├── *_chessboard_corners.npz         # 角点数据（含 left_files/right_files 溯源键）
├── stereo_params_updated_*.npz      # 参数完整快照
└── calibration_report.md            # 标定报告
```

根目录下的同名 npz 是"最新版"（继续覆盖），测深入口按固定路径读取。回滚到某次历史标定 = 把该 run 的参数 npz 复制回根目录（或改 `PARAM_FILE` 指向历史副本）。

## Run 索引（2026-09-08 手工规整历史数据）

本目录中 2026-09-08 当天及之前的 run 为手工规整（报告为事后生成，标注"事后整理"）：

| run 目录 | 阶段 | 状态 | 说明 |
|---|---|---|---|
| glass/20260907_225147 | 玻璃 | 已废弃 | 21 组，baseline 2.5707；K/dist 曾作为水下 21:35 标定的输入 |
| glass/20260908_003547 | 玻璃 | 启动即退出 | 目录为空，无任何数据 |
| glass/20260908_003639 | 玻璃 | **当前生效** | 33 采集/30 有效，stereo RMS 0.149 px，baseline 2.9621 |
| underwater/20260907_213528 | 水下 | **不可用（已污染）** | 含角点 180° 翻转的坏图 #30，双目 RMS 12.9 px，基线崩塌至 0.96 cm |
| underwater/20260907_225715 | 水下 | 采集未完成 | 10 组原图，未产生标定产物 |
| underwater/20260908_012025 | 水下 | 可用（备用） | 64 采集/45 有效，stereo RMS 0.334 px，baseline 2.9773，与玻璃阶段（2.9621）衔接良好；需要时把本 run 的水下 npz 复制回根目录即可回滚 |
| underwater/20260908_015940 | 水下 | 已离线修复（被取代） | 原始参数被 4 组 180° 翻转坏图污染（RMS 22.17px、基线塌缩 1.96）；离线翻转找回 57 组全保留重拟合后 RMS 0.683px、基线 2.9668，并加入深度规范修正（1.3333×(Z−1.1)）与 numDisp=192；自带 11 条 depth_samples 验证：13-39cm 十样本 RMS 0.94cm（卷尺误差内）。详见该目录 `repair_report.md` |
| underwater/20260908_024805 | 水下 | **当前生效** | 加固代码首次实战：66 采/55 有效，RMS 0.385px、基线 2.9457（与玻璃参照差 0.6%）、零翻转零门控事件；9 条实时验证样本 10-40cm 八条有效 **RMS 0.52cm、最大 0.89cm**（优于卷尺手测精度），7cm 样本超量程下限 8.1cm 属预期失配 |

## 空气阶段为什么不在本目录

空气参数由 MATLAB（transposition.m）→ 立体校正.py 确定性生成，无实时采集环节，且 `stereo_params_plain.mat` / `stereo_params_updated.npz` 是 rect.py / 立体校正.py / 测量深度.py 的固定路径输入，保留在仓库根目录（对应时间 2026-09-07 17:06–17:28；空气采集原图在根目录 left/ right/ pair/，不入库）。

## 旧目录去向

`glass_calibration_images/` 与 `underwater_calibration_images/`（旧归档根目录）已于 2026-09-08 并入本目录对应 run 后删除。
