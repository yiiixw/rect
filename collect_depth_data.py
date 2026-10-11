# -*- coding: utf-8 -*-
"""
collect_depth_data.py

下水数据采集入口：不重新标定、不做点击测深，只按帧采集深度数据。
先用 rect_water.py 做过水下标定；本入口加载参数后实时显示校正画面和视差，
按 d 把当前帧存为三件套（三个文件夹用同一编号对应同一帧）：

  sampledata/underwater_<启动时间戳>/
  ├── RGB/rgb_XXX.png                  与深度图逐像素对齐的校正左图（视差参考视图）
  ├── Depth/depth_XXX.png              原始深度图 uint16（数值=深度毫米，即 cm×10，无效像素 0）
  └── Depth_color/depth_color_XXX.png  彩色深度图（逐帧归一化，仅供肉眼查看）

用法（参数解析复用 measure_depth_underwater.py）：
  python collect_depth_data.py                  # 根目录水下 npz 最新版
  python collect_depth_data.py 20260908_015940  # 指定历史 run
  python collect_depth_data.py path/to/xxx.npz  # 任意 npz 路径
  python collect_depth_data.py --list           # 列出可用 run

按键（窗口 rectified_pair_collect / disparity_collect）：
  鼠标左键 临时查看某点深度（仅屏幕显示，不落盘）
  d      采集当前帧（当前帧无有效深度像素时仅警告不保存）
  q/ESC  退出
  本入口禁用 s 键验证采样（那是 measure_depth_underwater.py 的职责）。

深度口径与点击测深完全一致：Z_真实 = depth_scale * (|Z_raw| - depth_offset_cm)，
由 npz 的 depth_scale/depth_offset_cm 键驱动，stereo_depth_core 自动应用；
d 键采集逻辑与 measure_depth_underwater.py 共用（rect_water.py 的
SampleDataStereoDepthRunner），本文件只做"禁用 s 键"的入口定制。
"""

import sys
from datetime import datetime
from pathlib import Path

from rect_water import SampleDataStereoDepthRunner
from measure_depth_underwater import list_available_runs, resolve_param_file


# =========================
# 配置
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# 采集数据按启动时间戳归档到根目录 sampledata/underwater_<时间戳>/ 下。
SAMPLEDATA_ROOT = SCRIPT_DIR / "sampledata"

# 与 measure_depth_underwater.py 保持一致（当前机器双目摄像头枚举为 1）。
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 已安装 opencv-contrib-python 时使用 WLS，否则自动退回 SGBM。
USE_WLS = True

# 必须和水下标定时保持一致。
SWAP_CAPTURE_HALVES = True


class DataCollectRunner(SampleDataStereoDepthRunner):
    """
    纯采集入口 Runner：d 键三件套采集逻辑全部继承自
    rect_water.SampleDataStereoDepthRunner，仅禁用 s 键验证采样。
    """

    extra_key_help = SampleDataStereoDepthRunner.extra_key_help + [
        "  s：本入口已禁用（验证采样请用 measure_depth_underwater.py）",
    ]

    def save_depth_sample(self, left, right, point, result):
        """覆写基类 s 键动作：本入口只做整帧采集。"""
        print("ℹ️ 本入口是数据采集（d 键）；验证采样请用 measure_depth_underwater.py（s 键）。")


def main():
    args = sys.argv[1:]

    if args and args[0] in ("-l", "--list"):
        list_available_runs()
        return

    param_file = resolve_param_file(args[0] if args else None)

    session_dir = SAMPLEDATA_ROOT / (
        "underwater_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print("========== 下水数据采集（d 键整帧采集，不测深） ==========")
    print(f"参数文件: {param_file}")
    print("深度验证采样（s 键）请用 measure_depth_underwater.py，本入口已禁用。")
    print(f"采集数据（d 键）保存目录: {session_dir}")

    app = DataCollectRunner(
        param_file=param_file,
        camera_id=CAMERA_ID,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        use_wls=USE_WLS,
        swap_capture_halves=SWAP_CAPTURE_HALVES,
        true_z_cm=None,
        window_name="rectified_pair_collect",
        disparity_window_name="disparity_collect",
        sample_dir=session_dir,  # s 键已禁用，此目录实际不使用
        sampledata_dir=session_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
