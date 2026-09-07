# -*- coding: utf-8 -*-
"""
测量深度.py

主测距入口。
先运行：
1. transposition.m
2. 立体校正.py
再运行本文件。
"""

from pathlib import Path
from stereo_depth_core import StereoDepthRunner

# =========================
# 配置
# =========================
PARAM_FILE = Path(r"D:/rect/stereo_params_updated.npz")
# PARAM_FILE = Path(r"C:/Users/17434/Desktop/Camera/stereo_params_updated_with_glass.npz")
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 如果已安装 opencv-contrib-python，可以设 True 使用 WLS；否则会自动退回 SGBM。
USE_WLS = True

# 如果实时拼接画面左右半边本身反了，设 True。
# 你的当前摄像头实测需要 True；否则空气测距会跳到 140cm 左右。
SWAP_CAPTURE_HALVES = True

# 如果知道真实深度，例如 50 cm，就填 50.0；否则保持 None。
TRUE_Z_CM = 50
# TRUE_Z_CM = 50.0


if __name__ == "__main__":
    print("========== 空气测距 ==========")
    print(f"当前参数文件: {PARAM_FILE}")
    print("如果正在加玻璃测距，请不要使用这个入口。")

    app = StereoDepthRunner(
        param_file=PARAM_FILE,
        camera_id=CAMERA_ID,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        use_wls=USE_WLS,
        swap_capture_halves=SWAP_CAPTURE_HALVES,
        true_z_cm=TRUE_Z_CM,
        window_name="rectified_pair",
        disparity_window_name="disparity",
    )
    app.run()
