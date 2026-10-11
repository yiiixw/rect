# -*- coding: utf-8 -*-
"""
测量深度_玻璃.py

加玻璃测距入口。
先运行 rect.py 生成 stereo_params_updated_with_glass.npz，
再在玻璃仍然放在实际测量位置时运行本文件。
"""

from pathlib import Path
from stereo_depth_core import StereoDepthRunner

# =========================
# 配置
# =========================
PARAM_FILE = r"D:\rect\inner-param\rebotfish1\stereo_params_updated_with_glass.npz"
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 如果已安装 opencv-contrib-python，可以设 True 使用 WLS；否则会自动退回 SGBM。
USE_WLS = True

# 必须和玻璃标定 rect.py 保持一致。
# 当前摄像头拼接图的物理左右半边是反的，和 采集数据.py / 测量深度.py 保持一致。
SWAP_CAPTURE_HALVES = False

# 如果知道真实深度，例如 50 cm，就填 50.0；否则保持 None。
TRUE_Z_CM = None
# TRUE_Z_CM = 50.0


if __name__ == "__main__":
    print("========== 加玻璃测距 ==========")
    print(f"当前参数文件: {PARAM_FILE}")
    print("如果已经拿掉玻璃，请使用 测量深度.py。")

    app = StereoDepthRunner(
        param_file=PARAM_FILE,
        camera_id=CAMERA_ID,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        use_wls=USE_WLS,
        swap_capture_halves=SWAP_CAPTURE_HALVES,
        true_z_cm=TRUE_Z_CM,
        window_name="rectified_pair_with_glass",
        disparity_window_name="disparity_with_glass",
    )
    app.run()
