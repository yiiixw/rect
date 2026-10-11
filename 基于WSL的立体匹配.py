# -*- coding: utf-8 -*-
"""
基于WSL的立体匹配.py

这个文件现在不再单独保存一套测距逻辑，避免和“测量深度.py”出现单位、Q、左右顺序不一致。
它调用同一个 stereo_depth_core.py，只是默认窗口名保留为 WLS 风格。
"""

from pathlib import Path
from stereo_depth_core import StereoDepthRunner

# =========================
# 配置
# =========================
PARAM_FILE = Path(r"D:\rect\inner-param\rebotfish1\stereo_params_updated.npz")
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

USE_WLS = True
# 你的当前摄像头实测需要交换实时左右半图。
SWAP_CAPTURE_HALVES = True
TRUE_Z_CM = None
# TRUE_Z_CM = 50.0


if __name__ == "__main__":
    app = StereoDepthRunner(
        param_file=PARAM_FILE,
        camera_id=CAMERA_ID,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        use_wls=USE_WLS,
        swap_capture_halves=SWAP_CAPTURE_HALVES,
        true_z_cm=TRUE_Z_CM,
        window_name="rectified_pair",
        disparity_window_name="wls_disparity",
    )
    app.run()
