# -*- coding: utf-8 -*-
"""
measure_depth_underwater.py

下水测距入口：不重新标定，直接用已有水下标定参数测深。
先用 rect_water.py 做过至少一次水下标定，之后只要
玻璃/外壳/相机相对位置没动、设备仍处于下水状态，直接运行本文件即可。

用法：
  python measure_depth_underwater.py
        用根目录 stereo_params_updated_underwater.npz（"最新版"语义）
  python measure_depth_underwater.py 20260908_015940
        用指定历史 run 的参数（calibration_history/underwater/<时间戳>/ 下）
  python measure_depth_underwater.py path/to/xxx.npz
        直接指定任意 npz 参数文件路径
  python measure_depth_underwater.py --list
        列出 calibration_history/underwater/ 下所有可用 run

按键动作（测深窗口 rectified_pair_underwater / disparity_underwater）：
  鼠标左键 在校正图左半边点击选择待测点，实时显示该点 X/Y/Z/距离
  s      保存一条深度验证样本（要求当前已有待测点且视差有效，否则仅警告、不保存）：
         保存当前帧原始左右图与该点测量数据，弹窗输入实测真实深度（含测量误差），
         追加写入 depth_samples/underwater_<启动时间戳>/depth_samples.json；
         弹窗取消则整条放弃，不保存任何文件（弹窗期间实时画面暂停属正常）
  d      采集数据：当前帧深度图与对齐 RGB 图，按类别分三个子文件夹存到
         depth_samples/underwater_<启动时间戳>/sampledata/ 下——
           RGB/rgb_XXX.png                 与深度图逐像素对齐的校正左图（视差参考视图）
           Depth/depth_XXX.png             原始深度图 uint16（数值=深度毫米，即 cm×10，无效像素 0）
           Depth_color/depth_color_XXX.png 彩色深度图（逐帧归一化，仅供肉眼查看）
         三个文件夹用同一编号对应同一帧；当前帧无有效深度像素时仅警告不保存
  q/ESC  退出测深

（纯批量采集也可用专门的采集入口 collect_depth_data.py，d 键行为相同）

说明：
- 深度规范修正（折射"虚拟深度" -> 真实深度）由 npz 中的
  depth_scale / depth_offset_cm 键驱动，stereo_depth_core 自动应用；
  空气/玻璃参数无这两个键，恒等不受影响。
- s 键采样的 SamplingStereoDepthRunner 直接复用 rect_water.py 中的实现。
"""

import sys
from datetime import datetime
from pathlib import Path

from rect_water import SampleDataStereoDepthRunner


# =========================
# 配置
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# 默认参数：根目录水下 npz（"最新版"语义）。
DEFAULT_PARAM_FILE = SCRIPT_DIR / "stereo_params_updated_underwater.npz"
UNDERWATER_HISTORY_ROOT = SCRIPT_DIR / "calibration_history" / "underwater"
RUN_PARAM_FILENAME = "stereo_params_updated_underwater.npz"

# 与 rect_water.py 保持一致（当前机器双目摄像头枚举为 1）。
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 已安装 opencv-contrib-python 时使用 WLS，否则自动退回 SGBM。
USE_WLS = True

# 必须和水下标定时保持一致。
SWAP_CAPTURE_HALVES = True

# 如果知道真实深度，例如 50 cm，就填 50.0；否则保持 None。
TRUE_Z_CM = None


def list_available_runs():
    """列出 calibration_history/underwater/ 下含参数 npz 的 run。"""
    if not UNDERWATER_HISTORY_ROOT.exists():
        print(f"目录不存在: {UNDERWATER_HISTORY_ROOT}")
        return []

    runs = sorted(
        d.name
        for d in UNDERWATER_HISTORY_ROOT.iterdir()
        if d.is_dir() and (d / RUN_PARAM_FILENAME).is_file()
    )
    if runs:
        print(f"可用的水下标定 run（{UNDERWATER_HISTORY_ROOT}）：")
        for name in runs:
            print(f"  {name}")
    else:
        print(f"{UNDERWATER_HISTORY_ROOT} 下没有含 {RUN_PARAM_FILENAME} 的 run。")
    return runs


def resolve_param_file(arg):
    """
    解析命令行参数为参数 npz 路径：
    - None       -> 根目录最新版 npz
    - run 时间戳 -> calibration_history/underwater/<时间戳>/ 下的 npz
    - 文件路径   -> 直接使用该 npz
    """
    if arg is None:
        if not DEFAULT_PARAM_FILE.exists():
            sys.exit(
                f"❌ 找不到默认参数 {DEFAULT_PARAM_FILE}。\n"
                "   请先运行 rect_water.py 做水下标定，"
                "或用 --list 查看历史 run 并指定其一。"
            )
        return DEFAULT_PARAM_FILE

    direct = Path(arg)
    if direct.is_file():
        return direct.resolve()

    run_npz = UNDERWATER_HISTORY_ROOT / arg / RUN_PARAM_FILENAME
    if run_npz.is_file():
        return run_npz

    sys.exit(
        f"❌ 无法解析参数来源: {arg}\n"
        f"   既不是存在的 npz 文件，也不是 {UNDERWATER_HISTORY_ROOT} 下"
        f"含 {RUN_PARAM_FILENAME} 的 run 时间戳。\n"
        "   用 --list 查看可用 run。"
    )


def main():
    args = sys.argv[1:]

    if args and args[0] in ("-l", "--list"):
        list_available_runs()
        return

    param_file = resolve_param_file(args[0] if args else None)

    # 独立测深没有标定 run 目录，深度验证样本按启动时间戳
    # 归档到根目录 depth_samples/underwater_<时间戳>/ 下。
    samples_dir = SCRIPT_DIR / "depth_samples" / (
        "underwater_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print("========== 下水测距（使用已有水下标定参数，不重新标定） ==========")
    print(f"参数文件: {param_file}")
    print("前提：玻璃/外壳/相机相对位置与该次水下标定时一致，且设备处于下水状态。")
    print(f"深度验证样本（s 键）保存目录: {samples_dir}")
    print(f"采集数据（d 键）保存目录: {samples_dir / 'sampledata'}")

    app = SampleDataStereoDepthRunner(
        param_file=param_file,
        camera_id=CAMERA_ID,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        use_wls=USE_WLS,
        swap_capture_halves=SWAP_CAPTURE_HALVES,
        true_z_cm=TRUE_Z_CM,
        window_name="rectified_pair_underwater",
        disparity_window_name="disparity_underwater",
        sample_dir=samples_dir,
        sampledata_dir=samples_dir / "sampledata",
    )
    app.run()


if __name__ == "__main__":
    main()
