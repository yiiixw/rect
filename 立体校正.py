# -*- coding: utf-8 -*-
"""
立体校正.py

功能：
1. 从 MATLAB 导出的 stereo_params_plain.mat 读取原始 double 参数；
2. 可选交换左右相机顺序；
3. 生成 OpenCV 立体校正 maps 和 Q；
4. 保存 stereo_params_updated.npz，供测距脚本直接读取。

重要原则：
- 不再手动填写 K、dist、R、T；
- 不从 txt 复制参数；
- .mat -> Python float64 -> .npz，全程保留 double 数值，不做文本截断；
- Q 统一生成 cm 单位，所以测距脚本里不要再乘 UNIT_SCALE。
"""

from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat

# ================= 配置 =================
MAT_INPUT = Path(r"D:/rect/stereo_params_plain.mat")
NPZ_OUTPUT = Path(r"D:/rect/stereo_params_updated.npz")

# 如果 MATLAB 标定时的 Camera1/Camera2 与实时拼接画面的 left/right 反了，设为 True。
# 你之前新参数 T_x 为正，而实时 SGBM 默认 left-right 正视差，所以通常需要 True。
SWAP_LR = False

# stereoRectify 的 alpha：0 裁掉黑边；1 保留全部像素。
RECTIFY_ALPHA = 1

# 统一让 Q 输出 cm。MATLAB 标定常见单位是 mm，因此这里 T_cm = T_mm / 10。
# 如果你的 MATLAB worldUnits 不是 mm，请对应修改这里。
TRANSLATION_INPUT_UNIT = "mm"
Q_OUTPUT_UNIT = "cm"

np.set_printoptions(precision=17, suppress=False, linewidth=220, floatmode="unique")


def require_key(data, key):
    if key not in data:
        raise KeyError(f"MAT 文件中缺少变量: {key}")
    return data[key]


def load_plain_mat(mat_path: Path):
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到 MAT 文件: {mat_path}\n请先运行 transposition.m")

    data = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)

    K1 = np.asarray(require_key(data, "K1"), dtype=np.float64)
    K2 = np.asarray(require_key(data, "K2"), dtype=np.float64)
    dist1 = np.asarray(require_key(data, "dist1"), dtype=np.float64).reshape(-1)
    dist2 = np.asarray(require_key(data, "dist2"), dtype=np.float64).reshape(-1)
    R = np.asarray(require_key(data, "R"), dtype=np.float64)
    T = np.asarray(require_key(data, "T"), dtype=np.float64).reshape(3, 1)
    img_size_arr = np.asarray(require_key(data, "imgSize"), dtype=np.float64).reshape(-1)

    if img_size_arr.size != 2:
        raise ValueError(f"imgSize 应该是 [width, height]，但读到: {img_size_arr}")

    img_size = (int(img_size_arr[0]), int(img_size_arr[1]))

    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError(f"K1/K2 尺寸错误: K1={K1.shape}, K2={K2.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"R 尺寸错误: {R.shape}")
    if dist1.size not in (4, 5, 8, 12, 14) or dist2.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"dist1/dist2 长度异常: {dist1.size}, {dist2.size}")

    return K1, K2, dist1, dist2, R, T, img_size


def swap_left_right(K1, K2, dist1, dist2, R, T):
    """
    原外参表示 camera2 相对 camera1。
    交换相机顺序后，新外参为旧变换的逆：
        R_new = R_old.T
        T_new = -R_old.T @ T_old
    同时 K/dist 也必须交换。
    """
    K1_old = K1.copy()
    K2_old = K2.copy()
    dist1_old = dist1.copy()
    dist2_old = dist2.copy()
    R_old = R.copy()
    T_old = T.copy()

    K1_new = K2_old
    K2_new = K1_old
    dist1_new = dist2_old
    dist2_new = dist1_old
    R_new = R_old.T
    T_new = -R_old.T @ T_old

    return K1_new, K2_new, dist1_new, dist2_new, R_new, T_new


def convert_translation_to_cm(T):
    if TRANSLATION_INPUT_UNIT.lower() == "mm" and Q_OUTPUT_UNIT.lower() == "cm":
        return T / 10.0
    if TRANSLATION_INPUT_UNIT.lower() == "cm" and Q_OUTPUT_UNIT.lower() == "cm":
        return T.copy()
    if TRANSLATION_INPUT_UNIT.lower() == "m" and Q_OUTPUT_UNIT.lower() == "cm":
        return T * 100.0
    raise ValueError(f"未配置的单位转换: {TRANSLATION_INPUT_UNIT} -> {Q_OUTPUT_UNIT}")


def main():
    K1, K2, dist1, dist2, R, T_raw, img_size = load_plain_mat(MAT_INPUT)

    print(f"✅ 已读取原始参数: {MAT_INPUT}")
    print(f"✅ 单目图像尺寸 image_size = {img_size}")
    print(f"原始 T =\n{T_raw}")
    print(f"原始 baseline = {np.linalg.norm(T_raw):.17g} {TRANSLATION_INPUT_UNIT}")

    T_before_swap = T_raw.copy()

    if SWAP_LR:
        K1, K2, dist1, dist2, R, T_raw = swap_left_right(K1, K2, dist1, dist2, R, T_raw)
        print("⚠️ 已交换左右相机顺序，并同步反转 R/T。")

    T_cm = convert_translation_to_cm(T_raw)
    print(f"用于 stereoRectify 的 T =\n{T_cm}")
    print(f"用于 stereoRectify 的 baseline = {np.linalg.norm(T_cm):.17g} {Q_OUTPUT_UNIT}")
    print(f"用于 SGBM 正视差判断的 Tx = {T_cm[0, 0]:.17g} {Q_OUTPUT_UNIT}")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, img_size, R, T_cm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=RECTIFY_ALPHA
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, img_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, img_size, cv2.CV_32FC1
    )

    fx_rect = float(P1[0, 0])
    baseline_cm = float(np.linalg.norm(T_cm))

    np.savez(
        str(NPZ_OUTPUT),
        # 供实时测距直接使用
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        Q=Q,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        roi1=np.asarray(roi1, dtype=np.int32),
        roi2=np.asarray(roi2, dtype=np.int32),
        image_size=np.asarray(img_size, dtype=np.int32),
        img_size=np.asarray(img_size, dtype=np.int32),

        # 保存参与本次校正的原始参数，全部 float64
        K1=K1,
        K2=K2,
        dist1=dist1,
        dist2=dist2,
        R=R,
        T=T_raw,
        T_input_unit=np.asarray(TRANSLATION_INPUT_UNIT),
        Q_output_unit=np.asarray(Q_OUTPUT_UNIT),
        T_cm=T_cm,

        # 额外核对信息
        T_before_swap=T_before_swap,
        swap_lr=np.asarray(SWAP_LR, dtype=np.bool_),
        rectify_alpha=np.asarray(RECTIFY_ALPHA, dtype=np.float64),
        baseline_cm=np.asarray(baseline_cm, dtype=np.float64),
        fx_rect=np.asarray(fx_rect, dtype=np.float64),
    )

    print("\n" + "=" * 24 + " 核心参数核对 " + "=" * 24)
    print(f"K1 =\n{K1}")
    print(f"dist1 =\n{dist1}")
    print(f"K2 =\n{K2}")
    print(f"dist2 =\n{dist2}")
    print(f"R =\n{R}")
    print(f"T_raw =\n{T_raw}")
    print(f"T_cm =\n{T_cm}")
    print(f"P1 =\n{P1}")
    print(f"P2 =\n{P2}")
    print(f"Q =\n{Q}")
    print("-" * 64)
    print(f"baseline = {baseline_cm:.17g} cm")
    print(f"fx_rect = {fx_rect:.17g} px")
    print(f"理论 Z=50cm 时 disparity ≈ fx*B/Z = {fx_rect * baseline_cm / 50.0:.6f} px")
    print(f"理论 Z=100cm 时 disparity ≈ fx*B/Z = {fx_rect * baseline_cm / 100.0:.6f} px")
    print("-" * 64)
    print(f"✅ 已保存完整校正参数: {NPZ_OUTPUT}")
    print("下一步运行：测量深度.py")


if __name__ == "__main__":
    main()
