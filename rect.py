# -*- coding: utf-8 -*-
"""
rect_glass_calib_and_depth.py

功能：
1. 读取空气中双目标定得到的 K1、K2、dist1、dist2、R、T；
2. 实时采集“空气 + 平板玻璃”下的左右原图并按时间戳归档，结束后统一检测角点；
3. 固定空气中焦距和光心，只重新估计玻璃引起的左右畸变系数；
4. 可选重新估计空气+玻璃下的双目 R、T；
5. 重新生成 stereoRectify 的 map1/map2/Q；
6. 保存 stereo_params_updated_with_glass.npz；
7. 调用 SamplingStereoDepthRunner（StereoDepthRunner 子类）直接测深，
   测深阶段可按 s 保存深度验证样本。

重点：
- 焦距 fx/fy 和光心 cx/cy 固定不动；
- 主要更新 dist1_glass、dist2_glass；
- 双目测深必须左右相机都补偿；
- npz 必须包含 map1x/map1y/map2x/map2y/Q，否则你的测距核心无法运行。

两个阶段的按键动作：
【阶段一：标定采集】（窗口 left_glass_detect / right_glass_detect / pair_glass_detect）
  s      缓存并落盘当前左右原图（采集阶段不实时检测角点，避免卡顿）
  q/ESC  结束采集，随后统一批量检测角点并执行标定计算
【阶段二：测深】（窗口 rectified_pair_with_glass / disparity_with_glass）
  鼠标左键 在校正图左半边点击选择待测点，实时显示该点 X/Y/Z/距离
  s      保存一条深度验证样本（要求当前已有待测点且视差有效，否则仅提示警告、不保存）：
         保存当前帧原始左右图与该点测量数据，弹窗输入实测真实深度（含测量误差），
         一并追加到归档目录 depth_samples/ 的 depth_samples.json；
         弹窗取消则整条放弃，不保存任何文件（弹窗期间实时画面暂停属正常）
  q/ESC  退出测深
"""

import json

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.io import loadmat

import stereo_depth_core
from stereo_depth_core import StereoDepthRunner


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = SCRIPT_DIR / "inner-param" / "rebotfish1"

# 每次程序启动都用独立时间戳目录归档本次标定的全部产物
# （原图 / 角点 npz / 参数 npz / 标定报告），只增不覆盖，便于追溯历史。
# 根目录下的 NPZ_OUTPUT / CORNERS_OUTPUT 仍是"最新版"语义（继续覆盖），
# 供测距脚本按固定路径读取。
PROGRAM_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_ROOT = SCRIPT_DIR / "calibration_history" / "glass"
RUN_DIR = HISTORY_ROOT / PROGRAM_START_TIMESTAMP
GLASS_IMAGES_OUTPUT = RUN_DIR / "images"

MAT_INPUT = SCRIPT_DIR / "stereo_params_plain.mat"
NPZ_OUTPUT = SCRIPT_DIR / "stereo_params_updated_with_glass.npz"
AIR_RECTIFY_NPZ = SCRIPT_DIR / "stereo_params_updated.npz"

# 可选：保存采集到的角点，方便以后不重新采集
CORNERS_OUTPUT = SCRIPT_DIR / "glass_chessboard_corners.npz"

# 测深阶段按 s 保存的深度验证样本目录（原始左右图 + depth_samples.json）
DEPTH_SAMPLES_DIR = RUN_DIR / "depth_samples"


# =========================================================
# 2. 棋盘格配置
# =========================================================
# 注意：这里是“内角点数”，不是格子数。
# 如果你的棋盘格横向有 10 个格子、纵向有 13 个格子，则内角点是 9 x 12。
# 如果检测不到，可以尝试改成 (12, 9)。
# PATTERN_SIZE = (8, 11)
PATTERN_SIZE = (8, 11)

# 每个小方格真实边长，单位用 cm。
# 如果这里用 cm，那么重新估计出来的 T 也是 cm。
SQUARE_SIZE = 1.0


# =========================================================
# 3. 摄像头配置
# =========================================================
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 单目尺寸。你的双目是左右拼接，所以单目是 640x480。
SINGLE_WIDTH = FRAME_WIDTH // 2
SINGLE_HEIGHT = FRAME_HEIGHT
IMG_SIZE = (SINGLE_WIDTH, SINGLE_HEIGHT)  # OpenCV 顺序：width, height

# 当前摄像头拼接图的物理左右半边是反的，和 采集数据.py / 测量深度.py 保持一致。
SWAP_CAPTURE_HALVES = False


# =========================================================
# 4. 标定/校正配置
# =========================================================
# 是否用玻璃下采集的角点重新估计双目 R/T。
# True：推荐，更适合最终测深；
# False：只更新左右畸变，沿用空气中 R/T。
RECALIBRATE_STEREO_RT = True

# 空气参数是否左右交换，必须和 立体校正.py 生成 stereo_params_updated.npz 时的 SWAP_LR 一致。
# None 表示从 AIR_RECTIFY_NPZ 自动读取 swap_lr，避免和采集半图交换配置手动绑在一起。
SWAP_AIR_PARAMS_LR = None

# 如果你的 MATLAB 标定单位是 mm，而你想让 Q 输出 cm，就设 "mm"。
# 如果你的 MATLAB 标定单位已经是 cm，就设 "cm"。
# 注意：如果 RECALIBRATE_STEREO_RT=True，则 T 来自本脚本，单位就是 SQUARE_SIZE 的单位，也就是 cm。
MATLAB_T_UNIT = "mm"

# stereoRectify 的 alpha：
# 0：裁掉黑边，视差更稳定；
# 1：保留全部视野，黑边更多。
RECTIFY_ALPHA = 0

# 是否采集完成后直接启动测深
RUN_DEPTH_AFTER_CALIB = True

# 测深时是否使用 WLS
USE_WLS = True


np.set_printoptions(
    precision=17,
    suppress=False,
    linewidth=220,
    floatmode="unique"
)


# =========================================================
# 工具函数
# =========================================================
def imwrite_unicode(path, img):
    """支持含中文路径的 OpenCV 图像写入。"""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, img)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def split_frame(frame):
    """
    把 1280x480 的双目拼接图切成左右两张 640x480。
    """
    h, w = frame.shape[:2]
    mid = w // 2

    first = frame[:, :mid]
    second = frame[:, mid:]

    if SWAP_CAPTURE_HALVES:
        return second, first

    return first, second


def enhance_for_chessboard(gray):
    """
    玻璃会带来反光和对比度下降。
    这里用 CLAHE 提升棋盘格检测稳定性。
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return gray_eq


def canonicalize_corner_order(corners):
    """
    消除棋盘格角点的 180° 起点歧义。

    8x11 内角点网格本身 180° 旋转对称，检测器可能从任一对角返回角点链。
    单目标定对此无感（等价于板的另一种合法位姿），但左右视图起点不一致时
    双目逐点对应全错，会把 stereoCalibrate 整体拖歪（曾导致基线塌缩）。
    规定角点链首角在图像上方，左右两视图统一规范后即安全。
    """
    if corners[0, 1] > corners[-1, 1]:
        return corners[::-1].copy()
    return corners


def detect_corners(img_bgr, pattern_size):
    """
    检测单张图里的棋盘格角点。
    优先使用 findChessboardCornersSB，失败再用传统 findChessboardCorners。
    返回前统一角点链方向（消除 180° 起点歧义）。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = enhance_for_chessboard(gray)

    # 新版棋盘格检测，更稳
    ret, corners = cv2.findChessboardCornersSB(
        gray_eq,
        pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )

    if ret:
        corners = corners.astype(np.float32)
        return True, canonicalize_corner_order(corners.reshape(-1, 2))

    # 传统方法兜底
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret, corners = cv2.findChessboardCorners(gray_eq, pattern_size, flags)

    if ret:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4
        )
        corners = cv2.cornerSubPix(
            gray_eq,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        return True, canonicalize_corner_order(corners.reshape(-1, 2))

    return False, None


def load_air_stereo_params(mat_path: Path):
    """
    读取 MATLAB 导出的空气双目标定参数。
    要求 mat 里包含：
    K1, K2, dist1, dist2, R, T, imgSize
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到空气标定参数文件: {mat_path}")

    data = loadmat(str(mat_path), squeeze_me=True)

    required = ["K1", "K2", "dist1", "dist2", "R", "T", "imgSize"]
    for key in required:
        if key not in data:
            raise KeyError(f"MAT 文件中缺少变量: {key}")

    K1 = np.asarray(data["K1"], dtype=np.float64)
    K2 = np.asarray(data["K2"], dtype=np.float64)
    dist1 = np.asarray(data["dist1"], dtype=np.float64).reshape(-1)
    dist2 = np.asarray(data["dist2"], dtype=np.float64).reshape(-1)
    R = np.asarray(data["R"], dtype=np.float64)
    T = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)
    img_size_arr = np.asarray(data["imgSize"], dtype=np.float64).reshape(-1)
    img_size = (int(img_size_arr[0]), int(img_size_arr[1]))

    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError(f"K1/K2 尺寸错误: K1={K1.shape}, K2={K2.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"R 尺寸错误: {R.shape}")
    if T.shape != (3, 1):
        raise ValueError(f"T 尺寸错误: {T.shape}")

    # 保证畸变长度为 5: [k1,k2,p1,p2,k3]
    dist1 = normalize_dist_5(dist1)
    dist2 = normalize_dist_5(dist2)

    return K1, K2, dist1, dist2, R, T, img_size


def resolve_air_param_swap():
    """
    玻璃标定使用 stereo_params_plain.mat 的 K/R/T。
    是否交换这套空气参数，应该跟普通立体校正的 SWAP_LR 一致，
    而不是跟实时半图交换 SWAP_CAPTURE_HALVES 绑定。
    """
    if SWAP_AIR_PARAMS_LR is not None:
        return bool(SWAP_AIR_PARAMS_LR)

    if not AIR_RECTIFY_NPZ.exists():
        print(f"⚠️ 找不到普通校正参数 {AIR_RECTIFY_NPZ}，空气参数默认不交换。")
        print("   如需交换，请把 SWAP_AIR_PARAMS_LR 明确设为 True。")
        return False

    data = np.load(str(AIR_RECTIFY_NPZ), allow_pickle=False)
    if "swap_lr" not in data:
        print(f"⚠️ {AIR_RECTIFY_NPZ} 中没有 swap_lr，空气参数默认不交换。")
        print("   如需交换，请把 SWAP_AIR_PARAMS_LR 明确设为 True。")
        return False

    swap_lr = bool(np.asarray(data["swap_lr"]))
    print(f"✅ 已从普通校正参数读取 SWAP_LR = {swap_lr}")
    return swap_lr


def normalize_dist_5(dist):
    """
    统一畸变参数为 OpenCV 的 5 参数格式：
    [k1, k2, p1, p2, k3]
    """
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    if dist.size >= 5:
        return dist[:5].copy()

    out = np.zeros(5, dtype=np.float64)
    out[:dist.size] = dist
    return out


def swap_left_right(K1, K2, dist1, dist2, R, T):
    """
    与 立体校正.py 保持同一套左右交换规则。
    原外参表示 camera2 相对 camera1；交换相机顺序后使用逆变换。
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


def make_object_points(pattern_size, square_size):
    """
    生成棋盘格世界坐标。
    标定板平面 Z=0。
    单位由 square_size 决定，这里是 cm。
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def collect_glass_corners():
    """
    先采集并缓存空气+玻璃下的左右原图，采集结束后再统一检测角点。

    采集阶段不做棋盘格检测，避免实时画面卡顿。
    按 s 缓存的同时把原图落盘归档，与 rect_water.py 的归档方式对齐。
    """
    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    cached_pairs = []      # (left, right, left文件名, right文件名)
    dropped_indices = []   # 角点检测失败被丢弃的缓存组编号（从 1 开始）
    valid_left_files = []
    valid_right_files = []

    left_output = GLASS_IMAGES_OUTPUT / "left"
    right_output = GLASS_IMAGES_OUTPUT / "right"
    pair_output = GLASS_IMAGES_OUTPUT / "pair"
    left_output.mkdir(parents=True, exist_ok=True)
    right_output.mkdir(parents=True, exist_ok=True)
    pair_output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # 某些 DirectShow 驱动会忽略该设置，支持时可减少旧帧积压。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"摄像头实际分辨率: {actual_w} x {actual_h}")
    if actual_w != FRAME_WIDTH or actual_h != FRAME_HEIGHT:
        print(f"⚠️ 实际分辨率不是期望的 {FRAME_WIDTH}x{FRAME_HEIGHT}，切图可能错误。")

    cv2.namedWindow("left_glass_detect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("right_glass_detect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pair_glass_detect", cv2.WINDOW_NORMAL)

    print("\n========== 采集空气+玻璃棋盘格角点 ==========")
    print("操作：")
    print("  s：缓存并落盘当前左右原图")
    print("  q / ESC：结束采集，然后统一检测角点")
    print("建议：至少保存 15~30 组，姿态要有变化，棋盘格覆盖画面中心和边缘。")
    print(f"当前 PATTERN_SIZE = {PATTERN_SIZE}")
    print("如果一直检测不到，优先把 PATTERN_SIZE 改成 (12, 9) 或检查是否填的是内角点数。")
    print(f"标定原图保存目录: {GLASS_IMAGES_OUTPUT}\n")

    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取图像失败")
            continue

        left, right = split_frame(frame)

        vis_l = left.copy()
        vis_r = right.copy()
        cv2.putText(vis_l, f"cached={idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        pair_vis = cv2.hconcat([vis_l, vis_r])

        cv2.imshow("left_glass_detect", vis_l)
        cv2.imshow("right_glass_detect", vis_r)
        cv2.imshow("pair_glass_detect", pair_vis)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("s"), ord("S")):
            save_idx = idx + 1
            left_path = left_output / f"left_{save_idx:03d}.png"
            right_path = right_output / f"right_{save_idx:03d}.png"
            pair_path = pair_output / f"pair_{save_idx:03d}.png"
            pair_raw = cv2.hconcat([left, right])

            ok_save_l = imwrite_unicode(left_path, left)
            ok_save_r = imwrite_unicode(right_path, right)
            ok_save_pair = imwrite_unicode(pair_path, pair_raw)

            if ok_save_l and ok_save_r and ok_save_pair:
                cached_pairs.append((left.copy(), right.copy(), left_path.name, right_path.name))
                idx = save_idx
                print(f"✅ 已缓存并落盘第 {idx} 组原图")
            else:
                print(
                    "⚠️ 图像落盘失败，该组不缓存。"
                    f" left={ok_save_l}, right={ok_save_r}, pair={ok_save_pair}"
                )

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n开始统一检测 {len(cached_pairs)} 组缓存图像的棋盘格角点...")
    for pair_idx, (left, right, left_name, right_name) in enumerate(cached_pairs, start=1):
        ok_l, corners_l = detect_corners(left, PATTERN_SIZE)
        ok_r, corners_r = detect_corners(right, PATTERN_SIZE)

        if ok_l and ok_r:
            objpoints.append(objp.copy())
            imgpoints_left.append(corners_l.astype(np.float32).copy())
            imgpoints_right.append(corners_r.astype(np.float32).copy())
            valid_left_files.append(left_name)
            valid_right_files.append(right_name)
            print(f"✅ 第 {pair_idx}/{len(cached_pairs)} 组：左右角点检测成功")
        else:
            dropped_indices.append(pair_idx)
            print(
                f"⚠️ 第 {pair_idx}/{len(cached_pairs)} 组：已丢弃，"
                f"left={ok_l}, right={ok_r}"
            )

    print(
        f"角点检测完成：成功 {len(objpoints)} 组，"
        f"失败 {len(cached_pairs) - len(objpoints)} 组。"
    )

    if len(objpoints) < 10:
        raise RuntimeError(
            f"有效角点组数太少：{len(objpoints)}。建议至少 15~30 组，否则结果不稳。"
        )

    objpoints = [p.astype(np.float32) for p in objpoints]
    imgpoints_left = [p.astype(np.float32) for p in imgpoints_left]
    imgpoints_right = [p.astype(np.float32) for p in imgpoints_right]

    corners_data = dict(
        objpoints=np.asarray(objpoints, dtype=np.float32),
        imgpoints_left=np.asarray(imgpoints_left, dtype=np.float32),
        imgpoints_right=np.asarray(imgpoints_right, dtype=np.float32),
        # 每组角点对应的原图文件名（images/{left,right}/ 下），用于溯源。
        left_files=np.asarray(valid_left_files),
        right_files=np.asarray(valid_right_files),
        pattern_size=np.asarray(PATTERN_SIZE, dtype=np.int32),
        square_size=np.asarray(SQUARE_SIZE, dtype=np.float32),
        image_size=np.asarray(IMG_SIZE, dtype=np.int32),
    )
    np.savez(str(CORNERS_OUTPUT), **corners_data)
    np.savez(str(RUN_DIR / CORNERS_OUTPUT.name), **corners_data)

    print(f"✅ 已保存空气+玻璃角点数据: {CORNERS_OUTPUT}")
    print(f"✅ 已归档本次角点数据: {RUN_DIR / CORNERS_OUTPUT.name}")
    print(f"✅ 已保存空气+玻璃标定原图: {GLASS_IMAGES_OUTPUT}")

    capture_stats = {
        "cached": len(cached_pairs),
        "valid": len(objpoints),
        "dropped": dropped_indices,
    }

    return objpoints, imgpoints_left, imgpoints_right, capture_stats


def calibrate_one_camera_fixed_K(objpoints, imgpoints, K_air, dist_air, img_size, camera_name):
    """
    固定焦距和光心，只估计畸变系数。

    固定：
    fx, fy, cx, cy

    优化：
    k1, k2, p1, p2, k3
    以及每张标定图自己的 rvec/tvec
    """
    K_init = K_air.copy().astype(np.float64)
    dist_init = normalize_dist_5(dist_air)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7
    )

    rms, K_new, dist_new, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=K_init,
        distCoeffs=dist_init,
        flags=flags,
        criteria=criteria
    )

    dist_new = normalize_dist_5(dist_new)

    print("\n" + "=" * 70)
    print(f"{camera_name} 空气+玻璃单目标定结果")
    print("=" * 70)
    print(f"RMS reprojection error = {rms:.6f} px")
    print("K_air =")
    print(K_air)
    print("K_new =")
    print(K_new)
    print("说明：这里固定了焦距和光心，所以 K_new 理论上应基本等于 K_air。")
    print(f"dist_air [{camera_name}] = [k1, k2, p1, p2, k3]")
    print(dist_air)
    print(f"dist_glass [{camera_name}] = [k1, k2, p1, p2, k3]")
    print(dist_new)

    return K_new, dist_new, rms, rvecs, tvecs


def convert_matlab_T_to_cm(T_raw):
    """
    如果沿用 MATLAB 空气中 T，需要把单位转成 cm，用于 Q 输出 cm。
    """
    unit = MATLAB_T_UNIT.lower()

    if unit == "mm":
        return T_raw / 10.0

    if unit == "cm":
        return T_raw.copy()

    if unit == "m":
        return T_raw * 100.0

    raise ValueError(f"未知 MATLAB_T_UNIT: {MATLAB_T_UNIT}")


def stereo_calibrate_fixed_intrinsic(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1,
    dist1,
    K2,
    dist2,
    img_size,
    baseline_expect_cm=None
):
    """
    固定左右 K/dist，只重新估计双目 R/T。带逐图离群门控。

    单张图左右角点对应错乱（典型为 180° 起点翻转）会把最小二乘整体拖歪，
    门控流程：拟合 -> 逐图重投影误差检验 -> 坏组尝试角点倒序找回 ->
    找不回则剔除 -> 重拟合，迭代到无坏组为止。

    baseline_expect_cm：传入空气阶段基线（cm）作为物理参照，偏差过大时告警。
    """
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7
    )

    def fit(o, l, r):
        rms, _, _, _, _, Rm, T, _, _ = cv2.stereoCalibrate(
            objectPoints=o,
            imagePoints1=l,
            imagePoints2=r,
            cameraMatrix1=K1.copy(),
            distCoeffs1=dist1.copy(),
            cameraMatrix2=K2.copy(),
            distCoeffs2=dist2.copy(),
            imageSize=img_size,
            flags=flags,
            criteria=criteria
        )
        return Rm, T.reshape(3, 1), rms

    def per_image_errors(o, l, r, Rm, T):
        errs = []
        for oi, li, ri in zip(o, l, r):
            _, rv, tv = cv2.solvePnP(oi, li, K1, dist1)
            Rw, _ = cv2.Rodrigues(rv)
            proj, _ = cv2.projectPoints(oi, Rm @ Rw, Rm @ tv + T, K2, dist2)
            errs.append(float(np.sqrt(((proj.reshape(-1, 2) - ri) ** 2).sum(1).mean())))
        return np.asarray(errs)

    gate_stats = {"repaired": [], "dropped": []}
    o = list(objpoints)
    l = list(imgpoints_left)
    r = list(imgpoints_right)
    group_ids = list(range(1, len(o) + 1))

    R, T, rms = fit(o, l, r)

    for _ in range(3):
        errs = per_image_errors(o, l, r, R, T)
        med = float(np.median(errs))
        bad = [k for k, e in enumerate(errs) if e > max(3.0, 6.0 * med)]
        if not bad:
            break

        keep = []
        for k in range(len(o)):
            if k not in bad:
                keep.append(k)
                continue

            ri_flip = r[k][::-1].copy()
            _, rv, tv = cv2.solvePnP(o[k], l[k], K1, dist1)
            Rw, _ = cv2.Rodrigues(rv)
            proj, _ = cv2.projectPoints(o[k], R @ Rw, R @ tv + T, K2, dist2)
            e_flip = float(np.sqrt(((proj.reshape(-1, 2) - ri_flip) ** 2).sum(1).mean()))

            if e_flip < 1.5:
                r[k] = ri_flip
                gate_stats["repaired"].append(group_ids[k])
                keep.append(k)
                print(
                    f"⚠️ 组#{group_ids[k]}: 左右角点疑似 180° 翻转，"
                    f"已自动倒序找回（{errs[k]:.1f} -> {e_flip:.2f} px）"
                )
            else:
                gate_stats["dropped"].append(group_ids[k])
                print(
                    f"⚠️ 组#{group_ids[k]}: 重投影误差 {errs[k]:.1f} px"
                    f"（倒序后 {e_flip:.1f} px 仍差），已剔除"
                )

        o = [o[k] for k in keep]
        l = [l[k] for k in keep]
        r = [r[k] for k in keep]
        group_ids = [group_ids[k] for k in keep]
        R, T, rms = fit(o, l, r)

    print("\n" + "=" * 70)
    print("空气+玻璃双目标定 R/T 结果（含逐图离群门控）")
    print("=" * 70)
    print(f"Stereo RMS reprojection error = {rms:.6f} px")
    print(f"角点翻转找回组: {gate_stats['repaired'] if gate_stats['repaired'] else '无'}；"
          f"剔除组: {gate_stats['dropped'] if gate_stats['dropped'] else '无'}")
    print("R_glass =")
    print(R)
    print("T_glass_cm =")
    print(T)
    print(f"baseline_glass = {np.linalg.norm(T):.6f} cm")

    if baseline_expect_cm is not None:
        b = float(np.linalg.norm(T))
        if abs(b - baseline_expect_cm) / baseline_expect_cm > 0.10:
            print(
                f"⚠️ 基线 {b:.4f} cm 与物理参照 {baseline_expect_cm:.4f} cm 偏差超过 10%，"
                "标定结果可疑，建议重新采集！"
            )
        else:
            print(f"✅ 基线与物理参照 {baseline_expect_cm:.4f} cm 偏差 "
                  f"{abs(b - baseline_expect_cm):.4f} cm，正常。")

    return R, T, rms, gate_stats


def build_rectify_and_save_npz(K1, dist1, K2, dist2, R, T_cm, img_size, swap_air_params_lr):
    """
    根据玻璃补偿后的 K/dist/R/T 生成立体校正 map 和 Q，并保存给测距程序使用。
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1,
        distCoeffs1=dist1,
        cameraMatrix2=K2,
        distCoeffs2=dist2,
        imageSize=img_size,
        R=R,
        T=T_cm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=RECTIFY_ALPHA
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, img_size, cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, img_size, cv2.CV_32FC1
    )

    baseline_cm = float(np.linalg.norm(T_cm))
    fx_rect = float(P1[0, 0])

    npz_data = dict(
        # 测距核心必须读取这些
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

        # 保存玻璃补偿后的参数
        K1=K1,
        K2=K2,
        dist1=dist1,
        dist2=dist2,
        R=R,
        T=T_cm,
        T_cm=T_cm,

        # 额外核对信息
        baseline_cm=np.asarray(baseline_cm, dtype=np.float64),
        fx_rect=np.asarray(fx_rect, dtype=np.float64),
        rectify_alpha=np.asarray(RECTIFY_ALPHA, dtype=np.float64),
        swap_lr=np.asarray(swap_air_params_lr, dtype=np.bool_),
        swap_capture_halves=np.asarray(SWAP_CAPTURE_HALVES, dtype=np.bool_),
    )
    np.savez(str(NPZ_OUTPUT), **npz_data)
    np.savez(str(RUN_DIR / NPZ_OUTPUT.name), **npz_data)

    print("\n" + "=" * 70)
    print("已生成空气+玻璃立体校正参数")
    print("=" * 70)
    print(f"保存路径(最新版): {NPZ_OUTPUT}")
    print(f"归档路径(历史版): {RUN_DIR / NPZ_OUTPUT.name}")
    print(f"image_size = {img_size}")
    print(f"baseline = {baseline_cm:.6f} cm")
    print(f"fx_rect = {fx_rect:.6f} px")
    print(f"swap_lr = {swap_air_params_lr}")
    print(f"swap_capture_halves = {SWAP_CAPTURE_HALVES}")
    print(f"理论 Z=50cm 时 disparity ≈ {fx_rect * baseline_cm / 50.0:.6f} px")
    print(f"理论 Z=100cm 时 disparity ≈ {fx_rect * baseline_cm / 100.0:.6f} px")
    print("Q =")
    print(Q)

    return baseline_cm, fx_rect


def write_calibration_report(
    capture_stats,
    rms_left,
    rms_right,
    stereo_rms,
    K1, K2, dist1, dist2, R, T_cm,
    baseline_cm,
    fx_rect,
    swap_air_params_lr,
    gate_stats=None,
):
    """
    把本次标定的完整信息写成 markdown 报告归档到 RUN_DIR。
    控制台输出用完即丢，报告是唯一的持久化记录。
    """
    dropped = capture_stats["dropped"]
    dropped_text = ", ".join(str(i) for i in dropped) if dropped else "无"

    if stereo_rms is None:
        stereo_rms_text = "未重估（RECALIBRATE_STEREO_RT=False，沿用空气 R/T）"
    else:
        stereo_rms_text = f"{stereo_rms:.6f} px"

    if gate_stats is None:
        gate_stats = {"repaired": [], "dropped": []}
    gate_text = (
        f"- 角点 180° 翻转自动找回组: {gate_stats['repaired'] if gate_stats['repaired'] else '无'}\n"
        f"- 门控剔除组: {gate_stats['dropped'] if gate_stats['dropped'] else '无'}"
    )

    lines = [
        "# 空气+玻璃标定报告",
        "",
        f"- 运行时间戳: {PROGRAM_START_TIMESTAMP}",
        "- 阶段: 空气+平板玻璃",
        f"- 输入源: {MAT_INPUT}",
        f"- 归档目录: {RUN_DIR}",
        "",
        "## 配置快照",
        "",
        f"- PATTERN_SIZE = {PATTERN_SIZE}（内角点数）",
        f"- SQUARE_SIZE = {SQUARE_SIZE} cm",
        f"- CAMERA_ID = {CAMERA_ID}",
        f"- IMG_SIZE = {IMG_SIZE}",
        f"- SWAP_CAPTURE_HALVES = {SWAP_CAPTURE_HALVES}",
        f"- RECALIBRATE_STEREO_RT = {RECALIBRATE_STEREO_RT}",
        f"- RECTIFY_ALPHA = {RECTIFY_ALPHA}",
        f"- MATLAB_T_UNIT = {MATLAB_T_UNIT}",
        f"- swap_air_params_lr(实际生效) = {swap_air_params_lr}",
        f"- USE_WLS = {USE_WLS}",
        "",
        "## 采集统计",
        "",
        f"- 缓存组数: {capture_stats['cached']}",
        f"- 角点检测成功（计入标定）: {capture_stats['valid']}",
        f"- 被丢弃的组编号: {dropped_text}",
        "",
        "## 拟合指标",
        "",
        f"- 单目固定 K RMS（左）: {rms_left:.6f} px",
        f"- 单目固定 K RMS（右）: {rms_right:.6f} px",
        f"- 双目 stereo RMS: {stereo_rms_text}",
        gate_text,
        "",
        "## 最终参数",
        "",
        "```",
        f"K1 =\n{np.array2string(np.asarray(K1), precision=6)}",
        f"K2 =\n{np.array2string(np.asarray(K2), precision=6)}",
        f"dist1 = {np.array2string(np.asarray(dist1).reshape(-1), precision=6)}",
        f"dist2 = {np.array2string(np.asarray(dist2).reshape(-1), precision=6)}",
        f"R =\n{np.array2string(np.asarray(R), precision=6)}",
        f"T_cm = {np.array2string(np.asarray(T_cm).reshape(-1), precision=6)}",
        "```",
        "",
        f"- baseline = {baseline_cm:.6f} cm",
        f"- fx_rect = {fx_rect:.6f} px",
        f"- swap_lr = {swap_air_params_lr}",
        f"- swap_capture_halves = {SWAP_CAPTURE_HALVES}",
        "",
        "## 理论视差核对",
        "",
        f"- Z=30cm → disparity ≈ {fx_rect * baseline_cm / 30.0:.4f} px",
        f"- Z=50cm → disparity ≈ {fx_rect * baseline_cm / 50.0:.4f} px",
        f"- Z=100cm → disparity ≈ {fx_rect * baseline_cm / 100.0:.4f} px",
        "",
    ]

    report_path = RUN_DIR / "calibration_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 已归档标定报告: {report_path}")


# =========================================================
# 5. 测深阶段 s 键采样：真实深度输入弹窗 + 采样版 Runner
# =========================================================
def ask_true_depth_cm():
    """
    弹出 tkinter 输入框，让用户输入测点的实测真实深度（cm）。
    用户取消或直接关闭弹窗时返回 None。
    tkinter 在函数内延迟导入，避免不用此功能时的额外开销。
    """
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        value = simpledialog.askfloat(
            title="输入真实深度",
            prompt="请输入该测点的实测真实深度 Z（cm）：",
            parent=root,
            minvalue=0.0,
        )
    finally:
        root.destroy()
    return value


class SamplingStereoDepthRunner(StereoDepthRunner):
    """
    在 StereoDepthRunner 基础上增加测深阶段的 s 键采样动作：

    按下 s 时（要求当前已点击待测点且该点视差有效，否则仅警告、不保存）：
    1. 保存当前帧的原始左右图（未校正、未标注，可事后离线复算）；
    2. 弹窗输入实测真实深度，取消则整条放弃；
    3. 把图像文件名、像素坐标、测量值（disp/X/Y/Z/距离）、真实值和误差
       追加写入 sample_dir/depth_samples.json。
    """

    def __init__(self, *args, sample_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dir = Path(sample_dir)
        self.sample_json_path = self.sample_dir / "depth_samples.json"

    def save_depth_sample(self, left, right, point, result):
        """s 键动作的实际落盘逻辑。"""
        if point is None:
            print("⚠️ 当前没有待检测的点：请先在校正图左半边点击目标点，再按 s 保存样本。")
            return

        if result is None:
            print("⚠️ 该点本帧无有效深度（视差无效），本条样本不保存。")
            return

        true_z = ask_true_depth_cm()
        if true_z is None:
            print("⚠️ 已取消输入真实深度，本条样本整条放弃（未保存图像和数据）。")
            return

        self.sample_dir.mkdir(parents=True, exist_ok=True)

        records = []
        if self.sample_json_path.exists():
            try:
                with open(self.sample_json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except ValueError:
                print(f"⚠️ {self.sample_json_path} 解析失败，忽略旧记录重新计数（将被覆写）。")
                records = []

        index = len(records) + 1
        left_name = f"left_{index:03d}.png"
        right_name = f"right_{index:03d}.png"

        ok_l = imwrite_unicode(self.sample_dir / left_name, left)
        ok_r = imwrite_unicode(self.sample_dir / right_name, right)
        if not (ok_l and ok_r):
            print(f"⚠️ 样本图像落盘失败（left={ok_l}, right={ok_r}），本条记录不写入 json。")
            return

        err = result["Z"] - true_z
        err_rate = abs(err) / true_z * 100.0

        records.append({
            "index": index,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "left_image": left_name,
            "right_image": right_name,
            "pixel": {"x": int(point[0]), "y": int(point[1])},
            "measured": {
                "disp_px": result["disp"],
                "X_cm": result["X"],
                "Y_cm": result["Y"],
                "Z_cm": result["Z"],
                "dist_cm": result["D"],
                "valid_count": result["valid_count"],
            },
            "true_z_cm": true_z,
            "error_cm": err,
            "error_rate_percent": err_rate,
        })

        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(
            f"✅ 已保存深度样本 #{index}: 测量 Z={result['Z']:.3f} cm, "
            f"真实 Z={true_z:.3f} cm, 误差 {err:+.3f} cm ({err_rate:.2f}%)"
        )
        print(f"   样本目录: {self.sample_dir}")

    def run(self):
        """
        覆写基类 run()：主循环与基类一致，仅增加 s 键采样分支。
        点击点的全局量在 stereo_depth_core 模块内，必须通过模块属性读写。
        """
        # 复位进程内可能残留的上一次点击点
        stereo_depth_core.clicked_point = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.disparity_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            self.window_name,
            stereo_depth_core.on_mouse,
            {"left_width": self.left_width, "height": self.height},
        )

        print("\n========== 测深阶段按键 ==========")
        print("  鼠标左键：在校正图左半边点击选择待测点，实时显示深度")
        print("  s：保存深度验证样本（当前左右原图 + 测量数据 + 弹窗输入真实深度）")
        print("  q / ESC：退出")
        print(f"  样本保存目录: {self.sample_dir}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("读取摄像头失败")
                break

            left, right = self.split_frame(frame)
            rect_left, rect_right = self.rectify(left, right)
            disparity = self.compute_disparity(rect_left, rect_right)
            points_3d = self.reproject(disparity)

            show_left = rect_left.copy()
            show_right = rect_right.copy()

            point = stereo_depth_core.clicked_point
            result = None
            if point is not None:
                x, y = point
                if 0 <= x < self.left_width and 0 <= y < self.height:
                    result = self.measure_depth(disparity, points_3d, x, y)
                    show_left = self.draw_measurement(show_left, point, result)
                    self.print_measurement(point, result)

            for yy in range(0, self.height, 40):
                cv2.line(show_left, (0, yy), (self.left_width, yy), (255, 0, 0), 1)
                cv2.line(show_right, (0, yy), (self.left_width, yy), (255, 0, 0), 1)

            pair_show = cv2.hconcat([show_left, show_right])
            disp_show = self.disparity_to_color(disparity)

            cv2.imshow(self.window_name, pair_show)
            cv2.imshow(self.disparity_window_name, disp_show)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                # left/right 是喂给校正的原始左右图；result 是本帧该测点的测量结果
                self.save_depth_sample(left, right, point, result)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # 在任何标定计算之前创建本次标定的归档目录（原图/角点/参数/报告都放这里）。
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"本次玻璃标定归档目录: {RUN_DIR}")

    # =====================================================
    # Step 0：读取空气标定参数
    # =====================================================
    K1_air, K2_air, dist1_air, dist2_air, R_air, T_air_raw, img_size_from_mat = load_air_stereo_params(MAT_INPUT)

    print("✅ 已读取空气标定参数")
    print(f"MAT 中图像尺寸 img_size = {img_size_from_mat}")
    print(f"当前脚本单目尺寸 IMG_SIZE = {IMG_SIZE}")

    if img_size_from_mat != IMG_SIZE:
        print("⚠️ 注意：MAT 中的单目尺寸和当前摄像头切分尺寸不一致。")
        print("   如果你的实际单目图确实是 640x480，而 MAT 不是这个尺寸，后面会不准。")

    print("K1_air =")
    print(K1_air)
    print("dist1_air =")
    print(dist1_air)
    print("K2_air =")
    print(K2_air)
    print("dist2_air =")
    print(dist2_air)
    print("R_air =")
    print(R_air)
    print("T_air_raw =")
    print(T_air_raw)

    swap_air_params_lr = resolve_air_param_swap()

    if swap_air_params_lr:
        K1_air, K2_air, dist1_air, dist2_air, R_air, T_air_raw = swap_left_right(
            K1_air,
            K2_air,
            dist1_air,
            dist2_air,
            R_air,
            T_air_raw
        )
        print("⚠️ 已按 立体校正.py 的 SWAP_LR=True 交换空气参数。")
        print("交换后的 K1_air/K2_air/R_air/T_air_raw 将用于玻璃标定。")
    else:
        print("✅ 空气参数不交换：使用 stereo_params_plain.mat 的原始左右顺序。")

    # =====================================================
    # Step 1：采集空气+玻璃角点
    # =====================================================
    objpoints, imgpoints_left, imgpoints_right, capture_stats = collect_glass_corners()

    # =====================================================
    # Step 2：固定 K，只估计左右玻璃畸变
    # =====================================================
    K1_glass, dist1_glass, rms1, _, _ = calibrate_one_camera_fixed_K(
        objpoints=objpoints,
        imgpoints=imgpoints_left,
        K_air=K1_air,
        dist_air=dist1_air,
        img_size=IMG_SIZE,
        camera_name="LEFT"
    )

    K2_glass, dist2_glass, rms2, _, _ = calibrate_one_camera_fixed_K(
        objpoints=objpoints,
        imgpoints=imgpoints_right,
        K_air=K2_air,
        dist_air=dist2_air,
        img_size=IMG_SIZE,
        camera_name="RIGHT"
    )

    # =====================================================
    # Step 3：处理双目 R/T（带逐图离群门控）
    # =====================================================
    stereo_rms = None
    gate_stereo = None

    if RECALIBRATE_STEREO_RT:
        # 推荐：用玻璃下角点重新估计 R/T。
        # 因为 SQUARE_SIZE 是 cm，所以这里 T_glass_cm 单位就是 cm。
        R_use, T_use_cm, stereo_rms, gate_stereo = stereo_calibrate_fixed_intrinsic(
            objpoints=objpoints,
            imgpoints_left=imgpoints_left,
            imgpoints_right=imgpoints_right,
            K1=K1_glass,
            dist1=dist1_glass,
            K2=K2_glass,
            dist2=dist2_glass,
            img_size=IMG_SIZE,
            baseline_expect_cm=float(np.linalg.norm(convert_matlab_T_to_cm(T_air_raw)))
        )
    else:
        # 只更新畸变，沿用空气中 R/T。
        # 这时要把 MATLAB 的 T 转成 cm。
        R_use = R_air.copy()
        T_use_cm = convert_matlab_T_to_cm(T_air_raw)

        print("\n" + "=" * 70)
        print("沿用空气中 R/T")
        print("=" * 70)
        print("R_air =")
        print(R_use)
        print("T_use_cm =")
        print(T_use_cm)
        print(f"baseline = {np.linalg.norm(T_use_cm):.6f} cm")

    # =====================================================
    # Step 4：生成 stereoRectify map 和 Q，保存 npz（根目录最新版 + 历史归档）
    # =====================================================
    baseline_cm, fx_rect = build_rectify_and_save_npz(
        K1=K1_glass,
        dist1=dist1_glass,
        K2=K2_glass,
        dist2=dist2_glass,
        R=R_use,
        T_cm=T_use_cm,
        img_size=IMG_SIZE,
        swap_air_params_lr=swap_air_params_lr
    )

    # =====================================================
    # Step 4.5：归档本次标定报告
    # =====================================================
    write_calibration_report(
        capture_stats=capture_stats,
        rms_left=rms1,
        rms_right=rms2,
        stereo_rms=stereo_rms,
        K1=K1_glass,
        K2=K2_glass,
        dist1=dist1_glass,
        dist2=dist2_glass,
        R=R_use,
        T_cm=T_use_cm,
        baseline_cm=baseline_cm,
        fx_rect=fx_rect,
        swap_air_params_lr=swap_air_params_lr,
        gate_stats=gate_stereo,
    )

    # =====================================================
    # Step 5：直接启动测深
    # =====================================================
    if RUN_DEPTH_AFTER_CALIB:
        print("\n========== 启动空气+玻璃补偿后的实时测深 ==========")
        runner = SamplingStereoDepthRunner(
            param_file=NPZ_OUTPUT,
            camera_id=CAMERA_ID,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            use_wls=USE_WLS,
            swap_capture_halves=SWAP_CAPTURE_HALVES,
            true_z_cm=None,
            window_name="rectified_pair_with_glass",
            disparity_window_name="disparity_with_glass",
            sample_dir=DEPTH_SAMPLES_DIR,
        )
        runner.run()


if __name__ == "__main__":
    main()
