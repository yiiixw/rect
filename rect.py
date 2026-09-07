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
7. 调用你已有的 StereoDepthRunner 直接测深。

重点：
- 焦距 fx/fy 和光心 cx/cy 固定不动；
- 主要更新 dist1_glass、dist2_glass；
- 双目测深必须左右相机都补偿；
- npz 必须包含 map1x/map1y/map2x/map2y/Q，否则你的测距核心无法运行。
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.io import loadmat

from stereo_depth_core import StereoDepthRunner


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 每次程序启动都使用独立时间戳目录保存玻璃标定原图。
PROGRAM_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GLASS_IMAGES_ROOT = SCRIPT_DIR / "glass_calibration_images"
GLASS_IMAGES_OUTPUT = GLASS_IMAGES_ROOT / PROGRAM_START_TIMESTAMP

MAT_INPUT = SCRIPT_DIR / "stereo_params_plain.mat"
NPZ_OUTPUT = SCRIPT_DIR / "stereo_params_updated_with_glass.npz"
AIR_RECTIFY_NPZ = SCRIPT_DIR / "stereo_params_updated.npz"

# 可选：保存采集到的角点，方便以后不重新采集
CORNERS_OUTPUT = SCRIPT_DIR / "glass_chessboard_corners.npz"


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
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 单目尺寸。你的双目是左右拼接，所以单目是 640x480。
SINGLE_WIDTH = FRAME_WIDTH // 2
SINGLE_HEIGHT = FRAME_HEIGHT
IMG_SIZE = (SINGLE_WIDTH, SINGLE_HEIGHT)  # OpenCV 顺序：width, height

# 当前摄像头拼接图的物理左右半边是反的，和 采集数据.py / 测量深度.py 保持一致。
SWAP_CAPTURE_HALVES = True


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
RECTIFY_ALPHA = 1

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


def detect_corners(img_bgr, pattern_size):
    """
    检测单张图里的棋盘格角点。
    优先使用 findChessboardCornersSB，失败再用传统 findChessboardCorners。
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
        return True, corners.reshape(-1, 2)

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
        return True, corners.reshape(-1, 2)

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
    cached_pairs = []

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
                cached_pairs.append((left.copy(), right.copy()))
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
    for pair_idx, (left, right) in enumerate(cached_pairs, start=1):
        ok_l, corners_l = detect_corners(left, PATTERN_SIZE)
        ok_r, corners_r = detect_corners(right, PATTERN_SIZE)

        if ok_l and ok_r:
            objpoints.append(objp.copy())
            imgpoints_left.append(corners_l.astype(np.float32).copy())
            imgpoints_right.append(corners_r.astype(np.float32).copy())
            print(f"✅ 第 {pair_idx}/{len(cached_pairs)} 组：左右角点检测成功")
        else:
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

    np.savez(
        str(CORNERS_OUTPUT),
        objpoints=np.asarray(objpoints, dtype=np.float32),
        imgpoints_left=np.asarray(imgpoints_left, dtype=np.float32),
        imgpoints_right=np.asarray(imgpoints_right, dtype=np.float32),
        pattern_size=np.asarray(PATTERN_SIZE, dtype=np.int32),
        square_size=np.asarray(SQUARE_SIZE, dtype=np.float32),
        image_size=np.asarray(IMG_SIZE, dtype=np.int32),
    )

    print(f"✅ 已保存空气+玻璃角点数据: {CORNERS_OUTPUT}")
    print(f"✅ 已保存空气+玻璃标定原图: {GLASS_IMAGES_OUTPUT}")

    return objpoints, imgpoints_left, imgpoints_right


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
    img_size
):
    """
    固定左右 K/dist，只重新估计双目 R/T。

    因为 objpoints 的单位是 cm，所以这里得到的 T 也是 cm。
    """
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7
    )

    rms, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_left,
        imagePoints2=imgpoints_right,
        cameraMatrix1=K1.copy(),
        distCoeffs1=dist1.copy(),
        cameraMatrix2=K2.copy(),
        distCoeffs2=dist2.copy(),
        imageSize=img_size,
        flags=flags,
        criteria=criteria
    )

    T = T.reshape(3, 1)

    print("\n" + "=" * 70)
    print("空气+玻璃双目标定 R/T 结果")
    print("=" * 70)
    print(f"Stereo RMS reprojection error = {rms:.6f} px")
    print("R_glass =")
    print(R)
    print("T_glass_cm =")
    print(T)
    print(f"baseline_glass = {np.linalg.norm(T):.6f} cm")

    return R, T, rms


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

    np.savez(
        str(NPZ_OUTPUT),

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

    print("\n" + "=" * 70)
    print("已生成空气+玻璃立体校正参数")
    print("=" * 70)
    print(f"保存路径: {NPZ_OUTPUT}")
    print(f"image_size = {img_size}")
    print(f"baseline = {baseline_cm:.6f} cm")
    print(f"fx_rect = {fx_rect:.6f} px")
    print(f"swap_lr = {swap_air_params_lr}")
    print(f"swap_capture_halves = {SWAP_CAPTURE_HALVES}")
    print(f"理论 Z=50cm 时 disparity ≈ {fx_rect * baseline_cm / 50.0:.6f} px")
    print(f"理论 Z=100cm 时 disparity ≈ {fx_rect * baseline_cm / 100.0:.6f} px")
    print("Q =")
    print(Q)


def main():
    # 在任何标定计算之前创建本次程序启动的图像保存目录。
    GLASS_IMAGES_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"本次玻璃标定图像保存目录: {GLASS_IMAGES_OUTPUT}")

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
    objpoints, imgpoints_left, imgpoints_right = collect_glass_corners()

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
    # Step 3：处理双目 R/T
    # =====================================================
    if RECALIBRATE_STEREO_RT:
        # 推荐：用玻璃下角点重新估计 R/T。
        # 因为 SQUARE_SIZE 是 cm，所以这里 T_glass_cm 单位就是 cm。
        R_use, T_use_cm, stereo_rms = stereo_calibrate_fixed_intrinsic(
            objpoints=objpoints,
            imgpoints_left=imgpoints_left,
            imgpoints_right=imgpoints_right,
            K1=K1_glass,
            dist1=dist1_glass,
            K2=K2_glass,
            dist2=dist2_glass,
            img_size=IMG_SIZE
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
    # Step 4：生成 stereoRectify map 和 Q，保存 npz
    # =====================================================
    build_rectify_and_save_npz(
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
    # Step 5：直接启动测深
    # =====================================================
    if RUN_DEPTH_AFTER_CALIB:
        print("\n========== 启动空气+玻璃补偿后的实时测深 ==========")
        runner = StereoDepthRunner(
            param_file=NPZ_OUTPUT,
            camera_id=CAMERA_ID,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            use_wls=USE_WLS,
            swap_capture_halves=SWAP_CAPTURE_HALVES,
            true_z_cm=None,
            window_name="rectified_pair_with_glass",
            disparity_window_name="disparity_with_glass",
        )
        runner.run()


if __name__ == "__main__":
    main()
