# -*- coding: utf-8 -*-
"""
rect_underwater_calib_and_depth.py

下水标定 + 下水测深入口。

核心思想：
1. 读取已经验证没问题的玻璃参数 stereo_params_updated_with_glass.npz；
2. 在真实下水状态下采集左右原图并按时间戳归档，结束后统一检测角点；
3. 固定玻璃模型中的 K1/K2，只重新最大似然估计水下 dist1/dist2；
4. 固定水下 K/dist 后，重新最大似然估计水下双目 R/T；
5. 重新 stereoRectify，生成水下专用 map 和 Q；
6. 保存 stereo_params_updated_underwater.npz；
7. 调用 SamplingStereoDepthRunner（StereoDepthRunner 子类）测深，
   测深阶段可按 s 保存深度验证样本。

注意：
- 不使用 1.33 经验缩放；
- Q 仍然输出 cm；
- SQUARE_SIZE 用 cm，则水下 T 和 Q 都是 cm；
- 下水测试时，玻璃/外壳/相机相对位置必须和采集标定时一致。

两个阶段的按键动作：
【阶段一：标定采集】（窗口 underwater_left_detect / underwater_right_detect / underwater_pair_detect）
  s      缓存并落盘当前左右原图（采集阶段不实时检测角点，避免卡顿）
  q/ESC  结束采集，随后统一批量检测角点并执行标定计算
【阶段二：测深】（窗口 rectified_pair_underwater / disparity_underwater）
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

import stereo_depth_core
from stereo_depth_core import StereoDepthRunner


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 每次程序启动都用独立时间戳目录归档本次标定的全部产物
# （原图 / 角点 npz / 参数 npz / 标定报告），只增不覆盖，便于追溯历史。
# 根目录下的 UNDERWATER_NPZ_OUTPUT / UNDERWATER_CORNERS_OUTPUT 仍是"最新版"语义（继续覆盖），
# 供测距脚本按固定路径读取。
PROGRAM_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_ROOT = SCRIPT_DIR / "calibration_history" / "underwater"
RUN_DIR = HISTORY_ROOT / PROGRAM_START_TIMESTAMP
UNDERWATER_IMAGES_OUTPUT = RUN_DIR / "images"

# 输入：你现在已经测玻璃没问题的参数
GLASS_NPZ_INPUT = SCRIPT_DIR / "stereo_params_updated_with_glass.npz"

# 输出：下水后重新最大似然估计得到的水下参数
UNDERWATER_NPZ_OUTPUT = SCRIPT_DIR / "stereo_params_updated_underwater.npz"

# 保存水下角点，方便以后不重新采集
UNDERWATER_CORNERS_OUTPUT = SCRIPT_DIR / "underwater_chessboard_corners.npz"

# 测深阶段按 s 保存的深度验证样本目录（原始左右图 + depth_samples.json）
DEPTH_SAMPLES_DIR = RUN_DIR / "depth_samples"


# =========================================================
# 2. 棋盘格配置
# =========================================================
# 内角点数，必须和你玻璃标定时一致
PATTERN_SIZE = (8, 11)

# 每个小方格真实边长，单位 cm
# 如果你的格子是 1 cm，就保持 1.0
SQUARE_SIZE = 1.0


# =========================================================
# 3. 摄像头配置
# =========================================================
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

SINGLE_WIDTH = FRAME_WIDTH // 2
SINGLE_HEIGHT = FRAME_HEIGHT
IMG_SIZE = (SINGLE_WIDTH, SINGLE_HEIGHT)

# 必须和你玻璃标定/测深时保持一致
SWAP_CAPTURE_HALVES = True


# =========================================================
# 4. 下水标定配置
# =========================================================
# True：推荐。下水后重新估计 R/T
# False：只更新左右畸变，沿用玻璃 R/T
RECALIBRATE_STEREO_RT = True

# stereoRectify alpha
# 0：裁黑边，适合测深
# 1：保留全视野，黑边多
RECTIFY_ALPHA = 0

# 标定完成后是否直接启动水下测深
RUN_DEPTH_AFTER_CALIB = True

# 测深是否使用 WLS
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


def normalize_dist_5(dist):
    """
    统一畸变参数为 OpenCV 5 参数：
    [k1, k2, p1, p2, k3]
    """
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    if dist.size >= 5:
        return dist[:5].copy()

    out = np.zeros(5, dtype=np.float64)
    out[:dist.size] = dist
    return out


def load_glass_params(npz_path: Path):
    """
    读取已经验证过的玻璃参数。
    这里用玻璃参数作为水下最大似然估计的初值。
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到玻璃参数文件: {npz_path}")

    data = np.load(str(npz_path), allow_pickle=False)

    required = ["K1", "K2", "dist1", "dist2", "R", "T_cm", "image_size"]
    for key in required:
        if key not in data:
            raise KeyError(f"玻璃参数 npz 缺少 key: {key}")

    K1 = np.asarray(data["K1"], dtype=np.float64)
    K2 = np.asarray(data["K2"], dtype=np.float64)
    dist1 = normalize_dist_5(data["dist1"])
    dist2 = normalize_dist_5(data["dist2"])
    R = np.asarray(data["R"], dtype=np.float64)
    T_cm = np.asarray(data["T_cm"], dtype=np.float64).reshape(3, 1)
    img_size = tuple(np.asarray(data["image_size"]).reshape(-1).astype(int))

    swap_capture_halves = (
        bool(np.asarray(data["swap_capture_halves"]))
        if "swap_capture_halves" in data
        else None
    )

    print("=" * 70)
    print("✅ 已读取玻璃参数，作为水下 MLE 初值")
    print(f"glass npz: {npz_path}")
    print(f"image_size = {img_size}")
    print("K1_glass =")
    print(K1)
    print("dist1_glass =")
    print(dist1)
    print("K2_glass =")
    print(K2)
    print("dist2_glass =")
    print(dist2)
    print("R_glass =")
    print(R)
    print("T_glass_cm =")
    print(T_cm)
    print(f"baseline_glass = {np.linalg.norm(T_cm):.8f} cm")

    if swap_capture_halves is not None:
        print(f"玻璃参数生成时 SWAP_CAPTURE_HALVES = {swap_capture_halves}")
        if swap_capture_halves != SWAP_CAPTURE_HALVES:
            print("⚠️ 当前 SWAP_CAPTURE_HALVES 与玻璃参数生成时不一致，请确认。")

    print("=" * 70)

    return K1, K2, dist1, dist2, R, T_cm, img_size


def split_frame(frame):
    h, w = frame.shape[:2]
    mid = w // 2

    first = frame[:, :mid]
    second = frame[:, mid:]

    if SWAP_CAPTURE_HALVES:
        return second, first

    return first, second


def enhance_for_underwater_chessboard(gray):
    """
    水下图像容易低对比、偏色、浑浊。
    这里只对灰度做 CLAHE，避免改变几何。
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_corners(img_bgr, pattern_size):
    """
    检测水下棋盘格角点。
    优先 findChessboardCornersSB，失败后传统方法兜底。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = enhance_for_underwater_chessboard(gray)

    ret, corners = cv2.findChessboardCornersSB(
        gray_eq,
        pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )

    if ret:
        corners = corners.astype(np.float32)
        return True, corners.reshape(-1, 2)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret, corners = cv2.findChessboardCorners(gray_eq, pattern_size, flags)

    if ret:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            60,
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


def make_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def collect_underwater_corners():
    """
    与 rect.py 采集模式对齐：
    先采集并缓存下水状态下的左右原图（同时落盘归档），采集结束后再统一检测角点。

    采集阶段不做棋盘格检测，避免实时画面卡顿。
    """
    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    cached_pairs = []      # (left, right, left文件名, right文件名)
    dropped_indices = []   # 角点检测失败被丢弃的缓存组编号（从 1 开始）
    valid_left_files = []
    valid_right_files = []

    left_output = UNDERWATER_IMAGES_OUTPUT / "left"
    right_output = UNDERWATER_IMAGES_OUTPUT / "right"
    pair_output = UNDERWATER_IMAGES_OUTPUT / "pair"
    left_output.mkdir(parents=True, exist_ok=True)
    right_output.mkdir(parents=True, exist_ok=True)
    pair_output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"摄像头实际分辨率: {actual_w} x {actual_h}")
    if actual_w != FRAME_WIDTH or actual_h != FRAME_HEIGHT:
        print(f"⚠️ 实际分辨率不是期望的 {FRAME_WIDTH}x{FRAME_HEIGHT}，切图可能错误。")

    cv2.namedWindow("underwater_left_detect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("underwater_right_detect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("underwater_pair_detect", cv2.WINDOW_NORMAL)

    print("\n========== 采集下水棋盘格角点 ==========")
    print("操作：")
    print("  s：缓存并落盘当前左右原图")
    print("  q / ESC：结束采集，然后统一检测角点")
    print("建议：至少保存 20~40 组。")
    print("姿态要求：近/中/远、左/中/右、上/中/下、轻微俯仰/偏航都要有。")
    print("注意：棋盘格必须真实在水中，且玻璃/外壳/相机位置不能再变。")
    print(f"当前 PATTERN_SIZE = {PATTERN_SIZE}")
    print(f"当前 SQUARE_SIZE = {SQUARE_SIZE} cm")
    print(f"标定原图保存目录: {UNDERWATER_IMAGES_OUTPUT}\n")

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

        cv2.imshow("underwater_left_detect", vis_l)
        cv2.imshow("underwater_right_detect", vis_r)
        cv2.imshow("underwater_pair_detect", pair_vis)

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
                print(f"✅ 已缓存并落盘第 {idx} 组水下原图")
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
            f"有效角点组数太少：{len(objpoints)}。建议至少 20~40 组，否则水下 MLE 不稳。"
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
    np.savez(str(UNDERWATER_CORNERS_OUTPUT), **corners_data)
    np.savez(str(RUN_DIR / UNDERWATER_CORNERS_OUTPUT.name), **corners_data)

    print(f"✅ 已保存水下角点数据: {UNDERWATER_CORNERS_OUTPUT}")
    print(f"✅ 已归档本次角点数据: {RUN_DIR / UNDERWATER_CORNERS_OUTPUT.name}")
    print(f"✅ 已保存水下标定原图: {UNDERWATER_IMAGES_OUTPUT}")

    capture_stats = {
        "cached": len(cached_pairs),
        "valid": len(objpoints),
        "dropped": dropped_indices,
    }

    return objpoints, imgpoints_left, imgpoints_right, capture_stats


def calibrate_one_camera_fixed_K(objpoints, imgpoints, K_init, dist_init, img_size, camera_name):
    """
    固定 K，只最大似然估计水下畸变参数。

    OpenCV calibrateCamera 最小化重投影误差平方和；
    在角点误差为独立同方差高斯噪声时，等价于最大似然估计。
    """
    K0 = K_init.copy().astype(np.float64)
    dist0 = normalize_dist_5(dist_init)

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
        120,
        1e-7
    )

    rms, K_new, dist_new, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=K0,
        distCoeffs=dist0,
        flags=flags,
        criteria=criteria
    )

    dist_new = normalize_dist_5(dist_new)

    print("\n" + "=" * 70)
    print(f"{camera_name} 水下单目标定结果")
    print("=" * 70)
    print(f"RMS reprojection error = {rms:.6f} px")
    print("K_init =")
    print(K_init)
    print("K_new =")
    print(K_new)
    print("说明：这里固定了焦距和光心，所以 K_new 理论上应基本等于 K_init。")
    print(f"dist_init [{camera_name}] =")
    print(dist_init)
    print(f"dist_underwater [{camera_name}] =")
    print(dist_new)

    return K_new, dist_new, rms


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
    固定左右 K/dist，只最大似然估计水下双目 R/T。

    因为 objpoints 单位是 cm，所以 T 单位也是 cm。
    """
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        120,
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
    print("水下双目标定 R/T 结果")
    print("=" * 70)
    print(f"Stereo RMS reprojection error = {rms:.6f} px")
    print("R_underwater =")
    print(R)
    print("T_underwater_cm =")
    print(T)
    print(f"baseline_underwater = {np.linalg.norm(T):.6f} cm")

    return R, T, rms


def build_rectify_and_save_npz(K1, dist1, K2, dist2, R, T_cm, img_size):
    """
    根据水下最大似然估计后的 K/dist/R/T 生成 map 和 Q。
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

        K1=K1,
        K2=K2,
        dist1=dist1,
        dist2=dist2,
        R=R,
        T=T_cm,
        T_cm=T_cm,

        baseline_cm=np.asarray(baseline_cm, dtype=np.float64),
        fx_rect=np.asarray(fx_rect, dtype=np.float64),
        rectify_alpha=np.asarray(RECTIFY_ALPHA, dtype=np.float64),
        swap_capture_halves=np.asarray(SWAP_CAPTURE_HALVES, dtype=np.bool_),
        medium=np.asarray("underwater"),
        source_param=np.asarray(str(GLASS_NPZ_INPUT)),
    )
    np.savez(str(UNDERWATER_NPZ_OUTPUT), **npz_data)
    np.savez(str(RUN_DIR / UNDERWATER_NPZ_OUTPUT.name), **npz_data)

    print("\n" + "=" * 70)
    print("✅ 已生成水下立体校正参数")
    print("=" * 70)
    print(f"保存路径(最新版): {UNDERWATER_NPZ_OUTPUT}")
    print(f"归档路径(历史版): {RUN_DIR / UNDERWATER_NPZ_OUTPUT.name}")
    print(f"image_size = {img_size}")
    print(f"baseline_underwater = {baseline_cm:.6f} cm")
    print(f"fx_rect = {fx_rect:.6f} px")
    print(f"rectify_alpha = {RECTIFY_ALPHA}")
    print(f"swap_capture_halves = {SWAP_CAPTURE_HALVES}")
    print(f"理论 Z=30cm 时 disparity ≈ {fx_rect * baseline_cm / 30.0:.6f} px")
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
):
    """
    把本次标定的完整信息写成 markdown 报告归档到 RUN_DIR。
    控制台输出用完即丢，报告是唯一的持久化记录。
    """
    dropped = capture_stats["dropped"]
    dropped_text = ", ".join(str(i) for i in dropped) if dropped else "无"

    if stereo_rms is None:
        stereo_rms_text = "未重估（RECALIBRATE_STEREO_RT=False，沿用玻璃 R/T）"
    else:
        stereo_rms_text = f"{stereo_rms:.6f} px"

    lines = [
        "# 水下标定报告",
        "",
        f"- 运行时间戳: {PROGRAM_START_TIMESTAMP}",
        "- 阶段: 水下（水+玻璃+外壳）",
        f"- 输入源: {GLASS_NPZ_INPUT}",
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
    print(f"本次水下标定归档目录: {RUN_DIR}")

    # Step 0：读取玻璃参数作为水下初值
    K1_glass, K2_glass, dist1_glass, dist2_glass, R_glass, T_glass_cm, img_size_from_npz = load_glass_params(
        GLASS_NPZ_INPUT
    )

    print(f"当前脚本 IMG_SIZE = {IMG_SIZE}")
    if img_size_from_npz != IMG_SIZE:
        print("⚠️ 注意：玻璃参数中的 image_size 和当前摄像头切分尺寸不一致。")
        print("   如果实际单目不是 640x480，后续会不准。")

    # Step 1：采集真实水下角点
    objpoints, imgpoints_left, imgpoints_right, capture_stats = collect_underwater_corners()

    # Step 2：固定 K，只估计水下左右畸变
    K1_underwater, dist1_underwater, rms1 = calibrate_one_camera_fixed_K(
        objpoints=objpoints,
        imgpoints=imgpoints_left,
        K_init=K1_glass,
        dist_init=dist1_glass,
        img_size=IMG_SIZE,
        camera_name="LEFT"
    )

    K2_underwater, dist2_underwater, rms2 = calibrate_one_camera_fixed_K(
        objpoints=objpoints,
        imgpoints=imgpoints_right,
        K_init=K2_glass,
        dist_init=dist2_glass,
        img_size=IMG_SIZE,
        camera_name="RIGHT"
    )

    # Step 3：重新估计水下 R/T
    stereo_rms = None

    if RECALIBRATE_STEREO_RT:
        R_use, T_use_cm, stereo_rms = stereo_calibrate_fixed_intrinsic(
            objpoints=objpoints,
            imgpoints_left=imgpoints_left,
            imgpoints_right=imgpoints_right,
            K1=K1_underwater,
            dist1=dist1_underwater,
            K2=K2_underwater,
            dist2=dist2_underwater,
            img_size=IMG_SIZE
        )
    else:
        R_use = R_glass.copy()
        T_use_cm = T_glass_cm.copy()
        print("\n" + "=" * 70)
        print("沿用玻璃 R/T，只更新水下畸变")
        print("=" * 70)
        print("R_glass =")
        print(R_use)
        print("T_glass_cm =")
        print(T_use_cm)
        print(f"baseline = {np.linalg.norm(T_use_cm):.6f} cm")

    # Step 4：生成水下 map/Q（根目录最新版 + 历史归档）
    baseline_cm, fx_rect = build_rectify_and_save_npz(
        K1=K1_underwater,
        dist1=dist1_underwater,
        K2=K2_underwater,
        dist2=dist2_underwater,
        R=R_use,
        T_cm=T_use_cm,
        img_size=IMG_SIZE
    )

    # Step 4.5：归档本次标定报告
    write_calibration_report(
        capture_stats=capture_stats,
        rms_left=rms1,
        rms_right=rms2,
        stereo_rms=stereo_rms,
        K1=K1_underwater,
        K2=K2_underwater,
        dist1=dist1_underwater,
        dist2=dist2_underwater,
        R=R_use,
        T_cm=T_use_cm,
        baseline_cm=baseline_cm,
        fx_rect=fx_rect,
    )

    # Step 5：直接启动水下测深
    if RUN_DEPTH_AFTER_CALIB:
        print("\n========== 启动水下实时测深 ==========")
        runner = SamplingStereoDepthRunner(
            param_file=UNDERWATER_NPZ_OUTPUT,
            camera_id=CAMERA_ID,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            use_wls=USE_WLS,
            swap_capture_halves=SWAP_CAPTURE_HALVES,
            true_z_cm=None,
            window_name="rectified_pair_underwater",
            disparity_window_name="disparity_underwater",
            sample_dir=DEPTH_SAMPLES_DIR,
        )
        runner.run()


if __name__ == "__main__":
    main()
