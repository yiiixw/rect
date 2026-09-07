# -*- coding: utf-8 -*-
"""
rect_water_article.py

按论文“三步标定法”的水下版本：

已完成：
1. 空气标定；
2. 空气 + 平板玻璃标定，得到 stereo_params_updated_with_glass.npz；

本脚本做：
3. 水下标定：
   - 加载空气+玻璃参数作为初值；
   - 水下有效焦距初值 = 玻璃焦距 * 1.33；
   - 允许 fx、fy 重新优化；
   - 可选固定主点 cx、cy；
   - 重新优化水下畸变 dist1/dist2；
   - 固定水下 K/dist 后重新 stereoCalibrate 得到水下 R/T；
   - stereoRectify 生成 map 和 Q；
   - 保存 stereo_params_updated_water_article.npz；
   - 直接进入水下测深。

注意：
- PATTERN_SIZE 是内角点数，不是黑白格子数。
- 如果你棋盘是横向 12 格、纵向 9 格，则内角点是 (11, 8)。
- SQUARE_SIZE 单位用 cm，则 T 和 Q 输出单位也是 cm。
"""

import cv2
import numpy as np
from pathlib import Path

from stereo_depth_core import StereoDepthRunner


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 输入：你前面空气+玻璃标定得到的参数
GLASS_NPZ = SCRIPT_DIR / "stereo_params_updated_with_glass.npz"

# 输出：水下论文版参数
WATER_NPZ_OUTPUT = SCRIPT_DIR / "stereo_params_updated_water_article.npz"

# 保存水下角点，方便复查
WATER_CORNERS_OUTPUT = SCRIPT_DIR / "water_article_chessboard_corners.npz"


# =========================================================
# 2. 摄像头配置
# =========================================================
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

SINGLE_WIDTH = FRAME_WIDTH // 2
SINGLE_HEIGHT = FRAME_HEIGHT
IMG_SIZE = (SINGLE_WIDTH, SINGLE_HEIGHT)

# 必须和你前面空气/玻璃采集、测深保持一致
SWAP_CAPTURE_HALVES = True


# =========================================================
# 3. 棋盘格配置
# =========================================================
# 你的 9*12 黑白格子，横向 12 格、纵向 9 格时，内角点为 (11, 8)
PATTERN_SIZE = (11, 8)

# 每个小格真实边长，单位 cm
SQUARE_SIZE = 1.0


# =========================================================
# 4. 论文水下焦距配置
# =========================================================
# 论文表 2 中水下焦距 / 空气焦距 ≈ 1.326~1.332
WATER_FOCAL_SCALE = 1.333

# True：只让 fx、fy 和畸变变，主点 cx、cy 固定；
# False：fx、fy、cx、cy 都允许变，可能更贴近论文表格，但更容易不稳定。
FIX_PRINCIPAL_POINT = True

# True：水下重新估计双目 R/T，符合论文“三步标定”路线；
# False：只更新水下 K/dist，沿用空气+玻璃 R/T。
RECALIBRATE_STEREO_RT = True

# 如果重新估计 R/T 后 baseline 偏离玻璃 baseline 太多，是否强制使用玻璃 baseline。
# 一般先 False，看输出的 baseline 是否合理。
FORCE_GLASS_RT_IF_BASELINE_DRIFT_TOO_LARGE = False

# baseline 漂移超过该比例就报警
BASELINE_DRIFT_WARN_RATIO = 0.15


# =========================================================
# 5. stereoRectify 配置
# =========================================================
# 0：裁黑边，适合测深；1：保留全视野
RECTIFY_ALPHA = 0

# 标定完成后是否直接进入测深
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
def normalize_dist_5(dist):
    """
    统一为 OpenCV 5参数畸变：
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
    加载空气+玻璃参数。
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到玻璃参数文件: {npz_path}")

    data = np.load(str(npz_path), allow_pickle=False)

    required = ["K1", "K2", "dist1", "dist2", "R"]
    for key in required:
        if key not in data:
            raise KeyError(f"玻璃参数文件缺少 key: {key}")

    K1 = np.asarray(data["K1"], dtype=np.float64)
    K2 = np.asarray(data["K2"], dtype=np.float64)
    dist1 = normalize_dist_5(data["dist1"])
    dist2 = normalize_dist_5(data["dist2"])
    R = np.asarray(data["R"], dtype=np.float64)

    if "T_cm" in data:
        T_cm = np.asarray(data["T_cm"], dtype=np.float64).reshape(3, 1)
    elif "T" in data:
        T_cm = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)
    else:
        raise KeyError("玻璃参数文件中没有 T_cm 或 T")

    image_size = None
    if "image_size" in data:
        image_size = tuple(np.asarray(data["image_size"]).reshape(-1).astype(int))

    print("=" * 80)
    print("✅ 已加载空气+玻璃参数")
    print(f"文件: {npz_path}")
    print(f"image_size in npz = {image_size}")
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
    print("=" * 80)

    return K1, K2, dist1, dist2, R, T_cm


def make_water_initial_K(K_glass, scale=WATER_FOCAL_SCALE):
    """
    按论文思路构造水下内参初值：
    fx_water_init = fx_glass * 1.33
    fy_water_init = fy_glass * 1.33
    cx, cy 初值沿用玻璃标定结果。
    """
    K = K_glass.copy().astype(np.float64)
    K[0, 0] *= scale
    K[1, 1] *= scale
    return K


def split_frame(frame):
    """
    切左右图。
    """
    h, w = frame.shape[:2]
    mid = w // 2

    first = frame[:, :mid]
    second = frame[:, mid:]

    if SWAP_CAPTURE_HALVES:
        return second, first

    return first, second


def preprocess_gray(gray):
    """
    水下棋盘格增强，只处理灰度，不改变几何。
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_corners(img_bgr):
    """
    检测棋盘格角点。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = preprocess_gray(gray)

    ret, corners = cv2.findChessboardCornersSB(
        gray_eq,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )

    if ret:
        return True, corners.astype(np.float32).reshape(-1, 2)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret, corners = cv2.findChessboardCorners(gray_eq, PATTERN_SIZE, flags)

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
        return True, corners.astype(np.float32).reshape(-1, 2)

    return False, None


def make_object_points():
    """
    生成棋盘格世界坐标，Z=0。
    单位由 SQUARE_SIZE 决定，这里是 cm。
    """
    cols, rows = PATTERN_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(SQUARE_SIZE)
    return objp


def open_camera():
    """
    打开摄像头。
    Windows 下使用 CAP_DSHOW 更稳。
    """
    print("正在打开摄像头...")

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"❌ 摄像头打不开: CAMERA_ID={CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✅ 摄像头已打开，实际分辨率: {actual_w} x {actual_h}")

    if actual_w != FRAME_WIDTH or actual_h != FRAME_HEIGHT:
        raise RuntimeError(
            f"❌ 摄像头分辨率错误：实际 {actual_w}x{actual_h}，期望 {FRAME_WIDTH}x{FRAME_HEIGHT}。\n"
            f"如果这里不一致，左右切图会错。"
        )

    return cap


def collect_water_corners():
    """
    水下采集左右棋盘格角点。
    按 s 检测并保存；按 q 结束。
    """
    cap = open_camera()

    objp = make_object_points()

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    cv2.namedWindow("water_left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("water_right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("water_pair", cv2.WINDOW_NORMAL)

    print("\n========== 水下棋盘格采集 ==========")
    print("操作：")
    print("  s：检测并保存当前左右角点")
    print("  q / ESC：结束采集")
    print("建议：保存 20~40 组，覆盖近/中/远、左/中/右、上/中/下、轻微倾斜姿态。")
    print(f"PATTERN_SIZE = {PATTERN_SIZE}")
    print(f"SQUARE_SIZE = {SQUARE_SIZE} cm")
    print("=" * 80)

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ 读取帧失败")
            continue

        frame_id += 1

        left, right = split_frame(frame)

        if left.shape[:2] != (SINGLE_HEIGHT, SINGLE_WIDTH):
            raise RuntimeError(f"左图尺寸错误: {left.shape}, 期望 {(SINGLE_HEIGHT, SINGLE_WIDTH)}")
        if right.shape[:2] != (SINGLE_HEIGHT, SINGLE_WIDTH):
            raise RuntimeError(f"右图尺寸错误: {right.shape}, 期望 {(SINGLE_HEIGHT, SINGLE_WIDTH)}")

        vis_l = left.copy()
        vis_r = right.copy()

        cv2.putText(
            vis_l,
            f"saved={len(objpoints)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )
        cv2.putText(
            vis_r,
            "press s to detect/save",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        pair = cv2.hconcat([vis_l, vis_r])
        cv2.imshow("water_left", vis_l)
        cv2.imshow("water_right", vis_r)
        cv2.imshow("water_pair", pair)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("s"), ord("S")):
            print("正在检测当前帧角点...")

            ok_l, corners_l = detect_corners(left)
            ok_r, corners_r = detect_corners(right)

            if ok_l and ok_r:
                objpoints.append(objp.copy())
                imgpoints_left.append(corners_l.reshape(-1, 1, 2).copy())
                imgpoints_right.append(corners_r.reshape(-1, 1, 2).copy())

                print(f"✅ 保存成功：{len(objpoints)} 组")

                draw_l = left.copy()
                draw_r = right.copy()

                cv2.drawChessboardCorners(
                    draw_l,
                    PATTERN_SIZE,
                    corners_l.reshape(-1, 1, 2),
                    True
                )
                cv2.drawChessboardCorners(
                    draw_r,
                    PATTERN_SIZE,
                    corners_r.reshape(-1, 1, 2),
                    True
                )

                cv2.imshow("water_left", draw_l)
                cv2.imshow("water_right", draw_r)
                cv2.waitKey(300)

            else:
                print(f"❌ 检测失败: left={ok_l}, right={ok_r}")
                print("   检查：棋盘格是否完整进入左右图；PATTERN_SIZE 是否应改为 (8, 11)。")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"❌ 有效角点组数太少：{len(objpoints)}。建议至少 20 组。")

    np.savez(
        str(WATER_CORNERS_OUTPUT),
        objpoints=np.asarray(objpoints, dtype=np.float32),
        imgpoints_left=np.asarray(imgpoints_left, dtype=np.float32),
        imgpoints_right=np.asarray(imgpoints_right, dtype=np.float32),
        pattern_size=np.asarray(PATTERN_SIZE, dtype=np.int32),
        square_size=np.asarray(SQUARE_SIZE, dtype=np.float32),
        image_size=np.asarray(IMG_SIZE, dtype=np.int32)
    )

    print(f"✅ 水下角点已保存: {WATER_CORNERS_OUTPUT}")

    return objpoints, imgpoints_left, imgpoints_right


def calibrate_water_camera_article(objpoints, imgpoints, K_init, dist_init, img_size, name):
    """
    论文版水下单目标定：

    与工程稳定版不同：
    - 不固定 fx/fy；
    - fx/fy 初值为玻璃焦距 * 1.33；
    - 允许 fx/fy 在水下重新优化；
    - 默认固定主点 cx/cy，避免不稳定。
    """
    K0 = K_init.copy().astype(np.float64)
    dist0 = normalize_dist_5(dist_init)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )

    if FIX_PRINCIPAL_POINT:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-8
    )

    rms, K_water, dist_water, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=K0,
        distCoeffs=dist0,
        flags=flags,
        criteria=criteria
    )

    dist_water = normalize_dist_5(dist_water)

    print("\n" + "=" * 80)
    print(f"{name} 水下单目标定结果（论文版）")
    print("=" * 80)
    print(f"RMS reprojection error = {rms:.6f} px")
    print("K_init =")
    print(K_init)
    print("K_water =")
    print(K_water)
    print(f"fx ratio = {K_water[0, 0] / K_init[0, 0]:.6f}")
    print(f"fy ratio = {K_water[1, 1] / K_init[1, 1]:.6f}")
    print("dist_init =")
    print(dist_init)
    print("dist_water =")
    print(dist_water)
    print("=" * 80)

    return K_water, dist_water, rms


def stereo_calibrate_water_article(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1_water,
    dist1_water,
    K2_water,
    dist2_water,
    img_size
):
    """
    使用水下 K/dist，重新估计水下 R/T。
    这一步对应文章中得到水下双目外参。
    """
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-8
    )

    rms, K1_out, dist1_out, K2_out, dist2_out, R_water, T_water, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_left,
        imagePoints2=imgpoints_right,
        cameraMatrix1=K1_water.copy(),
        distCoeffs1=dist1_water.copy(),
        cameraMatrix2=K2_water.copy(),
        distCoeffs2=dist2_water.copy(),
        imageSize=img_size,
        flags=flags,
        criteria=criteria
    )

    T_water = T_water.reshape(3, 1)

    print("\n" + "=" * 80)
    print("水下双目标定结果（论文版）")
    print("=" * 80)
    print(f"Stereo RMS = {rms:.6f} px")
    print("R_water =")
    print(R_water)
    print("T_water_cm =")
    print(T_water)
    print(f"baseline_water = {np.linalg.norm(T_water):.8f} cm")
    print("=" * 80)

    return R_water, T_water, rms


def build_rectify_and_save(
    K1_water,
    dist1_water,
    K2_water,
    dist2_water,
    R_water,
    T_water_cm,
    K1_glass,
    K2_glass,
    dist1_glass,
    dist2_glass,
    R_glass,
    T_glass_cm
):
    """
    用水下参数生成 stereoRectify 和 Q。
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1_water,
        distCoeffs1=dist1_water,
        cameraMatrix2=K2_water,
        distCoeffs2=dist2_water,
        imageSize=IMG_SIZE,
        R=R_water,
        T=T_water_cm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=RECTIFY_ALPHA
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1_water,
        dist1_water,
        R1,
        P1,
        IMG_SIZE,
        cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K2_water,
        dist2_water,
        R2,
        P2,
        IMG_SIZE,
        cv2.CV_32FC1
    )

    baseline_water = float(np.linalg.norm(T_water_cm))
    baseline_glass = float(np.linalg.norm(T_glass_cm))
    fx_rect = float(P1[0, 0])

    drift = abs(baseline_water - baseline_glass) / max(baseline_glass, 1e-9)

    if drift > BASELINE_DRIFT_WARN_RATIO:
        print("⚠️ 警告：水下重新估计出的 baseline 与玻璃 baseline 偏差较大")
        print(f"   baseline_glass = {baseline_glass:.8f} cm")
        print(f"   baseline_water = {baseline_water:.8f} cm")
        print(f"   drift = {drift * 100:.2f}%")

    np.savez(
        str(WATER_NPZ_OUTPUT),

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
        image_size=np.asarray(IMG_SIZE, dtype=np.int32),
        img_size=np.asarray(IMG_SIZE, dtype=np.int32),

        K1=K1_water,
        K2=K2_water,
        dist1=dist1_water,
        dist2=dist2_water,
        R=R_water,
        T=T_water_cm,
        T_cm=T_water_cm,

        K1_glass=K1_glass,
        K2_glass=K2_glass,
        dist1_glass=dist1_glass,
        dist2_glass=dist2_glass,
        R_glass=R_glass,
        T_glass_cm=T_glass_cm,

        baseline_cm=np.asarray(baseline_water, dtype=np.float64),
        baseline_glass_cm=np.asarray(baseline_glass, dtype=np.float64),
        fx_rect=np.asarray(fx_rect, dtype=np.float64),
        rectify_alpha=np.asarray(RECTIFY_ALPHA, dtype=np.float64),
        water_focal_scale_init=np.asarray(WATER_FOCAL_SCALE, dtype=np.float64),
        fix_principal_point=np.asarray(FIX_PRINCIPAL_POINT, dtype=np.bool_),
        swap_capture_halves=np.asarray(SWAP_CAPTURE_HALVES, dtype=np.bool_),
        medium=np.asarray("water_article_style")
    )

    print("\n" + "=" * 80)
    print("✅ 已保存论文版水下参数")
    print("=" * 80)
    print(f"保存路径: {WATER_NPZ_OUTPUT}")
    print(f"baseline_water = {baseline_water:.8f} cm")
    print(f"baseline_glass = {baseline_glass:.8f} cm")
    print(f"fx_rect = {fx_rect:.8f} px")
    print(f"理论 Z=35cm disparity ≈ {fx_rect * baseline_water / 35.0:.4f} px")
    print(f"理论 Z=50cm disparity ≈ {fx_rect * baseline_water / 50.0:.4f} px")
    print(f"理论 Z=100cm disparity ≈ {fx_rect * baseline_water / 100.0:.4f} px")
    print("Q =")
    print(Q)
    print("=" * 80)


def main():
    # 1. 加载空气+玻璃参数
    K1_glass, K2_glass, dist1_glass, dist2_glass, R_glass, T_glass_cm = load_glass_params(GLASS_NPZ)

    # 2. 按论文结果构造水下 K 初值，fx/fy 约乘 1.33
    K1_water_init = make_water_initial_K(K1_glass, WATER_FOCAL_SCALE)
    K2_water_init = make_water_initial_K(K2_glass, WATER_FOCAL_SCALE)

    print("\n" + "=" * 80)
    print("水下 K 初值")
    print("=" * 80)
    print("K1_water_init =")
    print(K1_water_init)
    print("K2_water_init =")
    print(K2_water_init)
    print(f"水下焦距初值比例 WATER_FOCAL_SCALE = {WATER_FOCAL_SCALE}")
    print("=" * 80)

    # 3. 水下采集棋盘格角点
    objpoints, imgpoints_left, imgpoints_right = collect_water_corners()

    # 4. 水下单目标定：允许 fx/fy 和畸变更新
    K1_water, dist1_water, rms1 = calibrate_water_camera_article(
        objpoints=objpoints,
        imgpoints=imgpoints_left,
        K_init=K1_water_init,
        dist_init=dist1_glass,
        img_size=IMG_SIZE,
        name="LEFT"
    )

    K2_water, dist2_water, rms2 = calibrate_water_camera_article(
        objpoints=objpoints,
        imgpoints=imgpoints_right,
        K_init=K2_water_init,
        dist_init=dist2_glass,
        img_size=IMG_SIZE,
        name="RIGHT"
    )

    # 5. 水下双目标定：用水下 K/dist 重新估计 R/T
    if RECALIBRATE_STEREO_RT:
        R_water, T_water_cm, stereo_rms = stereo_calibrate_water_article(
            objpoints=objpoints,
            imgpoints_left=imgpoints_left,
            imgpoints_right=imgpoints_right,
            K1_water=K1_water,
            dist1_water=dist1_water,
            K2_water=K2_water,
            dist2_water=dist2_water,
            img_size=IMG_SIZE
        )

        baseline_glass = float(np.linalg.norm(T_glass_cm))
        baseline_water = float(np.linalg.norm(T_water_cm))
        drift = abs(baseline_water - baseline_glass) / max(baseline_glass, 1e-9)

        if FORCE_GLASS_RT_IF_BASELINE_DRIFT_TOO_LARGE and drift > BASELINE_DRIFT_WARN_RATIO:
            print("⚠️ baseline 漂移过大，按配置强制使用玻璃 R/T")
            R_use = R_glass.copy()
            T_use = T_glass_cm.copy()
        else:
            R_use = R_water
            T_use = T_water_cm

    else:
        print("⚠️ 当前配置为不重新估计 R/T，沿用玻璃 R/T")
        R_use = R_glass.copy()
        T_use = T_glass_cm.copy()

    # 6. 生成水下立体校正参数
    build_rectify_and_save(
        K1_water=K1_water,
        dist1_water=dist1_water,
        K2_water=K2_water,
        dist2_water=dist2_water,
        R_water=R_use,
        T_water_cm=T_use,
        K1_glass=K1_glass,
        K2_glass=K2_glass,
        dist1_glass=dist1_glass,
        dist2_glass=dist2_glass,
        R_glass=R_glass,
        T_glass_cm=T_glass_cm
    )

    # 7. 直接启动水下测深
    if RUN_DEPTH_AFTER_CALIB:
        print("\n========== 启动论文版水下测深 ==========")

        app = StereoDepthRunner(
            param_file=WATER_NPZ_OUTPUT,
            camera_id=CAMERA_ID,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            use_wls=USE_WLS,
            swap_capture_halves=SWAP_CAPTURE_HALVES,
            true_z_cm=None,
            window_name="rectified_pair_water_article",
            disparity_window_name="disparity_water_article",
        )

        app.run()


if __name__ == "__main__":
    main()