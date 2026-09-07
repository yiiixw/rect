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
7. 直接调用 StereoDepthRunner 测深。

注意：
- 不使用 1.33 经验缩放；
- Q 仍然输出 cm；
- SQUARE_SIZE 用 cm，则水下 T 和 Q 都是 cm；
- 下水测试时，玻璃/外壳/相机相对位置必须和采集标定时一致。
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from stereo_depth_core import StereoDepthRunner


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 每次程序启动都使用独立时间戳目录保存水下标定原图。
PROGRAM_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
UNDERWATER_IMAGES_ROOT = SCRIPT_DIR / "underwater_calibration_images"
UNDERWATER_IMAGES_OUTPUT = UNDERWATER_IMAGES_ROOT / PROGRAM_START_TIMESTAMP

# 输入：你现在已经测玻璃没问题的参数
GLASS_NPZ_INPUT = SCRIPT_DIR / "stereo_params_updated_with_glass.npz"

# 输出：下水后重新最大似然估计得到的水下参数
UNDERWATER_NPZ_OUTPUT = SCRIPT_DIR / "stereo_params_updated_underwater.npz"

# 保存水下角点，方便以后不重新采集
UNDERWATER_CORNERS_OUTPUT = SCRIPT_DIR / "underwater_chessboard_corners.npz"


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
CAMERA_ID = 0
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
    cached_pairs = []

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
                cached_pairs.append((left.copy(), right.copy()))
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
            f"有效角点组数太少：{len(objpoints)}。建议至少 20~40 组，否则水下 MLE 不稳。"
        )

    objpoints = [p.astype(np.float32) for p in objpoints]
    imgpoints_left = [p.astype(np.float32) for p in imgpoints_left]
    imgpoints_right = [p.astype(np.float32) for p in imgpoints_right]

    np.savez(
        str(UNDERWATER_CORNERS_OUTPUT),
        objpoints=np.asarray(objpoints, dtype=np.float32),
        imgpoints_left=np.asarray(imgpoints_left, dtype=np.float32),
        imgpoints_right=np.asarray(imgpoints_right, dtype=np.float32),
        pattern_size=np.asarray(PATTERN_SIZE, dtype=np.int32),
        square_size=np.asarray(SQUARE_SIZE, dtype=np.float32),
        image_size=np.asarray(IMG_SIZE, dtype=np.int32),
    )

    print(f"✅ 已保存水下角点数据: {UNDERWATER_CORNERS_OUTPUT}")
    print(f"✅ 已保存水下标定原图: {UNDERWATER_IMAGES_OUTPUT}")

    return objpoints, imgpoints_left, imgpoints_right


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

    np.savez(
        str(UNDERWATER_NPZ_OUTPUT),

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

    print("\n" + "=" * 70)
    print("✅ 已生成水下立体校正参数")
    print("=" * 70)
    print(f"保存路径: {UNDERWATER_NPZ_OUTPUT}")
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


def main():
    # 在任何标定计算之前创建本次程序启动的图像保存目录。
    UNDERWATER_IMAGES_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"本次水下标定图像保存目录: {UNDERWATER_IMAGES_OUTPUT}")

    # Step 0：读取玻璃参数作为水下初值
    K1_glass, K2_glass, dist1_glass, dist2_glass, R_glass, T_glass_cm, img_size_from_npz = load_glass_params(
        GLASS_NPZ_INPUT
    )

    print(f"当前脚本 IMG_SIZE = {IMG_SIZE}")
    if img_size_from_npz != IMG_SIZE:
        print("⚠️ 注意：玻璃参数中的 image_size 和当前摄像头切分尺寸不一致。")
        print("   如果实际单目不是 640x480，后续会不准。")

    # Step 1：采集真实水下角点
    objpoints, imgpoints_left, imgpoints_right = collect_underwater_corners()

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

    # Step 4：生成水下 map/Q
    build_rectify_and_save_npz(
        K1=K1_underwater,
        dist1=dist1_underwater,
        K2=K2_underwater,
        dist2=dist2_underwater,
        R=R_use,
        T_cm=T_use_cm,
        img_size=IMG_SIZE
    )

    # Step 5：直接启动水下测深
    if RUN_DEPTH_AFTER_CALIB:
        print("\n========== 启动水下实时测深 ==========")
        runner = StereoDepthRunner(
            param_file=UNDERWATER_NPZ_OUTPUT,
            camera_id=CAMERA_ID,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            use_wls=USE_WLS,
            swap_capture_halves=SWAP_CAPTURE_HALVES,
            true_z_cm=None,
            window_name="rectified_pair_underwater",
            disparity_window_name="disparity_underwater",
        )
        runner.run()


if __name__ == "__main__":
    main()
