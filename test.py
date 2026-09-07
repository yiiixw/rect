# -*- coding: utf-8 -*-
"""
测试测距诊断.py

用途：
1. 读取 stereo_params_updated.npz；
2. 打印 baseline、fx_rect、rectify_alpha、swap_lr 等核心参数；
3. 实时显示校正后的左右图和视差图；
4. 鼠标点击左图，输出局部视差中值、视差标准差、Z_raw、Z_abs、误差；
5. 用于判断测距误差到底是参数问题还是视差匹配问题。

重要：
- 这个版本默认 Q 已经是 cm 单位；
- 不再使用 UNIT_SCALE；
- 调试阶段建议 USE_WLS=False。
"""

from pathlib import Path
import cv2
import numpy as np


# =========================
# 配置
# =========================
PARAM_FILE = Path(r"C:/Users/17434/Desktop/rect/stereo_params_updated.npz")

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# 调试阶段建议 False；确认 SGBM 正常后再试 True
USE_WLS = False

# 如果实时拼接画面左右半边本身反了，才设 True
# 你现在一般保持 False
SWAP_CAPTURE_HALVES = False

# 真实 Z 深度，单位 cm；不知道就设 None
TRUE_Z_CM = 35.0
# TRUE_Z_CM = None

# 点击点邻域窗口，win=5 表示 11x11
MEASURE_WIN = 5

# 有效点太少说明该区域视差不可靠
MIN_VALID_COUNT = 40

# 视差标准差阈值，越小越严格
MAX_DISP_STD = 3.0

clicked_point = None


def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        left_width = param["left_width"]
        height = param["height"]

        if 0 <= x < left_width and 0 <= y < height:
            clicked_point = (x, y)
            print(f"\n点击坐标: ({x}, {y})")
        else:
            print("请点击左侧校正图像区域")


def load_params(param_file):
    if not param_file.exists():
        raise FileNotFoundError(f"参数文件不存在: {param_file}")

    data = np.load(str(param_file), allow_pickle=False)

    required_keys = ["map1x", "map1y", "map2x", "map2y", "Q"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"参数文件缺少 key: {k}")

    map1x = data["map1x"]
    map1y = data["map1y"]
    map2x = data["map2x"]
    map2y = data["map2y"]
    Q = data["Q"].astype(np.float64)

    if "image_size" in data:
        image_size = tuple(np.asarray(data["image_size"]).reshape(-1).astype(int))
    elif "img_size" in data:
        image_size = tuple(np.asarray(data["img_size"]).reshape(-1).astype(int))
    else:
        image_size = (map1x.shape[1], map1x.shape[0])

    left_width, height = image_size

    fx_rect = float(np.asarray(data["fx_rect"])) if "fx_rect" in data else None
    baseline_cm = float(np.asarray(data["baseline_cm"])) if "baseline_cm" in data else None
    rectify_alpha = float(np.asarray(data["rectify_alpha"])) if "rectify_alpha" in data else None
    swap_lr = bool(np.asarray(data["swap_lr"])) if "swap_lr" in data else None

    P1 = data["P1"] if "P1" in data else None
    P2 = data["P2"] if "P2" in data else None

    print("=" * 70)
    print(f"✅ 已加载参数文件: {param_file}")
    print(f"✅ 单目尺寸: {left_width} x {height}")

    if rectify_alpha is not None:
        print(f"✅ rectify_alpha = {rectify_alpha}")

    if swap_lr is not None:
        print(f"✅ 参数生成时 SWAP_LR = {swap_lr}")

    if baseline_cm is not None:
        print(f"✅ baseline = {baseline_cm:.8f} cm")

    if fx_rect is not None:
        print(f"✅ fx_rect = {fx_rect:.8f} px")

    if fx_rect is not None and baseline_cm is not None:
        print("-" * 70)
        print(f"理论 Z=35cm 时 disparity ≈ {fx_rect * baseline_cm / 35.0:.4f} px")
        print(f"理论 Z=50cm 时 disparity ≈ {fx_rect * baseline_cm / 50.0:.4f} px")
        print(f"理论 Z=100cm 时 disparity ≈ {fx_rect * baseline_cm / 100.0:.4f} px")

        if TRUE_Z_CM is not None and float(TRUE_Z_CM) > 0:
            print(f"当前 TRUE_Z_CM={TRUE_Z_CM:.2f}cm，对应理论 disparity ≈ {fx_rect * baseline_cm / float(TRUE_Z_CM):.4f} px")

    if P1 is not None and P2 is not None:
        print("-" * 70)
        print(f"P1[0,0] = {P1[0, 0]:.8f}")
        print(f"P2[0,3] = {P2[0, 3]:.8f}")

    print("=" * 70)

    return {
        "map1x": map1x,
        "map1y": map1y,
        "map2x": map2x,
        "map2y": map2y,
        "Q": Q,
        "left_width": int(left_width),
        "height": int(height),
        "fx_rect": fx_rect,
        "baseline_cm": baseline_cm,
        "rectify_alpha": rectify_alpha,
        "swap_lr": swap_lr,
    }


def create_matcher(use_wls=False):
    min_disp = 0
    num_disp = 16 * 8

    # 原来是 5，先加大到 7，让平面区域更稳定
    block_size = 7

    # 你的输入是灰度图，所以通道数应该用 1，不建议继续用 3
    cn = 1

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,

        P1=8 * cn * block_size * block_size,
        P2=32 * cn * block_size * block_size,

        disp12MaxDiff=1,

        # 原来 10，稍微提高，减少错误匹配
        uniquenessRatio=15,

        # 加强小块错误视差过滤
        speckleWindowSize=150,
        speckleRange=2,

        preFilterCap=31,

        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    right_matcher = None
    wls_filter = None
    real_use_wls = False

    if use_wls:
        if hasattr(cv2, "ximgproc"):
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(30000)
            wls_filter.setSigmaColor(1.2)
            real_use_wls = True
            print("✅ 已启用 WLS")
        else:
            print("⚠️ 当前 OpenCV 没有 ximgproc，自动使用原始 SGBM")
    else:
        print("✅ 使用原始 SGBM")

    return left_matcher, right_matcher, wls_filter, real_use_wls


def split_frame(frame):
    h, w = frame.shape[:2]
    mid = w // 2

    first = frame[:, :mid]
    second = frame[:, mid:]

    if SWAP_CAPTURE_HALVES:
        return second, first

    return first, second


def compute_disparity(rect_left, rect_right, left_matcher, right_matcher, wls_filter, use_wls):
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    if use_wls:
        disp_left = left_matcher.compute(gray_left, gray_right)
        disp_right = right_matcher.compute(gray_right, gray_left)
        filtered = wls_filter.filter(disp_left, gray_left, None, disp_right)
        disparity = filtered.astype(np.float32) / 16.0
    else:
        disparity = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

    return disparity


def disparity_to_color(disparity):
    disp = disparity.copy()
    disp[~np.isfinite(disp)] = 0
    disp[disp < 0] = 0

    if np.max(disp) > np.min(disp):
        disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    else:
        disp = np.zeros_like(disp)

    disp = disp.astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def measure_point(disparity, points_3d, x, y, win=5):
    h, w = disparity.shape

    x0 = max(0, x - win)
    x1 = min(w, x + win + 1)
    y0 = max(0, y - win)
    y1 = min(h, y + win + 1)

    disp_region = disparity[y0:y1, x0:x1].reshape(-1)
    pts_region = points_3d[y0:y1, x0:x1].reshape(-1, 3)

    valid = (
        np.isfinite(disp_region)
        & (disp_region > 0)
        & np.isfinite(pts_region).all(axis=1)
    )

    valid_count = int(np.count_nonzero(valid))

    if valid_count == 0:
        return None

    disp_valid = disp_region[valid]
    pts_valid = pts_region[valid]

    disp_med = float(np.median(disp_valid))
    disp_mean = float(np.mean(disp_valid))
    disp_std = float(np.std(disp_valid))

    X = float(np.median(pts_valid[:, 0]))
    Y = float(np.median(pts_valid[:, 1]))
    Z_raw = float(np.median(pts_valid[:, 2]))
    Z_abs = abs(Z_raw)
    Dist = float(np.sqrt(X * X + Y * Y + Z_abs * Z_abs))

    reliable = True
    unreliable_reasons = []

    if valid_count < MIN_VALID_COUNT:
        reliable = False
        unreliable_reasons.append(f"valid太少({valid_count})")

    if disp_std > MAX_DISP_STD:
        reliable = False
        unreliable_reasons.append(f"disp_std过大({disp_std:.2f})")

    if Z_raw <= 0:
        reliable = False
        unreliable_reasons.append(f"Z_raw为负或零({Z_raw:.2f})")

    return {
        "disp_med": disp_med,
        "disp_mean": disp_mean,
        "disp_std": disp_std,
        "X": X,
        "Y": Y,
        "Z_raw": Z_raw,
        "Z_abs": Z_abs,
        "Dist": Dist,
        "valid_count": valid_count,
        "reliable": reliable,
        "reasons": unreliable_reasons,
    }


def draw_result(img, point, result):
    x, y = point
    vis = img.copy()

    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    if result is None:
        cv2.putText(
            vis,
            "Invalid",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return vis

    color = (0, 255, 0) if result["reliable"] else (0, 255, 255)

    cv2.putText(
        vis,
        f"Disp={result['disp_med']:.2f}px std={result['disp_std']:.2f}",
        (x + 10, y - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    cv2.putText(
        vis,
        f"Z_raw={result['Z_raw']:.2f}cm Z={result['Z_abs']:.2f}cm",
        (x + 10, y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    cv2.putText(
        vis,
        f"Dist={result['Dist']:.2f}cm valid={result['valid_count']}",
        (x + 10, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    if TRUE_Z_CM is not None and float(TRUE_Z_CM) > 0:
        err = result["Z_abs"] - float(TRUE_Z_CM)
        err_rate = abs(err) / float(TRUE_Z_CM) * 100.0
        cv2.putText(
            vis,
            f"Err={err:.2f}cm {err_rate:.2f}%",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if not result["reliable"]:
        cv2.putText(
            vis,
            "Unreliable",
            (x + 10, y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return vis


def print_result(point, result):
    x, y = point

    if result is None:
        print(f"[点击点] pixel=({x},{y}) 无有效视差")
        return

    msg = (
        f"[点击点] pixel=({x},{y}), "
        f"disp_med={result['disp_med']:.4f}px, "
        f"disp_mean={result['disp_mean']:.4f}px, "
        f"disp_std={result['disp_std']:.4f}px, "
        f"X={result['X']:.4f}cm, "
        f"Y={result['Y']:.4f}cm, "
        f"Z_raw={result['Z_raw']:.4f}cm, "
        f"Z_abs={result['Z_abs']:.4f}cm, "
        f"Dist={result['Dist']:.4f}cm, "
        f"valid={result['valid_count']}, "
        f"reliable={result['reliable']}"
    )

    if TRUE_Z_CM is not None and float(TRUE_Z_CM) > 0:
        err = result["Z_abs"] - float(TRUE_Z_CM)
        err_rate = abs(err) / float(TRUE_Z_CM) * 100.0
        msg += (
            f", trueZ={float(TRUE_Z_CM):.4f}cm, "
            f"err={err:.4f}cm, "
            f"err_rate={err_rate:.2f}%"
        )

    if not result["reliable"]:
        msg += f", reason={'; '.join(result['reasons'])}"

    print(msg)


def main():
    global clicked_point

    params = load_params(PARAM_FILE)

    left_width = params["left_width"]
    height = params["height"]

    expected_w = left_width * 2
    expected_h = height

    left_matcher, right_matcher, wls_filter, real_use_wls = create_matcher(USE_WLS)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✅ 摄像头实际分辨率: {actual_w} x {actual_h}")

    if actual_w != expected_w or actual_h != expected_h:
        raise RuntimeError(
            f"摄像头分辨率不匹配：实际 {actual_w}x{actual_h}，期望 {expected_w}x{expected_h}。"
            "如果这里不一致，左右切图一定会错。"
        )

    cv2.namedWindow("rectified_pair", cv2.WINDOW_NORMAL)
    cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback(
        "rectified_pair",
        on_mouse,
        {"left_width": left_width, "height": height},
    )

    print("\n操作说明：")
    print("  鼠标点击左侧校正图像，输出测距诊断")
    print("  q / ESC：退出")
    print("  s：保存当前校正图和视差图")
    print("-" * 70)

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("读取摄像头失败")
            break

        frame_idx += 1

        left, right = split_frame(frame)

        if left.shape[1] != left_width or left.shape[0] != height:
            raise RuntimeError(f"左图尺寸错误: {left.shape}, 期望: ({height}, {left_width}, 3)")
        if right.shape[1] != left_width or right.shape[0] != height:
            raise RuntimeError(f"右图尺寸错误: {right.shape}, 期望: ({height}, {left_width}, 3)")

        rect_left = cv2.remap(left, params["map1x"], params["map1y"], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, params["map2x"], params["map2y"], cv2.INTER_LINEAR)

        disparity = compute_disparity(
            rect_left,
            rect_right,
            left_matcher,
            right_matcher,
            wls_filter,
            real_use_wls,
        )

        points_3d = cv2.reprojectImageTo3D(disparity, params["Q"])

        show_left = rect_left.copy()
        show_right = rect_right.copy()

        # 画水平辅助线，用于观察校正是否行对齐
        for yy in range(0, height, 40):
            cv2.line(show_left, (0, yy), (left_width, yy), (255, 0, 0), 1)
            cv2.line(show_right, (0, yy), (left_width, yy), (255, 0, 0), 1)

        if clicked_point is not None:
            x, y = clicked_point

            if 0 <= x < left_width and 0 <= y < height:
                result = measure_point(disparity, points_3d, x, y, win=MEASURE_WIN)
                show_left = draw_result(show_left, clicked_point, result)
                print_result(clicked_point, result)

        pair_show = cv2.hconcat([show_left, show_right])
        disp_show = disparity_to_color(disparity)

        cv2.imshow("rectified_pair", pair_show)
        cv2.imshow("disparity", disp_show)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("s"), ord("S")):
            out_dir = Path(r"F:/Camera/debug_output")
            out_dir.mkdir(parents=True, exist_ok=True)

            pair_path = out_dir / f"pair_{frame_idx:06d}.png"
            disp_path = out_dir / f"disp_{frame_idx:06d}.png"
            left_path = out_dir / f"left_rect_{frame_idx:06d}.png"
            right_path = out_dir / f"right_rect_{frame_idx:06d}.png"

            cv2.imwrite(str(pair_path), pair_show)
            cv2.imwrite(str(disp_path), disp_show)
            cv2.imwrite(str(left_path), rect_left)
            cv2.imwrite(str(right_path), rect_right)

            print("已保存调试图像：")
            print(f"  {pair_path}")
            print(f"  {disp_path}")
            print(f"  {left_path}")
            print(f"  {right_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()