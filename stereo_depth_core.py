# -*- coding: utf-8 -*-
"""
stereo_depth_core.py

统一的双目测距核心代码。

测距链路：
1. 读取 立体校正.py 生成的 stereo_params_updated.npz；
2. 使用 npz 中的 map1x/map1y/map2x/map2y 做校正；
3. 使用 npz 中的 Q 做 reprojectImageTo3D；
4. Q 已经是 cm 单位，因此这里不再使用 UNIT_SCALE。
"""

from pathlib import Path
import cv2
import numpy as np

clicked_point = None


def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        left_width = int(param["left_width"])
        height = int(param["height"])
        if 0 <= x < left_width and 0 <= y < height:
            clicked_point = (x, y)
            print(f"点击坐标: ({x}, {y})")
        else:
            print("请点击左侧校正图像区域")


class StereoDepthRunner:
    def __init__(
        self,
        param_file,
        camera_id=0,
        frame_width=1280,
        frame_height=480,
        use_wls=True,
        swap_capture_halves=False,
        true_z_cm=None,
        window_name="rectified_pair",
        disparity_window_name="disparity",
    ):
        self.param_file = Path(param_file)
        self.camera_id = camera_id
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.use_wls_requested = bool(use_wls)
        self.use_wls = bool(use_wls)
        self.swap_capture_halves = bool(swap_capture_halves)
        self.true_z_cm = true_z_cm
        self.window_name = window_name
        self.disparity_window_name = disparity_window_name

        if not self.param_file.exists():
            raise FileNotFoundError(f"参数文件不存在: {self.param_file}")

        self.load_params()
        self.create_matcher()
        self.open_camera()

    def load_params(self):
        data = np.load(str(self.param_file), allow_pickle=False)

        required = ["map1x", "map1y", "map2x", "map2y", "Q", "image_size"]
        for key in required:
            if key not in data:
                raise KeyError(f"npz 参数文件缺少 key: {key}")

        self.map1x = data["map1x"]
        self.map1y = data["map1y"]
        self.map2x = data["map2x"]
        self.map2y = data["map2y"]
        self.Q = data["Q"].astype(np.float64, copy=False)

        img_size = tuple(np.asarray(data["image_size"]).reshape(-1).astype(int))
        if len(img_size) != 2:
            raise ValueError(f"image_size 应该为 [width, height]，但读到: {img_size}")
        self.left_width = int(img_size[0])
        self.height = int(img_size[1])

        self.P1 = data["P1"] if "P1" in data else None
        self.P2 = data["P2"] if "P2" in data else None
        self.T_cm = data["T_cm"] if "T_cm" in data else None
        self.baseline_cm = float(np.asarray(data["baseline_cm"])) if "baseline_cm" in data else None
        self.fx_rect = float(np.asarray(data["fx_rect"])) if "fx_rect" in data else None
        self.swap_lr_in_params = bool(np.asarray(data["swap_lr"])) if "swap_lr" in data else None
        self.swap_capture_halves_in_params = (
            bool(np.asarray(data["swap_capture_halves"]))
            if "swap_capture_halves" in data
            else None
        )

        print(f"✅ 已加载参数文件: {self.param_file}")
        print(f"✅ 单目校正尺寸: {self.left_width} x {self.height}")
        if self.baseline_cm is not None:
            print(f"✅ baseline: {self.baseline_cm:.8f} cm")
        if self.fx_rect is not None and self.baseline_cm is not None:
            print(f"参考：Z=50cm 时理论 disparity ≈ {self.fx_rect * self.baseline_cm / 50.0:.4f} px")
            print(f"参考：Z=100cm 时理论 disparity ≈ {self.fx_rect * self.baseline_cm / 100.0:.4f} px")
        if self.swap_lr_in_params is not None:
            print(f"参数生成时 SWAP_LR = {self.swap_lr_in_params}")
        if self.swap_capture_halves_in_params is not None:
            print(f"参数生成时 SWAP_CAPTURE_HALVES = {self.swap_capture_halves_in_params}")
            if self.swap_capture_halves_in_params != self.swap_capture_halves:
                print("⚠️ 当前运行时 SWAP_CAPTURE_HALVES 和参数生成时不一致，画面可能会错位或花屏。")

    def create_matcher(self):
        min_disp = 0
        num_disp = 16 * 8
        block_size = 5

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.right_matcher = None
        self.wls_filter = None

        if self.use_wls_requested and hasattr(cv2, "ximgproc"):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(80000)
            self.wls_filter.setSigmaColor(1.3)
            self.use_wls = True
            print("✅ 已启用 WLS 视差滤波")
        elif self.use_wls_requested and not hasattr(cv2, "ximgproc"):
            self.use_wls = False
            print("⚠️ 当前 OpenCV 没有 ximgproc，自动改用原始 SGBM。")
            print("   如需 WLS：python -m pip uninstall opencv-python opencv-contrib-python -y")
            print("             python -m pip install opencv-contrib-python")
        else:
            self.use_wls = False
            print("✅ 使用原始 SGBM 视差")

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ 摄像头已打开: {actual_w} x {actual_h}")

        expected_w = self.left_width * 2
        expected_h = self.height
        if actual_w != expected_w or actual_h != expected_h:
            print(f"⚠️ 注意：摄像头实际分辨率不是期望的 {expected_w} x {expected_h}")
            print("   如果实际画面不是左右各一半拼接，split_frame 会切错，测距会不准。")

    def split_frame(self, frame):
        h, w = frame.shape[:2]
        mid = w // 2
        first = frame[:, :mid]
        second = frame[:, mid:]
        if self.swap_capture_halves:
            return second, first
        return first, second

    def rectify(self, left, right):
        if left.shape[1] != self.left_width or left.shape[0] != self.height:
            raise RuntimeError(f"左图尺寸错误: {left.shape[:2]}，期望 {(self.height, self.left_width)}")
        if right.shape[1] != self.left_width or right.shape[0] != self.height:
            raise RuntimeError(f"右图尺寸错误: {right.shape[:2]}，期望 {(self.height, self.left_width)}")

        rect_left = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def compute_disparity(self, rect_left, rect_right):
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        if self.use_wls:
            disp_left = self.left_matcher.compute(gray_left, gray_right)
            disp_right = self.right_matcher.compute(gray_right, gray_left)
            filtered = self.wls_filter.filter(disp_left, gray_left, None, disp_right)
            disparity = filtered.astype(np.float32) / 16.0
        else:
            disparity = self.left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity

    def reproject(self, disparity):
        return cv2.reprojectImageTo3D(disparity, self.Q)

    @staticmethod
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

    def measure_depth(self, disparity, points_3d, x, y, win=4):
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

        if np.count_nonzero(valid) == 0:
            return None

        disp = float(np.median(disp_region[valid]))
        X = float(np.median(pts_region[valid, 0]))
        Y = float(np.median(pts_region[valid, 1]))
        Z_raw = float(np.median(pts_region[valid, 2]))
        Z = abs(Z_raw)
        distance = float(np.sqrt(X * X + Y * Y + Z * Z))

        if Z <= 0 or Z > 1000:
            return None

        return {
            "disp": disp,
            "X": X,
            "Y": Y,
            "Z": Z,
            "Z_raw": Z_raw,
            "D": distance,
            "valid_count": int(np.count_nonzero(valid)),
        }

    def draw_measurement(self, img, point, result):
        x, y = point
        vis = img.copy()
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        if result is None:
            cv2.putText(vis, "Invalid depth", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        disp = result["disp"]
        X = result["X"]
        Y = result["Y"]
        Z = result["Z"]
        D = result["D"]

        cv2.putText(vis, f"Disp: {disp:.2f}px", (x + 10, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(vis, f"X={X:.2f} Y={Y:.2f} cm", (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(vis, f"Z={Z:.2f} cm", (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(vis, f"Dist={D:.2f} cm", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if self.true_z_cm is not None:
            err = Z - float(self.true_z_cm)
            err_rate = abs(err) / float(self.true_z_cm) * 100.0
            cv2.putText(vis, f"Err={err:.2f}cm {err_rate:.2f}%", (x + 10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        return vis

    def print_measurement(self, point, result):
        x, y = point
        if result is None:
            print(f"[点击点] pixel=({x},{y}) 无有效视差/深度")
            return

        msg = (
            f"[点击点] pixel=({x},{y}), "
            f"disp={result['disp']:.4f}px, "
            f"X={result['X']:.4f}cm, "
            f"Y={result['Y']:.4f}cm, "
            f"Z={result['Z']:.4f}cm, "
            f"Dist={result['D']:.4f}cm, "
            f"valid={result['valid_count']}"
        )
        if self.true_z_cm is not None:
            err = result["Z"] - float(self.true_z_cm)
            err_rate = abs(err) / float(self.true_z_cm) * 100.0
            msg += f", trueZ={float(self.true_z_cm):.4f}cm, err={err:.4f}cm, err_rate={err_rate:.2f}%"
        print(msg)

    def run(self):
        global clicked_point

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.disparity_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            self.window_name,
            on_mouse,
            {"left_width": self.left_width, "height": self.height},
        )

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

            if clicked_point is not None:
                x, y = clicked_point
                if 0 <= x < self.left_width and 0 <= y < self.height:
                    result = self.measure_depth(disparity, points_3d, x, y)
                    show_left = self.draw_measurement(show_left, clicked_point, result)
                    self.print_measurement(clicked_point, result)

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

        self.cap.release()
        cv2.destroyAllWindows()
