# -*- coding: utf-8 -*-
import cv2
from pathlib import Path

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

SAVE_ROOT = Path(r"D:/rect")
LEFT_DIR = SAVE_ROOT / "left"
RIGHT_DIR = SAVE_ROOT / "right"
PAIR_DIR = SAVE_ROOT / "pair"

LEFT_DIR.mkdir(parents=True, exist_ok=True)
RIGHT_DIR.mkdir(parents=True, exist_ok=True)
PAIR_DIR.mkdir(parents=True, exist_ok=True)

# 如果实时拼接画面左半边和右半边本身反了，才改 True
SWAP_CAPTURE_HALVES = True


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def split_frame(frame):
    h, w = frame.shape[:2]
    mid = w // 2

    first = frame[:, :mid]
    second = frame[:, mid:]

    if SWAP_CAPTURE_HALVES:
        return second, first

    return first, second


def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"摄像头实际分辨率: {actual_w} x {actual_h}")

    if actual_w != FRAME_WIDTH or actual_h != FRAME_HEIGHT:
        raise RuntimeError(
            f"实际分辨率 {actual_w}x{actual_h} 与期望 {FRAME_WIDTH}x{FRAME_HEIGHT} 不一致"
        )

    idx = 1

    cv2.namedWindow("capture_pair", cv2.WINDOW_NORMAL)

    print("操作说明：")
    print("  s：保存当前左右图")
    print("  q / ESC：退出")
    print("保存路径：")
    print(f"  left:  {LEFT_DIR}")
    print(f"  right: {RIGHT_DIR}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取图像失败")
            break

        left, right = split_frame(frame)

        show = cv2.hconcat([left, right])

        cv2.putText(show, f"idx={idx:03d}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for yy in range(0, show.shape[0], 40):
            cv2.line(show, (0, yy), (show.shape[1], yy), (255, 0, 0), 1)

        cv2.imshow("capture_pair", show)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("s"), ord("S")):
            left_path = LEFT_DIR / f"left_{idx:03d}.png"
            right_path = RIGHT_DIR / f"right_{idx:03d}.png"
            pair_path = PAIR_DIR / f"pair_{idx:03d}.png"

            ok1 = imwrite_unicode(left_path, left)
            ok2 = imwrite_unicode(right_path, right)
            ok3 = imwrite_unicode(pair_path, show)

            if ok1 and ok2 and ok3:
                print(f"已保存第 {idx:03d} 组")
                print(f"  {left_path}")
                print(f"  {right_path}")
                idx += 1
            else:
                print("保存失败")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()