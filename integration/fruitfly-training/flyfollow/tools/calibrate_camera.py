"""R0 checkerboard calibration of the Tello camera (plan 3.4): fx, fy, cx, cy, distortion, HFOV and VFOV.

    python -m flyfollow.tools.calibrate_camera --live                    # capture about 20 views from the Tello
    python -m flyfollow.tools.calibrate_camera --frames data/r0/calib_frames_<stamp>
    python -m flyfollow.tools.calibrate_camera --synthetic               # self-test on rendered views, no drone

Board: --board 9x6 inner corners (a 10 x 7 squares chessboard), --square-mm 25. Print it flat (A4 or a screen at
100 % zoom; measure one square with a ruler and pass the real size). Writes configs/camera.json (the runtime and
update_sim_from_r0 read it) and a copy in data/r0/. Live mode wakes the Tello with raw UDP ("command", "streamon")
and reads video with OpenCV, so no PyAV/djitellopy in this process (latency does not matter here).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

from flyfollow.interfaces import IMG_H, IMG_W, configs_dir
from flyfollow.tools.stick_response import r0_dir


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


def find_corners(gray: np.ndarray, cols: int, rows: int):
    import cv2

    try:
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            return c.astype(np.float32)
    except AttributeError:
        pass
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))


def calibrate(corners: list[np.ndarray], cols: int, rows: int, square_m: float, size=(IMG_W, IMG_H), prune: bool = True) -> dict:
    """cv2.calibrateCamera on the detected views; with prune, drops views whose error is > 2x the median and refits."""
    import cv2

    obj = board_points(cols, rows, square_m)
    views = list(range(len(corners)))
    for _ in range(2):
        rms, K, dist, rv, tv = cv2.calibrateCamera([obj] * len(views), [corners[i] for i in views], size, None, None)
        errs = []
        for j, i in enumerate(views):
            pr, _ = cv2.projectPoints(obj, rv[j], tv[j], K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - corners[i].reshape(-1, 2)) ** 2, axis=1)))))
        med = float(np.median(errs))
        keep = [i for i, e in zip(views, errs) if e <= 2.0 * med]
        if not prune or len(keep) == len(views) or len(keep) < 8:
            break
        views = keep
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    W, H = size
    return {
        "img_w": W, "img_h": H, "fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2), "cy": round(cy, 2),
        "dist": [round(float(d), 5) for d in dist.ravel()[:5]],
        "hfov_deg": round(math.degrees(2 * math.atan(W / (2 * fx))), 2),
        "vfov_deg": round(math.degrees(2 * math.atan(H / (2 * fy))), 2),
        "dfov_deg": round(math.degrees(2 * math.atan(math.hypot(W / fx, H / fy) / 2)), 2),
        "rms_px": round(float(rms), 3), "n_views": len(views), "n_views_detected": len(corners),
        "view_err_px": [round(e, 3) for e in errs], "board": f"{cols}x{rows}", "square_m": square_m,
    }


def write_result(res: dict, source: str, out: Path | None = None) -> Path:
    res = dict(res, source=source, stamp=time.strftime("%Y%m%d_%H%M%S"))
    out = out or configs_dir() / "camera.json"
    out.write_text(json.dumps(res, indent=1))
    (r0_dir() / f"camera_{res['stamp']}.json").write_text(json.dumps(res, indent=1))
    return out


def report(res: dict) -> str:
    return (f"fx {res['fx']:.1f} fy {res['fy']:.1f} cx {res['cx']:.1f} cy {res['cy']:.1f} | HFOV {res['hfov_deg']:.1f} deg, "
            f"VFOV {res['vfov_deg']:.1f} deg, DFOV {res['dfov_deg']:.1f} deg | rms {res['rms_px']:.2f} px over "
            f"{res['n_views']} views | dist {res['dist']}\n(plan 3.4 assumed fx 921 fy 919 -> HFOV 55, VFOV 43; "
            f"rms above 1 px means blurry views or a board that is not flat)")


# ------------------------------------------------------------------------------------------------ sources
def from_folder(folder: str, cols: int, rows: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    import cv2

    files = sorted(glob.glob(str(Path(folder) / "*.png")) + glob.glob(str(Path(folder) / "*.jpg")))
    corners, size = [], (IMG_W, IMG_H)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        c = find_corners(img, cols, rows)
        print(f"{Path(f).name}: {'board found' if c is not None else 'no board'}")
        if c is not None:
            corners.append(c)
    return corners, size


def synthetic_views(cols: int, rows: int, square_m: float, n: int = 16, fx: float = 921.0, fy: float = 919.0,
                    seed: int = 0) -> list[np.ndarray]:
    """Render a chessboard with a known pinhole camera at n random poses (for the self-test)."""
    import cv2

    rng = np.random.default_rng(seed)
    K = np.array([[fx, 0, IMG_W / 2 + 4], [0, fy, IMG_H / 2 - 3], [0, 0, 1]])
    sq = 60
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    board = np.full((bh + 2 * sq, bw + 2 * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[sq + r * sq: sq + (r + 1) * sq, sq + c * sq: sq + (c + 1) * sq] = 0
    # board pixel -> board meters: corner (0, 0) of the pattern's inner grid sits at pixel (2 sq, 2 sq)
    m_per_px = square_m / sq
    out = []
    while len(out) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        dist_m = rng.uniform(0.5, 0.9)
        tvec = np.array([rng.uniform(-0.12, 0.05), rng.uniform(-0.08, 0.04), dist_m])
        R, _ = cv2.Rodrigues(rvec)
        src = np.array([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]], np.float32)
        pts3 = np.array([[(x - 2 * sq) * m_per_px, (y - 2 * sq) * m_per_px, 0.0] for x, y in src])
        pr, _ = cv2.projectPoints(pts3, rvec, tvec, K, None)
        if np.any((R @ pts3.T + tvec[:, None])[2] < 0.1):
            continue
        Hm = cv2.getPerspectiveTransform(src, pr.reshape(-1, 2).astype(np.float32))
        img = cv2.warpPerspective(board, Hm, (IMG_W, IMG_H), borderValue=170)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        c = find_corners(img, cols, rows)
        if c is not None:
            out.append(c)
    return out


def tello_wake(ip: str) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3.0)
    try:
        s.bind(("", 8889))
    except OSError as e:
        sys.exit(f"cannot bind UDP 8889 ({e}): another Tello process is running (lsof -nP -iUDP:8889)")
    try:
        for cmd in ("command", "streamon"):
            s.sendto(cmd.encode(), (ip, 8889))
            try:
                print(cmd, "->", s.recvfrom(1024)[0].decode(errors="replace"))
            except TimeoutError:
                sys.exit(f"no reply to '{cmd}': is the laptop on the TELLO-XXXXXX Wi-Fi?")
            time.sleep(0.3)
    finally:
        s.close()


def live(a, cols: int, rows: int) -> tuple[list[np.ndarray], str]:
    import cv2

    print("""
R0 CAMERA CALIBRATION (drone does not fly):
 1. Laptop on the TELLO-XXXXXX Wi-Fi; no tello_io running. Board flat, well lit, no glare.
 2. Hold the Tello (or the board) so the whole board is in view at 0.4 to 1.2 m. Vary distance, tilt (up to about
    40 deg) and position: corners and edges of the image matter most for distortion.
 3. Keys in the video window: SPACE capture (only when the board is found, green), a toggles auto-capture
    (every 1 s when the board has moved), c calibrate (>= 12 views, aim for 20), q quit.
""")
    tello_wake(a.ip)
    cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
    t0 = time.time()
    while not cap.isOpened() and time.time() - t0 < 10:
        time.sleep(0.2)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save = r0_dir() / f"calib_frames_{stamp}"
    save.mkdir(parents=True, exist_ok=True)
    corners: list[np.ndarray] = []
    auto, last_auto, last_c = False, 0.0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        ok_s, _ = cv2.findChessboardCorners(small, (cols, rows), cv2.CALIB_CB_FAST_CHECK)
        c = find_corners(gray, cols, rows) if ok_s else None
        vis = frame.copy()
        if c is not None:
            cv2.drawChessboardCorners(vis, (cols, rows), c, True)
        cv2.putText(vis, f"views {len(corners)}  auto {'ON' if auto else 'off'}  {'BOARD' if c is not None else 'no board'}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if c is not None else (0, 0, 255), 2)
        cv2.imshow("calibrate_camera", vis)
        k = cv2.waitKey(1) & 0xFF
        grab = k == ord(" ") and c is not None
        if auto and c is not None and time.time() - last_auto > 1.0:
            moved = last_c is None or float(np.linalg.norm(c.reshape(-1, 2).mean(0) - last_c.reshape(-1, 2).mean(0))) > 40
            grab = grab or moved
        if grab:
            corners.append(c)
            last_c, last_auto = c, time.time()
            cv2.imwrite(str(save / f"view_{len(corners):02d}.png"), frame)
            print(f"captured view {len(corners)}")
        if k == ord("a"):
            auto = not auto
        if k == ord("q") or (k == ord("c") and len(corners) >= 12):
            break
    cap.release()
    cv2.destroyAllWindows()
    return corners, str(save)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="R0 checkerboard calibration of the Tello camera")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true")
    g.add_argument("--frames", metavar="DIR", help="folder of saved 960x720 frames (png/jpg)")
    g.add_argument("--synthetic", action="store_true", help="self-test on rendered views (writes nothing)")
    ap.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--ip", default="192.168.10.1")
    ap.add_argument("--out", default=None, help="default configs/camera.json")
    a = ap.parse_args(argv)
    cols, rows = (int(x) for x in a.board.lower().split("x"))
    sq = a.square_mm / 1000.0
    size = (IMG_W, IMG_H)
    if a.synthetic:
        corners, src = synthetic_views(cols, rows, sq), "synthetic"
    elif a.frames:
        corners, size = from_folder(a.frames, cols, rows)
        src = a.frames
    else:
        corners, src = live(a, cols, rows)
    if len(corners) < 8:
        sys.exit(f"only {len(corners)} views with the board: need at least 8 (20 is better)")
    if size != (IMG_W, IMG_H):
        print(f"WARNING: frames are {size[0]}x{size[1]}, not the Tello's {IMG_W}x{IMG_H}")
    res = calibrate(corners, cols, rows, sq, size)
    print(report(res))
    if a.synthetic:
        print("synthetic truth: fx 921 fy 919 cx 484 cy 357")
        return
    out = write_result(res, src, Path(a.out) if a.out else None)
    print(f"wrote {out}\nnext: python -m flyfollow.tools.update_sim_from_r0 (fx_px in profiles.demo) and send settings "
          f"fx={res['fx']} fy={res['fy']} cx={res['cx']} cy={res['cy']} to the runtime")


if __name__ == "__main__":
    main()
