"""Live YOLO on the Tello camera. The drone stays on the ground: this only connects and streams video.

    cd rewrite
    python tests/yolo_live.py --detect person
    python tests/yolo_live.py --detect bottle

Join the TELLO-xxxxxx Wi-Fi first, and close the Tello phone app (only one program can receive the video).
The window shows the detections, inference time, processed frames per second and battery.
Quit with q or Esc in the window, or Ctrl+C in the terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import av
import cv2
from djitellopy import Tello, TelloException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rewrite/, for `import yolo`
from yolo import DETECTORS, draw  # noqa: E402

WINDOW = "Tello YOLO"
PLACEHOLDER_HW = (300, 400)  # djitellopy's black frame before the first decoded one (Tello frames are 720x960)
VIDEO_OPEN_TIMEOUT_S = 10  # the decoder's open can wait this long for the first keyframe
VIDEO_OPEN_ATTEMPTS = 3
STALE_S = 1.0  # no new frame for this long: say so in the window


def start_video(tello: Tello):
    """Start the stream and return djitellopy's frame reader (frame = newest decoded frame, RGB)."""
    try:
        tello.streamoff()  # a stream left on by an earlier run restarts, so a fresh keyframe comes soon
    except TelloException:
        pass
    tello.streamon()
    Tello.FRAME_GRAB_TIMEOUT = VIDEO_OPEN_TIMEOUT_S
    for attempt in range(1, VIDEO_OPEN_ATTEMPTS + 1):
        try:
            return tello.get_frame_read()
        except (TelloException, av.error.FFmpegError) as e:
            print(f"video open attempt {attempt}/{VIDEO_OPEN_ATTEMPTS} failed: {e}")
    raise SystemExit("No video from the Tello. Close the Tello app and any other program using UDP port 11111.")


def put_text(img, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def battery(tello: Tello) -> str:
    try:
        return f"{tello.get_battery()} %"
    except TelloException:
        return "?"


def run(tello: Tello, detector, name: str) -> None:
    print("connecting to the Tello ...")
    tello.connect()
    print(f"connected, battery {battery(tello)}")
    reader = start_video(tello)
    print("waiting for the first video frame ...")

    last = None  # the last frame object processed (djitellopy makes a new array for every decoded frame)
    shown = None  # the last annotated image
    t_start = t_new = time.time()
    fps = 0.0
    stale_shown = warned = False
    while True:
        frame = reader.frame
        now = time.time()
        if frame is not last and frame.shape[:2] != PLACEHOLDER_HW:
            if last is None:
                print(f"video ok: {frame.shape[1]}x{frame.shape[0]}. Press q or Esc in the window to quit.")
            else:
                rate = 1.0 / max(now - t_new, 1e-3)
                fps = rate if fps == 0.0 else 0.9 * fps + 0.1 * rate
            last, t_new, stale_shown = frame, now, False
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # djitellopy gives RGB, YOLO and OpenCV want BGR
            t0 = time.perf_counter()
            dets = detector.detect(bgr)
            infer_ms = (time.perf_counter() - t0) * 1e3
            draw(bgr, dets)
            best = f", best {dets[0].conf:.2f}" if dets else ""
            put_text(bgr, f"{name}: {len(dets)} found{best}", 30, (0, 255, 0) if dets else (255, 255, 255))
            put_text(bgr, f"infer {infer_ms:.0f} ms   {fps:.1f} fps   battery {battery(tello)}", 60)
            shown = bgr
            cv2.imshow(WINDOW, shown)
        elif shown is not None and now - t_new > STALE_S and not stale_shown:
            stale = shown.copy()
            put_text(stale, "NO NEW VIDEO FRAMES", 100, (0, 0, 255))
            cv2.imshow(WINDOW, stale)
            stale_shown = True
            print("no new video frames for over 1 s")
        elif shown is None:
            if now - t_start > 15 and not warned:
                print("still no video frame after 15 s: is the Tello app open, or another program on UDP 11111?")
                warned = True
            time.sleep(0.005)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detect", required=True, choices=sorted(DETECTORS), help="what YOLO looks for")
    args = p.parse_args()

    print(f"loading the {args.detect} detector ...")
    detector = DETECTORS[args.detect]()  # before touching the drone: a missing model fails here
    print(f"detector ready on {detector.device}")
    try:
        tello = Tello()
    except OSError as e:
        raise SystemExit(f"Cannot open the Tello ports ({e}). Is another drone program still running?")
    try:
        run(tello, detector, args.detect)
    except TelloException as e:
        print(f"Tello error: {e}\nIs this laptop on the TELLO-xxxxxx Wi-Fi, and is the drone on?")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        tello.retry_count = 1  # if the drone is gone, give up on streamoff after one 7 s timeout, not three
        tello.end()  # stops the stream and the frame reader (and would land if it were flying)


if __name__ == "__main__":
    main()
