"""Tune the dummy target's colour on site.

    python -m reachglass.tools.hsv_picker tello        # live Tello video (join its Wi-Fi first)
    python -m reachglass.tools.hsv_picker 0            # webcam
    python -m reachglass.tools.hsv_picker photo.jpg

Click the object: a range is set around its colour. Every click ADDS to the range, so click it in each
lighting you will fly in (daylight by the window, room lights on, in shadow) and the range covers all of them.
c = forget the clicks, p = print the config lines, q = quit. Adjust with the sliders.
Left: detections as the stack will see them (label, confidence, box). Right: the colour mask.
Check that NOTHING else in the room lights up in the mask (skin, wood, clothes, logos), in every lighting.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from ..detect import ColorBlobDetector
from .common import latest, open_source

WIN = "HSV picker (click the object; c = clear clicks; p = print config; q = quit)"
BARS = [("H lo", 100, 180), ("H hi", 122, 180), ("S lo", 100, 255), ("S hi", 255, 255), ("V lo", 70, 255), ("V hi", 255, 255)]


def ranges_from_bars() -> list[list[int]]:
    h0, h1, s0, s1, v0, v1 = (cv2.getTrackbarPos(n, WIN) for n, _, _ in BARS)
    s0, s1 = min(s0, s1), max(s0, s1)
    v0, v1 = min(v0, v1), max(v0, v1)
    return [[h0, s0, v0, h1, s1, v1]]  # h0 > h1 means wrap-around (red): the detector splits it


def range_from_samples(samples: list[tuple[int, int, int]], h_margin: int = 8, s_margin: int = 70,
                       v_margin: int = 80) -> list[int]:
    """One [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi] range covering every clicked (h, s, v) plus margins. Hue is
    circular (red sits on 0/180): the covering arc is the circle minus the largest gap between samples."""
    hs = sorted({h % 180 for h, _, _ in samples})
    gaps = [((hs[(i + 1) % len(hs)] - hs[i]) % 180 or 180, i) for i in range(len(hs))]
    _, i = max(gaps)
    lo, hi = hs[(i + 1) % len(hs)], hs[i]  # the arc starts after the largest gap and ends before it
    span = (hi - lo) % 180 + 2 * h_margin
    if span >= 180:
        h0, h1 = 0, 180
    else:
        h0, h1 = (lo - h_margin) % 180, (hi + h_margin) % 180
    s0 = max(60, min(s for _, s, _ in samples) - s_margin)
    v0 = max(60, min(v for _, _, v in samples) - v_margin)
    return [h0, s0, v0, h1, 255, 255]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec = argv[0] if argv else "tello"
    opened = open_source(spec)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, val, mx in BARS:
        cv2.createTrackbar(name, WIN, val, mx, lambda v: None)
    state = {"img": None, "clicks": []}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["img"] is None:
            return
        img = state["img"]
        x = int(x * img.shape[1] / 640)
        y = int(y * img.shape[0] / 480)
        patch = cv2.cvtColor(img[max(0, y - 3):y + 4, max(0, x - 3):x + 4], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = (int(np.median(patch[:, k])) for k in range(3))
        state["clicks"].append((h, s, v))
        print(f"clicked HSV = ({h}, {s}, {v})  [{len(state['clicks'])} click(s) in the range]")
        for (name, _, _), val in zip(BARS, range_from_samples(state["clicks"])):
            cv2.setTrackbarPos(name, WIN, int(val))

    cv2.setMouseCallback(WIN, on_click)
    try:
        while True:
            img = latest(opened.source, 10.0)
            if img is None:
                print("no image")
                return 1
            state["img"] = img
            try:
                det = ColorBlobDetector(hsv_ranges=ranges_from_bars())
            except ValueError as e:
                print(e)
                continue
            view = cv2.resize(img, (640, 480))
            sx, sy = 640 / img.shape[1], 480 / img.shape[0]
            for d in det.detect(img):
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 2)
                cv2.putText(view, f"{d.conf:.2f} {int(d.w)}x{int(d.h)}px", (int(x1 * sx), int(y1 * sy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            mask = cv2.cvtColor(cv2.resize(det.mask(img), (640, 480)), cv2.COLOR_GRAY2BGR)
            cv2.imshow(WIN, np.hstack([view, mask]))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                state["clicks"].clear()
                print("clicks cleared")
            if k == ord("p"):
                r = ranges_from_bars()[0]
                print("\n# paste into your config YAML:\nperception:\n  target_detector:\n    params:\n"
                      f"      hsv_ranges: [{r}]\n")
    finally:
        opened.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
