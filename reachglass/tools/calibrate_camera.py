"""Measure the Tello video's focal length (every distance estimate depends on it).

    python -m reachglass.tools.calibrate_camera tello --height 0.24 --distance 2.0            # drag a box around the whole bottle
    python -m reachglass.tools.calibrate_camera tello --height 0.19 --distance 2.0 --auto     # blue part of the dummy, no clicking

Put an object of known HEIGHT (m) at a known horizontal DISTANCE (m, tape measure) roughly in the middle of
the image, camera level with it. The tool measures its pixel height (--auto: the colour-blob detector finds
the blue dummy's blue body, 0.19 m; otherwise drag a box around it and press Enter) and prints f = pixel_height * distance / height.
Repeat at 2-3 distances; paste the median into the config (camera: fx / fy).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from ..detect import ColorBlobDetector
from .common import latest, open_source


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default="tello")
    p.add_argument("--height", type=float, required=True, help="object height in metres")
    p.add_argument("--distance", type=float, required=True, help="horizontal distance to the object in metres")
    p.add_argument("--auto", action="store_true", help="find the blue dummy automatically (20 frames, median)")
    args = p.parse_args(argv)
    opened = open_source(args.source)
    try:
        img = latest(opened.source, 10.0)
        if img is None:
            print("no image")
            return 1
        h_img, w_img = img.shape[:2]
        if args.auto:
            det = ColorBlobDetector()
            hs = []
            t_end = time.time() + 10
            while len(hs) < 20 and time.time() < t_end:
                d = det.detect(latest(opened.source))
                if d:
                    hs.append(d[0].h)
                time.sleep(0.05)
            if not hs:
                print("dummy not found: tune its colour with reachglass.tools.hsv_picker")
                return 2
            px = statistics.median(hs)
        else:
            x, y, w, h = cv2.selectROI("drag a box around the object, then Enter", img, showCrosshair=True)
            cv2.destroyAllWindows()
            if h <= 0:
                return 2
            px = float(h)
        f = px * args.distance / args.height
        f960 = f * 960.0 / w_img  # config values are for the 960x720 reference size
        print(f"\nobject {px:.1f} px tall in a {w_img}x{h_img} frame -> f = {f:.1f} px ({f960:.1f} px at 960x720)")
        print(f"\n# paste into your config YAML (median of a few measurements):\ncamera:\n  fx: {f960:.0f}\n  fy: {f960:.0f}\n")
    finally:
        opened.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
