"""See what the bottle detector sees, live: boxes, confidence, estimated distance and the time per frame.

    python -m reachglass.tools.target_view tello --config site.yaml               # YOLO-World (default)
    python -m reachglass.tools.target_view tello --config site.yaml --target color
    python -m reachglass.tools.target_view photo.jpg

Walk the bottle away from the drone (or hold the drone) to see how far it is still found, and check that nothing
else lights up. Green box = passes the confidence to confirm on a single sighting; yellow = a weaker candidate.
q = quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from ..config import load_config
from ..detect import DETECTORS
from ..geometry import camera_from_config
from .common import latest, open_source


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default="tello")
    p.add_argument("--config", help="YAML overrides, e.g. site.yaml")
    p.add_argument("--target", choices=["yolo-world", "color"], default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, target=args.target)
    det = DETECTORS.build(cfg.perception.target_detector)
    cam0 = camera_from_config(cfg.camera)
    confirm = cfg.tracking.confirm_conf
    opened = open_source(args.source)
    win = "target view (q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ms = []
    try:
        while True:
            img = latest(opened.source, 10.0)
            if img is None:
                print("no image")
                return 1
            t0 = time.perf_counter()
            found = det.detect(img)
            ms = (ms + [1000 * (time.perf_counter() - t0)])[-20:]
            cam = cam0.for_frame(img.shape[1], img.shape[0])
            view = img.copy()
            for d in found:
                h_m = cfg.perception.object_heights_m.get(d.cls)
                r = cam.range_from_height(d.h, h_m, d.cx) if h_m else None
                col = (0, 220, 0) if d.conf >= confirm else (0, 220, 220)
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
                label = f"{d.cls} {d.conf:.2f}" + (f" {r:.1f} m" if r else "")
                cv2.putText(view, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(view, f"{cfg.perception.target_detector.kind}: {sum(ms) / len(ms):.0f} ms/frame, "
                        f"{len(found)} found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        opened.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
