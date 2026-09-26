"""Download every YOLO model into integration/models, and test-run each once.

Copied from tools/download_models.py (the YOLO entries; the depth model stays there). From integration/:

    python -m yolo.download            # everything
    python -m yolo.download --only yolo11n-pose.pt yolo-world

  yolo11n.pt / yolo11n-pose.pt     default context detector / person + keypoints detector
  yolo11s.pt / yolo11s-pose.pt     more accurate, ~2.5x slower
  yolov8s-worldv2.pt (+ CLIP)      open-vocabulary detector: any text class ("keys", "door", "red mug")
"""

from __future__ import annotations

import argparse
import sys
import time

from .detector import MODELS_DIR as MODELS


def test_image():
    import cv2
    from ultralytics.utils import ASSETS

    return cv2.imread(str(ASSETS / "bus.jpg"))


def yolo(name: str):
    def run():
        from ultralytics import YOLO

        m = YOLO(str(MODELS / name))
        r = m.predict(test_image(), verbose=False)[0]
        extra = f", keypoints {tuple(r.keypoints.data.shape)}" if r.keypoints is not None else ""
        return f"{len(r.boxes)} boxes on bus.jpg{extra}"

    return run


def yolo_world():
    from ultralytics import YOLO

    m = YOLO(str(MODELS / "yolov8s-worldv2.pt"))
    m.set_classes(["person", "bus", "water bottle"])  # downloads/installs the CLIP text encoder on first use
    r = m.predict(test_image(), verbose=False)[0]
    return f"{len(r.boxes)} boxes for prompts {list(m.names.values())}"


ITEMS = {
    "yolo11n.pt": yolo("yolo11n.pt"),
    "yolo11n-pose.pt": yolo("yolo11n-pose.pt"),
    "yolo11s.pt": yolo("yolo11s.pt"),
    "yolo11s-pose.pt": yolo("yolo11s-pose.pt"),
    "yolo-world": yolo_world,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(ITEMS))
    args = p.parse_args()
    MODELS.mkdir(exist_ok=True)
    failed = []
    for name in args.only or ITEMS:
        t0 = time.time()
        try:
            msg = ITEMS[name]()
            print(f"OK    {name:<18} {time.time() - t0:5.1f}s  {msg}")
        except Exception as e:  # keep going: report every failure at the end
            failed.append(name)
            print(f"FAIL  {name:<18} {e.__class__.__name__}: {e}")
    print(f"\nmodels in {MODELS}: {sorted(p.name for p in MODELS.iterdir())}")
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
