#!/usr/bin/env python3
"""Download every model the ReachGlass stack can use into ./models, and test-run each once.

    python tools/download_models.py            # everything
    python tools/download_models.py --only yolo11n-pose.pt depth

Run it once per machine (the Mac and the Intel demo laptop). After it passes, the stack runs offline.

  yolo11n.pt / yolo11n-pose.pt     default context detector / person + keypoints detector
  yolo11s.pt / yolo11s-pose.pt     more accurate, ~2.5x slower
  yolov8s-worldv2.pt (+ CLIP)      open-vocabulary detector: any text class ("keys", "door", "red mug")
  depth                            Depth-Anything-V2-Small (Hugging Face) for monocular free space
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HF_DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"


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


def depth():
    import numpy as np
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=HF_DEPTH, model_kwargs={"cache_dir": str(MODELS / "hf")})
    out = pipe(Image.fromarray(test_image()[..., ::-1]))
    d = np.asarray(out["predicted_depth"]).squeeze()
    return f"depth map {d.shape}, range {float(d.min()):.2f}..{float(d.max()):.2f}"


ITEMS = {
    "yolo11n.pt": yolo("yolo11n.pt"),
    "yolo11n-pose.pt": yolo("yolo11n-pose.pt"),
    "yolo11s.pt": yolo("yolo11s.pt"),
    "yolo11s-pose.pt": yolo("yolo11s-pose.pt"),
    "yolo-world": yolo_world,
    "depth": depth,
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
