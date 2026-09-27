"""Put every model yolo/ needs into yolo/models/. Run once per laptop, with internet, before joining the Tello Wi-Fi.

    cd rewrite && python -m yolo.prepare           # add --force after changing BOTTLE_PROMPTS

1. yolo11n-pose.pt and yolov8s-worldv2.pt: downloaded from the Ultralytics releases if missing.
2. bottle-world.pt: yolov8s-worldv2 with BOTTLE_PROMPTS baked in, so detection never needs CLIP or internet.
   Baking loads the CLIP ViT-B/32 text encoder from yolo/models/clip/ (downloaded there, ~350 MB, if missing).
3. Builds both detectors once and prints their classes and device, to prove the setup works.
"""

from __future__ import annotations

import argparse

from .detector import MODELS_DIR
from .presets import BOTTLE, BOTTLE_PROMPTS, DETECTORS, PERSON

WORLD_BASE = "yolov8s-worldv2.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-bake bottle-world.pt even if it exists")
    args = p.parse_args()

    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    for name in (PERSON["weights"], WORLD_BASE):
        path = MODELS_DIR / name
        if not path.is_file():
            print(f"downloading {name}")
            YOLO(str(path))  # Ultralytics downloads its release assets to the path given
        print(f"ok  {path}")

    baked = MODELS_DIR / BOTTLE["weights"]
    if args.force or not baked.is_file():
        import ultralytics.nn.text_model as text_model

        text_model.WEIGHTS_DIR = MODELS_DIR  # Ultralytics loads CLIP from WEIGHTS_DIR/clip (downloads it there)
        world = YOLO(str(MODELS_DIR / WORLD_BASE))
        world.set_classes(list(BOTTLE_PROMPTS))
        world.model.clip_model = None  # don't save the 350 MB text encoder inside the file
        world.save(str(baked))
        print(f"baked {BOTTLE_PROMPTS} into {baked}")
    print(f"ok  {baked}")

    for name, make in DETECTORS.items():
        d = make()
        print(f"ok  {name} detector: classes {d.classes}, device {d.device}")


if __name__ == "__main__":
    main()
