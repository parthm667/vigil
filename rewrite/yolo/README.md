# yolo/ — YOLO detection (people, the team's blue bottle)

A self-contained package. It imports nothing from the old jerkgt13 code and nothing drone-related.
Import it with `rewrite/` on `sys.path`.

```python
from yolo import person_detector, bottle_detector, DETECTORS, draw

people = person_detector()               # loads the model and warms it up (a few seconds)
dets = people.detect(bgr)                # BGR uint8 HxWx3 -> list[Detection], most confident first
draw(bgr, dets)                          # boxes, "cls conf" labels, keypoints; draws in place
d = dets[0]; d.cls, d.conf, d.bbox, d.keypoints, d.cx, d.cy, d.w, d.h, d.area
```

**Frames must be BGR.** djitellopy's `frame_read.frame` is **RGB**, so convert it first with
`cv2.cvtColor(f, cv2.COLOR_RGB2BGR)`. Passing RGB silently loses the bottle, because the blue check sees red.

## Files

| File | Contents |
|---|---|
| `detector.py` | `Detector` (Ultralytics wrapper: class filter, rename, agnostic NMS, HSV colour check), `Detection`, `auto_device`, `MODELS_DIR` |
| `presets.py` | `PERSON` and `BOTTLE` settings, `BOTTLE_PROMPTS`, `person_detector()`, `bottle_detector()`, `DETECTORS` (name to factory), bottle size 0.24 x 0.09 m |
| `draw.py` | `draw(image, detections)` |
| `prepare.py` | `python -m yolo.prepare`: puts every model in `models/` |
| `models/` | Weights. Gitignored by the root `.gitignore` (`models/`, `*.pt`) |

## Presets

Both come from jerkgt13's tuned perception config.

- **Person:** `yolo11n-pose.pt`, class person, conf 0.4, imgsz 640. `Detection.keypoints` is (17, 3) x, y, conf in COCO order.
- **Bottle:** `bottle-world.pt`, which is `yolov8s-worldv2` with the prompts "blue water bottle" and "hydro flask water bottle" baked in.
  - Both prompts are renamed to `bottle`; conf 0.2, imgsz 960, `agnostic_nms`.
  - Blue check: HSV `[95,60,40,130,255,255]` must cover at least 30 % of the box's middle (central 60 % of its width). The team's bottle scores 0.86 to 0.93 on the test photos.
  - Measured on this Mac (MPS): about 15 ms per 960x720 frame. Expect about 100 ms on a CPU.

`Detector(weights=..., classes=..., rename=..., conf=..., imgsz=..., agnostic_nms=..., require_color=...)`
builds anything else. Presets take keyword overrides, e.g. `bottle_detector(conf=0.3)`.

## Models and setup

- **Never downloads at runtime.** In flight the laptop is on the Tello Wi-Fi with no internet, so a missing file raises `FileNotFoundError` naming the prepare command.
- **`python -m yolo.prepare`** (run from `rewrite/`, once per laptop, with internet):
  - downloads `yolo11n-pose.pt` and `yolov8s-worldv2.pt`
  - bakes `bottle-world.pt`, then builds both detectors as a check
  - `--force` re-bakes after `BOTTLE_PROMPTS` change
- **Baking needs CLIP ViT-B/32**, loaded from `models/clip/` (~350 MB, downloaded there if missing). `prepare.py` points Ultralytics there by setting `ultralytics.nn.text_model.WEIGHTS_DIR = MODELS_DIR`. That's needed because Ultralytics' own `weights_dir` setting is the relative path `weights`, which depends on the current directory.
  - `clip_model` is set to None before saving, so the baked file stays at 26 MB.
  - Baked output was verified to match live prompts exactly: same boxes and scores on the 3 team photos.
- **Relative weight paths resolve inside `models/`**, never against the current directory. Absolute paths are used as given.
- **On this Mac, `models/` holds APFS clones** (`cp -c`) of `jerkgt13/models/*.pt` and `jerkgt13/weights/clip/ViT-B-32.pt`, so they take no extra disk space.

## Notes

- `device="auto"` picks `cuda:0`, then `mps`, then `cpu`. The Intel demo laptop will use the CPU unless OpenVINO is added later.
- Warmup runs one 960x720 inference at construction. A different image size makes the first call slower once (the first 1080x810 call took about 230 ms).
- `detect()` is not thread-safe. Call it from one thread.
- Drone test: `rewrite/tests/yolo_live.py --detect person|bottle` (djitellopy video, ground only).
- The objc warning "AVFFrameReceiver is implemented in both cv2 ... and av ..." appears because opencv-python and PyAV each bundle libavdevice. It concerns AVFoundation capture devices, which we don't use.
