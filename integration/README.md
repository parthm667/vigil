# integration/yolo — the YOLO detector as a standalone module

A **copy** of all the YOLO code in the ReachGlass stack, pulled out into one package that imports nothing from
`reachglass/`. The originals are untouched and the stack still uses them. This copy does not sync with them:
if the detector changes in `reachglass/`, update it here too.

```python
from yolo import UltralyticsDetector, YOLO_WORLD_BOTTLE, PERSON_POSE, CONTEXT_OBJECTS

bottle = UltralyticsDetector(**YOLO_WORLD_BOTTLE)   # the team's blue bottle (the stack's default target)
people = UltralyticsDetector(**PERSON_POSE)         # people + 17 COCO keypoints
for d in bottle.detect(bgr_image):                  # most confident first
    print(d.cls, d.conf, d.bbox)                    # bbox = x1, y1, x2, y2 in pixels
```

Put `integration/` on the import path (run from inside it, or `sys.path.insert(0, ".../integration")`).

## Where each piece came from

| Here | Copied from | What it is |
|---|---|---|
| `yolo/detector.py` | `reachglass/detect/yolo.py` | `UltralyticsDetector` (detect / pose / YOLO-World prompts / blue check), `resolve_weights`, `auto_device` |
| `yolo/presets.py` | `reachglass/config.py` | `YOLO_WORLD_BOTTLE`, the person-pose and context-object detector params, the bottle size and lock threshold that go with YOLO-World |
| `yolo/download.py` | `tools/download_models.py` | Fetch and test-run the YOLO weights (the depth model stays in the original) |
| `yolo/types.py` | `reachglass/types.py`, `reachglass/detect/base.py` | `Detection`, `Detector` |
| `yolo/color.py` | `reachglass/detect/color_blob.py` | `parse_hsv_ranges`, `hsv_mask` (behind `require_color`) |
| `tests/test_yolo.py` | `tests/test_detectors.py` | The YOLO tests, plus checks that the module stands alone |
| `tests/assets/bottle/` | `tests/assets/bottle/` | The team's bottle photos for the YOLO-World test |

Differences from the originals:

- Bare weight names resolve into `integration/models/` rather than the repo's `models/`, so the folder is
  self-contained. Both are gitignored.
- `Detection` is its own class. It has the same fields as `reachglass.types.Detection`.
- Detectors are built directly (`UltralyticsDetector(**params)`) rather than via reachglass's
  `DETECTORS.build(ComponentSpec("yolo", params))`. The params are the same.

## Setup and tests

```bash
pip install -r integration/requirements.txt
cd integration
python -m yolo.download          # weights into integration/models + a test run of each
python -m pytest                 # all tests; -m "not yolo" skips the ones that need weights
```
