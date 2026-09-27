"""Enroll teammates' faces: people/arthur.jpg -> "arthur" the drone can find.

  python tools/enroll_faces.py                      # insightface (downloads buffalo pack once)
  python tools/enroll_faces.py --backend opencv     # zero-install fallback (downloads 2 ONNX files)

One photo per person, filename stem = the spoken name. Re-run after changing
photos (the flight app also re-enrolls automatically if photos are newer than
people/faces.npz). Prints the pairwise similarity matrix: any pair above ~0.5
is confusable -- use clearer photos.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
OPENCV_MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download_opencv_models(models_dir: Path) -> None:
    models_dir.mkdir(exist_ok=True)
    for name, url in OPENCV_MODELS.items():
        dst = models_dir / name
        if dst.is_file():
            continue
        print(f"downloading {name}...")
        urllib.request.urlretrieve(url, dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--people", default=str(ROOT / "people"))
    p.add_argument("--backend", choices=["insightface", "opencv"], default="insightface")
    p.add_argument("--model", default="buffalo_l", help="insightface pack (buffalo_s if CPU-bound)")
    p.add_argument("--models-dir", default=str(ROOT / "models"), help="opencv: where the ONNX files go")
    args = p.parse_args()

    people = Path(args.people)
    people.mkdir(exist_ok=True)
    photos = sorted(p for p in people.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        print(f"no photos in {people}/ -- drop arthur.jpg etc. there and re-run")
        return 1

    if args.backend == "opencv":
        download_opencv_models(Path(args.models_dir))

    from reachglass.person.identify import FACE_IDENTIFIERS

    (people / "faces.npz").unlink(missing_ok=True)  # force a fresh enrollment
    kw = {"people_dir": str(people)}
    if args.backend == "insightface":
        kw["model"] = args.model
    else:
        kw["models_dir"] = args.models_dir
    ident = FACE_IDENTIFIERS.build(args.backend, **kw)

    if not ident.names:
        print("nothing enrolled (no face found in any photo?)")
        return 1
    embs = ident._embeddings
    print(f"\nenrolled {len(ident.names)}: {', '.join(ident.names)}")
    print("\npairwise similarity (above ~0.5 = confusable):")
    sims = embs @ embs.T
    width = max(len(n) for n in ident.names)
    print(" " * (width + 2) + "  ".join(f"{n[:6]:>6s}" for n in ident.names))
    confusable = []
    for i, n in enumerate(ident.names):
        print(f"{n:>{width}s}  " + "  ".join(f"{sims[i, j]:6.2f}" for j in range(len(ident.names))))
        for j in range(i + 1, len(ident.names)):
            if sims[i, j] > 0.5:
                confusable.append((n, ident.names[j], float(sims[i, j])))
    for a, b, s in confusable:
        print(f"WARNING: {a} vs {b} similarity {s:.2f} -- the drone may mix them up")
    print(f"\nsaved {people / 'faces.npz'} (backend {ident.backend_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
