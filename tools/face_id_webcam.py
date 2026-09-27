"""Webcam smoke test for face identification: the go/no-go check before flying.

  python tools/face_id_webcam.py                     # insightface
  python tools/face_id_webcam.py --backend opencv

Runs the person detector + FaceIdentifier on the laptop webcam and overlays
name/similarity on every person. Walk back to 3-4 m to see where identification
stops working. q quits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["insightface", "opencv"], default="insightface")
    p.add_argument("--people", default=str(ROOT / "people"))
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    import cv2

    from reachglass.config import load_config
    from reachglass.detect import DETECTORS
    from reachglass.person.identify import FACE_IDENTIFIERS
    from reachglass.track import SimpleTracker

    cfg = load_config()
    person_det = DETECTORS.build(cfg.perception.person_detector)
    ident = FACE_IDENTIFIERS.build(args.backend, people_dir=args.people)
    if not ident.names:
        print(f"nobody enrolled: put photos in {args.people}/ and run tools/enroll_faces.py")
        return 1
    print(f"can identify: {', '.join(ident.names)}")
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("no webcam")
        return 1
    t_fps, n_fps, fps = time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        persons = [d for d in person_det.detect(frame) if d.cls == "person"]
        dets = tracker.update(persons, t, (frame.shape[1], frame.shape[0]))
        matches = ident.identify(frame, dets, t)
        for det, m in zip(dets, matches):
            x1, y1, x2, y2 = map(int, det.bbox)
            color = (0, 200, 0) if m and not m.provisional else (0, 180, 255) if m else (160, 160, 160)
            label = f"{m.name} {m.sim:.2f}{'?' if m.provisional else ''}" if m else "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        n_fps += 1
        if t - t_fps >= 1.0:
            fps, n_fps, t_fps = n_fps / (t - t_fps), 0, t
        cv2.putText(frame, f"{fps:.1f} fps", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("face id", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
