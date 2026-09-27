"""Who is this person? Face identification on person-detection crops.

Enrollment is one photo per teammate in people/ (arthur.jpg -> "arthur"); there
is no training. At runtime the identifier only ever looks at crops around the
person detector's boxes/keypoints (a face at 2-5 m is 30-80 px -- running a face
detector on the full frame wastes time and finds nothing), and identity is
STICKY per track_id so the embedding model runs rarely.

Two facts of this codebase shape the policy here:
  * behaviors reset the tracker after every discrete move (base.py), so track
    ids do not survive a hop or a 45 deg scan turn. After a reset, if exactly
    one person is in frame shortly after, they provisionally inherit the last
    confirmed identity while re-verification keeps running.
  * detectors skip frames by stride, so time (reverify_s), not frame counts,
    drives re-verification.

Backends (FACE_IDENTIFIERS registry, picked by cfg.perception.face_identifier):
  insightface  SCRFD detect + ArcFace embed (best, needs `pip install insightface`)
  opencv       YuNet + SFace, ships with opencv-python + two ONNX files in models/
  null         disabled
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..registry import Registry
from ..types import Detection

FACE_IDENTIFIERS = Registry("face identifier")

# COCO-17 head keypoints (see person/estimator.py)
_HEAD_KP = (0, 1, 2, 3, 4)  # nose, eyes, ears


@dataclass
class FaceMatch:
    name: str
    sim: float
    provisional: bool = False  # inherited across a tracker reset, not yet re-verified


@dataclass
class _Track:
    name: str
    sim: float
    t_verified: float
    provisional: bool = False
    mismatches: int = 0


def face_crop(image: np.ndarray, det: Detection, upscale_below_px: int = 140,
              kp_conf: float = 0.3) -> np.ndarray | None:
    """Square crop around the head. Keypoints (nose/eyes/ears) when confident,
    else the upper third of the box. Upscaled x2 when small so the face
    detector sees a >=~48 px face."""
    h_img, w_img = image.shape[:2]
    pts = None
    if det.keypoints is not None and len(det.keypoints) >= 5:
        head = det.keypoints[list(_HEAD_KP)]
        head = head[head[:, 2] >= kp_conf][:, :2]
        if len(head) >= 2:
            pts = head
    if pts is not None:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        side = max(span * 3.0, det.w * 0.6, 32.0)
    else:
        cx, cy = det.cx, det.bbox[1] + det.h / 6.0
        side = max(det.w * 0.8, 32.0)
    half = side / 2.0
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w_img, cx + half)), int(min(h_img, cy + half))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = image[y1:y2, x1:x2]
    if max(crop.shape[:2]) < upscale_below_px:
        import cv2

        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return crop


class FaceIdentifier:
    """Shared enrollment + sticky-identity policy; backends implement embed_face."""

    def __init__(self, people_dir: str = "people", match_threshold: float = 0.40,
                 reject_threshold: float = 0.20, min_face_px: int = 24,
                 upscale_below_px: int = 140, max_crops: int = 2,
                 reverify_s: float = 2.0, relock_s: float = 5.0):
        self.people_dir = Path(people_dir)
        self.match_threshold = match_threshold
        self.reject_threshold = reject_threshold
        self.min_face_px = min_face_px
        self.upscale_below_px = upscale_below_px
        self.max_crops = max_crops
        self.reverify_s = reverify_s
        self.relock_s = relock_s
        self.names: list[str] = []
        self._embeddings: np.ndarray | None = None  # (N, D) L2-normalized
        self._cache: dict[int, _Track] = {}
        self._legacy: tuple[str, float, float] | None = None  # (name, sim, t) at reset
        self._load_people()

    # ------------------------------------------------------------------ backend API
    def embed_face(self, bgr: np.ndarray) -> np.ndarray | None:
        """Largest face in the image -> L2-normalized embedding, or None."""
        raise NotImplementedError

    # ------------------------------------------------------------------ enrollment
    def _photos(self) -> list[Path]:
        if not self.people_dir.is_dir():
            return []
        return sorted(p for p in self.people_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def _load_people(self) -> None:
        photos = self._photos()
        if not photos:
            return  # feature dormant: names stays []
        cache = self.people_dir / "faces.npz"
        stamp = [(p.name, p.stat().st_mtime) for p in photos]
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            if (str(z["backend"]) == self.backend_name
                    and list(map(tuple, z["stamp"])) == [(n, str(m)) for n, m in stamp]):
                self.names = [str(n) for n in z["names"]]
                self._embeddings = z["embeddings"].astype(np.float32)
                return
        self.enroll(photos, cache, stamp)

    def enroll(self, photos: list[Path], cache: Path, stamp) -> None:
        import cv2

        names, embs = [], []
        for p in photos:
            img = cv2.imread(str(p))
            emb = None if img is None else self.embed_face(img)
            if emb is None:
                print(f"[face] WARNING: no face found in {p.name}, skipping")
                continue
            names.append(p.stem.lower())
            embs.append(emb)
        self.names = names
        if embs:
            self._embeddings = np.stack(embs).astype(np.float32)
            np.savez(cache, names=np.array(names), embeddings=self._embeddings,
                     backend=self.backend_name,
                     stamp=np.array([(n, str(m)) for n, m in stamp]))
        print(f"[face] enrolled: {', '.join(names) if names else 'nobody'}")

    @property
    def backend_name(self) -> str:
        return type(self).__name__

    # ------------------------------------------------------------------ runtime
    def reset_tracks(self) -> None:
        """Tracker ids are about to change (drone moved). Remember the freshest
        confirmed identity so the (single) person reappearing can inherit it."""
        best = None
        for tr in self._cache.values():
            if not tr.provisional and (best is None or tr.t_verified > best.t_verified):
                best = tr
        if best is not None:
            self._legacy = (best.name, best.sim, None)  # stamped with frame time at the next identify()
        self._cache.clear()

    def identify(self, image: np.ndarray, person_dets: list[Detection],
                 t: float) -> list[FaceMatch | None]:
        """One FaceMatch (or None) per input detection. Runs the embedding model
        only on uncached / reverify-due tracks, largest max_crops people only."""
        out: list[FaceMatch | None] = [None] * len(person_dets)
        order = sorted(range(len(person_dets)), key=lambda i: -person_dets[i].area)
        for rank, i in enumerate(order):
            det = person_dets[i]
            tid = det.track_id
            tr = self._cache.get(tid) if tid is not None else None
            if tr is not None and t - tr.t_verified < self.reverify_s:
                out[i] = FaceMatch(tr.name, tr.sim, tr.provisional)
                continue
            if rank >= self.max_crops:
                if tr is not None:
                    out[i] = FaceMatch(tr.name, tr.sim, tr.provisional)
                continue
            if det.w < self.min_face_px * 2:  # face would be far below min_face_px
                if tr is not None:
                    out[i] = FaceMatch(tr.name, tr.sim, tr.provisional)
                continue
            out[i] = self._verify(image, det, tr, tid, t)
        self._inherit(person_dets, out, t)
        return out

    def _verify(self, image, det, tr, tid, t) -> FaceMatch | None:
        crop = face_crop(image, det, self.upscale_below_px)
        emb = None if crop is None else self.embed_face(crop)
        if emb is None or self._embeddings is None or not self.names:
            # face not readable this frame: keep whatever we believed
            return FaceMatch(tr.name, tr.sim, tr.provisional) if tr is not None else None
        sims = self._embeddings @ emb
        j = int(np.argmax(sims))
        sim = float(sims[j])
        if sim >= self.match_threshold:
            if tid is not None:
                self._cache[tid] = _Track(self.names[j], sim, t)
            return FaceMatch(self.names[j], sim)
        if tr is not None:
            if sim < self.reject_threshold:  # a good look at a clearly different face
                tr.mismatches += 1
                if tr.mismatches >= 2:
                    if tid is not None:
                        self._cache.pop(tid, None)
                    return None
            return FaceMatch(tr.name, tr.sim, tr.provisional)
        return None

    def _inherit(self, person_dets, out, t) -> None:
        """Exactly one person in frame right after a tracker reset: they keep the
        last confirmed identity, provisionally, until a face match says otherwise."""
        if self._legacy is None:
            return
        name, sim, t_reset = self._legacy
        if t_reset is None:  # first frame after the reset: start the relock clock (frame time, sim-safe)
            t_reset = t
            self._legacy = (name, sim, t)
        if t - t_reset > self.relock_s:
            self._legacy = None
            return
        if len(person_dets) != 1 or out[0] is not None:
            return
        tid = person_dets[0].track_id
        if tid is not None:
            # t_verified in the past so re-verification runs at every opportunity
            self._cache[tid] = _Track(name, sim, t - self.reverify_s, provisional=True)
        out[0] = FaceMatch(name, sim, provisional=True)


@FACE_IDENTIFIERS.register("null")
class NullIdentifier(FaceIdentifier):
    def __init__(self, **_):
        super().__init__(people_dir="__none__")

    def embed_face(self, bgr):
        return None


@FACE_IDENTIFIERS.register("sim")
class SimIdentifier(FaceIdentifier):
    """Simulator oracle: 'recognizes' the enrolled name when the person's box is big
    enough in frame (a stand-in for 'close enough to read the face', ~2.5-4 m)."""

    def __init__(self, name: str = "arthur", min_frac: float = 0.4, **kw):
        super().__init__(people_dir="__nonexistent__", **kw)
        self.names = [name]
        self.min_frac = min_frac

    def identify(self, image, person_dets, t):
        h_img = image.shape[0]
        return [FaceMatch(self.names[0], 0.95) if d.h >= self.min_frac * h_img else None
                for d in person_dets]

    def embed_face(self, bgr):
        return None


@FACE_IDENTIFIERS.register("insightface")
class InsightFaceIdentifier(FaceIdentifier):
    def __init__(self, model: str = "buffalo_l", **kw):
        from insightface.app import FaceAnalysis  # heavy import only when used

        self.app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"],
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        super().__init__(**kw)

    def embed_face(self, bgr):
        faces = self.app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = f.normed_embedding.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


@FACE_IDENTIFIERS.register("opencv")
class OpenCVIdentifier(FaceIdentifier):
    """Zero-install fallback: YuNet + SFace from models/ (see tools/enroll_faces.py
    for the download). SFace cosine threshold is lower than ArcFace's: use
    match_threshold ~0.363 (OpenCV's published value)."""

    DET = "face_detection_yunet_2023mar.onnx"
    REC = "face_recognition_sface_2021dec.onnx"

    def __init__(self, models_dir: str = "models", **kw):
        import cv2

        d = Path(models_dir)
        det_p, rec_p = d / self.DET, d / self.REC
        if not det_p.is_file() or not rec_p.is_file():
            raise FileNotFoundError(
                f"missing {det_p} / {rec_p}: run python tools/enroll_faces.py --backend opencv")
        self.det = cv2.FaceDetectorYN_create(str(det_p), "", (320, 320), 0.6, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF_create(str(rec_p), "")
        kw.setdefault("match_threshold", 0.363)
        kw.setdefault("reject_threshold", 0.15)
        super().__init__(**kw)

    def embed_face(self, bgr):
        import cv2

        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.rec.alignCrop(bgr, f)
        emb = self.rec.feature(aligned).reshape(-1).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)
