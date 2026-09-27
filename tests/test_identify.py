"""Face identification: sticky-identity policy (fake embedder, no ONNX) and the
perception integration for name targets ("find arthur")."""

import math

import numpy as np
import pytest

from reachglass.config import load_config
from reachglass.detect import Detector
from reachglass.person.identify import FaceIdentifier, face_crop
from reachglass.perception import Perception
from reachglass.types import Detection, Frame


class FakeIdentifier(FaceIdentifier):
    """Enrollment bypassed; `self.next` is the embedding the 'camera' sees (None = face unreadable)."""

    def __init__(self, **kw):
        super().__init__(people_dir="__nonexistent__", **kw)
        self.names = ["arthur"]
        self._embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        self.next = None
        self.embed_calls = 0

    def embed_face(self, bgr):
        self.embed_calls += 1
        return None if self.next is None else np.asarray(self.next, dtype=np.float32)


ARTHUR = [1.0, 0.0]
STRANGER = [0.0, 1.0]  # cosine 0 vs arthur: below reject_threshold


def img():
    return np.zeros((720, 960, 3), np.uint8)


def person(tid, x=400, w=120, h=360):
    return Detection("person", 0.8, (x, 200, x + w, 200 + h), track_id=tid)


# ------------------------------------------------------------------ policy
def test_match_then_sticky_cache():
    f = FakeIdentifier()
    f.next = ARTHUR
    m = f.identify(img(), [person(1)], t=0.0)[0]
    assert m.name == "arthur" and m.sim > 0.9 and not m.provisional
    calls = f.embed_calls
    m = f.identify(img(), [person(1)], t=1.0)[0]  # within reverify_s: cached, no embed
    assert m.name == "arthur" and f.embed_calls == calls
    f.next = None  # reverify due but the face is unreadable: keep believing
    m = f.identify(img(), [person(1)], t=3.0)[0]
    assert m is not None and m.name == "arthur"


def test_two_strong_mismatches_drop_identity():
    f = FakeIdentifier()
    f.next = ARTHUR
    assert f.identify(img(), [person(1)], t=0.0)[0].name == "arthur"
    f.next = STRANGER
    assert f.identify(img(), [person(1)], t=3.0)[0].name == "arthur"  # first mismatch: benefit of the doubt
    assert f.identify(img(), [person(1)], t=6.0)[0] is None  # second: dropped


def test_provisional_inheritance_after_reset():
    f = FakeIdentifier()
    f.next = ARTHUR
    assert f.identify(img(), [person(1)], t=0.0)[0].name == "arthur"
    f.reset_tracks()  # the drone moved: track ids change
    f.next = None  # too far to read the face yet
    m = f.identify(img(), [person(7)], t=1.0)[0]  # ONE person in frame: inherits
    assert m is not None and m.name == "arthur" and m.provisional
    # ...and a later face match makes it firm again
    f.next = ARTHUR
    m = f.identify(img(), [person(7)], t=4.0)[0]
    assert m.name == "arthur" and not m.provisional


def test_no_inheritance_with_two_people_or_after_relock_window():
    f = FakeIdentifier()
    f.next = ARTHUR
    f.identify(img(), [person(1)], t=0.0)
    f.reset_tracks()
    f.next = None
    ms = f.identify(img(), [person(7), person(8, x=700)], t=1.0)  # two people: identity must be re-earned
    assert ms == [None, None]

    f2 = FakeIdentifier()
    f2.next = ARTHUR
    f2.identify(img(), [person(1)], t=0.0)
    f2.reset_tracks()
    f2.next = None
    f2.identify(img(), [], t=1.0)  # relock clock starts
    assert f2.identify(img(), [person(7)], t=1.0 + f2.relock_s + 1)[0] is None  # window expired


def test_face_crop_upscales_small_heads():
    d = person(1, w=40, h=120)  # small/far person: head crop < 140 px
    crop = face_crop(img(), d)
    assert crop is not None and max(crop.shape[:2]) >= 60  # upscaled x2


# ------------------------------------------------------------------ perception integration
class Scripted(Detector):
    def __init__(self, classes, fn):
        self._classes, self.fn, self.calls = classes, fn, 0

    @property
    def classes(self):
        return self._classes

    def detect(self, image):
        self.calls += 1
        return self.fn(self.calls)


def frame(i):
    return Frame(np.zeros((720, 960, 3), np.uint8), i * 0.1, i)


def make_perception(fake):
    cfg = load_config()
    person_det = Scripted(["person"], lambda i: [Detection("person", 0.8, (400, 150, 520, 560))])
    target_det = Scripted(["bottle"], lambda i: [])
    return Perception(cfg, person_det, target_det, None, identifier=fake), person_det, target_det


def test_name_target_flows_through_perception():
    fake = FakeIdentifier()
    per, person_det, target_det = make_perception(fake)
    assert "arthur" in per.vocabulary() and "person" not in per.vocabulary()
    assert per.owner_of("arthur") == "person"

    per.set_mode("search")
    per.set_target("arthur")
    assert per.name_target()
    fake.next = ARTHUR
    rs = [per.update(frame(i)) for i in range(9)]
    assert target_det.calls == 0  # the bottle detector has nothing to contribute
    # person detector runs at the target stride (3), and target_ran follows it
    assert person_det.calls == 3 and [r.target_ran for r in rs] == [i % 3 == 0 for i in range(9)]
    hits = [r for r in rs if r.target is not None]
    assert hits and all(r.target.cls == "arthur" and r.target.confirmed for r in hits)
    assert hits[0].persons[0].name == "arthur"
    assert hits[0].target.range_m is not None  # from the person estimator
    assert rs[-1].target_unseen_s < 1.0

    # the tracker det stays "person": the copy was renamed, not the original
    assert all(d.cls == "person" for d in rs[0].detections)


def test_bottle_target_unaffected_by_identifier():
    fake = FakeIdentifier()
    per, person_det, target_det = make_perception(fake)
    per.set_mode("search")
    per.set_target("bottle")
    assert not per.name_target()
    per.update(frame(0))
    assert target_det.calls == 1  # object path runs exactly as before


def test_dormant_without_identifier():
    cfg = load_config()
    per = Perception(cfg, Scripted(["person"], lambda i: []), Scripted(["bottle"], lambda i: []), None)
    assert "arthur" not in per.vocabulary()
    assert per.owner_of("arthur") is None and not per.name_target()


# ------------------------------------------------------------------ end-to-end sim: "find arthur"
@pytest.mark.slow
def test_sim_find_arthur_announces_without_approaching():
    """The sim person IS arthur: query -> explore identifies them -> the mission announces
    where they are (person phrasing) WITHOUT flying at them (no APPROACH, no fly-over)."""
    from reachglass.config import load_config
    from reachglass.mission import ScriptedInbox
    from reachglass.person.identify import SimIdentifier
    from reachglass.sim.runner import SimRunner

    said = []
    cfg = load_config()
    inbox = ScriptedInbox([(26.0, "find arthur")])
    r = SimRunner(cfg, inbox=inbox, seed=0, announce=said.append)
    r.perception.identifier = SimIdentifier()
    r.run(25.0)
    assert r.mission.state == "FOLLOW"
    r.run(120.0, until=lambda rr: rr.mission.state in ("ARRIVED", "REACQUIRE", "HOLD", "LANDED"))
    states_seen = [s for _, s, _ in r.mission.history]
    assert r.mission.state == "ARRIVED", (states_seen, r.ctx.notes[-12:])
    assert "APPROACH" not in states_seen, states_seen  # a person is announced, not approached
    assert any("Looking for Arthur." in s for s in said), said
    g = r.mission.guidance
    assert g is not None and g.person and g.target_cls == "arthur"
    assert any("Arthur" in s for s in said[-2:]), said  # the arrival announcement names him
    # and it kept a respectful distance while identifying (explore clamps hops to clearance)
    px, py, _ = r.truth_person()
    d = r.sim.drone.pos
    assert math.hypot(d[0] - px, d[1] - py) > 0.9
    assert not any("fly over" in n for n in r.ctx.notes), [n for n in r.ctx.notes if "fly over" in n]
