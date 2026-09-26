"""Step 4: tracker + target lock."""

import pytest

from reachglass.track import SimpleTracker, TargetLock, largest
from reachglass.types import Detection

SIZE = (960, 720)


def box(cx, cy, w=40, h=80, cls="bottle", conf=0.8):
    return Detection(cls, conf, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def run(tracker, frames, dt=0.1):
    out = []
    for i, dets in enumerate(frames):
        out.append(tracker.update(dets, i * dt, SIZE))
    return out


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_id_through_motion_and_new_objects_get_new_ids():
    tr = SimpleTracker()
    frames = [[box(100 + 15 * i, 300)] for i in range(10)]
    frames[5] = frames[5] + [box(700, 300)]
    ids = run(tr, frames)
    assert len({f[0].track_id for f in ids}) == 1
    assert ids[5][1].track_id != ids[5][0].track_id


def test_tracker_fast_shift_matched_by_centre_distance():
    tr = SimpleTracker()
    a = tr.update([box(300, 300)], 0.0, SIZE)[0]
    b = tr.update([box(360, 300)], 0.1, SIZE)[0]  # moved 1.5 box widths: IoU 0, centre distance 5 % of diagonal
    assert a.track_id == b.track_id


def test_tracker_does_not_match_across_classes_or_very_different_sizes():
    tr = SimpleTracker()
    a = tr.update([box(300, 300, cls="bottle")], 0.0, SIZE)[0]
    b = tr.update([box(300, 300, cls="cup")], 0.1, SIZE)[0]
    assert a.track_id != b.track_id
    tr = SimpleTracker()
    a = tr.update([box(300, 300, w=200, h=300)], 0.0, SIZE)[0]
    b = tr.update([box(420, 300, w=20, h=30)], 0.1, SIZE)[0]  # tiny box nearby: not the same object
    assert a.track_id != b.track_id


def test_tracker_two_objects_keep_their_ids():
    tr = SimpleTracker()
    frames = [[box(200 + 10 * i, 300), box(600 - 10 * i, 320)] for i in range(12)]
    out = run(tr, frames)
    left_ids = {f[0].track_id for f in out}
    right_ids = {f[1].track_id for f in out}
    assert len(left_ids) == 1 and len(right_ids) == 1 and left_ids != right_ids


def test_tracker_expires_old_tracks():
    tr = SimpleTracker(max_age_s=0.5)
    a = tr.update([box(300, 300)], 0.0, SIZE)[0]
    tr.update([], 0.3, SIZE)
    assert tr.tracks
    tr.update([], 0.9, SIZE)
    assert not tr.tracks
    b = tr.update([box(300, 300)], 1.0, SIZE)[0]
    assert b.track_id != a.track_id


# ------------------------------------------------------------------ lock
def make_lock(**kw):
    tr = SimpleTracker()
    lk = TargetLock(tr, **kw)
    lk.set_class("bottle")
    return tr, lk


def step(tr, lk, dets, t):
    return lk.update(tr.update(dets, t, SIZE), t, SIZE)


def test_lock_confirms_after_n_hits_not_before():
    tr, lk = make_lock(confirm_hits=3, confirm_window=5, confirm_conf=0.99)
    states = [step(tr, lk, [box(300, 300, conf=0.6)], i * 0.1) for i in range(4)]
    assert [s.confirmed for s in states] == [False, False, True, True]
    assert all(s.visible and not s.lost for s in states)


def test_lock_sporadic_detections_do_not_confirm():
    tr, lk = make_lock(confirm_hits=3, confirm_window=5, confirm_conf=0.99, lost_after_s=5)
    seq = [[box(300, 300, conf=0.5)] if i % 3 == 0 else [] for i in range(12)]  # 1-2 hits per 5 frames
    states = [step(tr, lk, d, i * 0.1) for i, d in enumerate(seq)]
    assert not any(s.confirmed for s in states)


def test_lock_single_confident_plausible_detection_confirms():
    tr, lk = make_lock(confirm_conf=0.7, plausible=lambda d: d.h > 50)
    assert step(tr, lk, [box(300, 300, conf=0.9, h=80)], 0).confirmed
    tr, lk = make_lock(confirm_conf=0.7, plausible=lambda d: d.h > 50)
    assert not step(tr, lk, [box(300, 300, conf=0.9, h=30)], 0).confirmed  # implausible size


def test_lock_does_not_jump_to_a_more_confident_twin():
    tr, lk = make_lock()
    s = step(tr, lk, [box(200, 300, conf=0.6)], 0.0)
    locked = s.track_id
    for i in range(1, 10):
        # a second, MORE confident bottle appears; the lock must stay on the first
        s = step(tr, lk, [box(200 + 5 * i, 300, conf=0.6), box(700, 300, conf=0.95)], i * 0.1)
        assert s.track_id == locked and s.det.cx < 400


def test_lock_lost_after_timeout_then_relocks_nearby():
    tr, lk = make_lock(lost_after_s=0.5, confirm_hits=2)
    for i in range(3):
        s = step(tr, lk, [box(300, 300)], i * 0.1)
    assert s.confirmed
    s = step(tr, lk, [], 0.4)
    assert not s.lost and not s.visible and s.confirmed  # short dropout: still locked
    s = step(tr, lk, [], 0.9)
    assert s.lost and not s.confirmed
    # object reappears near where it was lost, under a new track id (tracker expired it)
    tr.tracks.clear()
    s = step(tr, lk, [box(310, 305), box(800, 300, conf=0.99)], 1.0)
    assert s.visible and s.det.cx == pytest.approx(310) and not s.lost


def test_lock_released_after_long_loss_picks_any_candidate():
    tr, lk = make_lock(lost_after_s=0.5)
    step(tr, lk, [box(100, 300)], 0.0)
    s = step(tr, lk, [box(800, 300, conf=0.9)], 2.0)  # 2 s later, far away: old lock released, new one taken
    assert s.visible and s.det.cx == pytest.approx(800)


def test_lock_class_switch_resets_and_filters():
    tr, lk = make_lock()
    step(tr, lk, [box(300, 300)], 0)
    lk.set_class("cup")
    s = lk.update(tr.update([box(300, 300), box(500, 300, cls="cup")], 0.1, SIZE), 0.1, SIZE)
    assert s.det.cls == "cup" and s.cls == "cup"
    lk.set_class(None)
    assert lk.update(tr.update([box(300, 300)], 0.2, SIZE), 0.2, SIZE).det is None


def test_lock_select_policy_largest_for_people():
    tr = SimpleTracker()
    lk = TargetLock(tr, select=largest)
    lk.set_class("person")
    s = lk.update(tr.update([box(200, 300, 60, 150, "person", 0.9), box(600, 300, 150, 400, "person", 0.5)], 0, SIZE), 0, SIZE)
    assert s.det.cx == pytest.approx(600)


def test_lock_never_seen_is_lost():
    tr, lk = make_lock()
    s = step(tr, lk, [], 0.0)
    assert s.lost and not s.confirmed and s.det is None
