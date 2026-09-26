"""Run log: one JSON line per loop (throttled) + events, for debugging a flight afterwards."""

from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, path: str | Path | None, every_s: float = 0.1):
        self.path = Path(path) if path else None
        self.every_s = every_s
        self._last = -1e9
        self._f = None
        self._state = None
        self._n_said = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("w")

    def _write(self, rec: dict) -> None:
        if self._f:
            self._f.write(json.dumps(rec, default=str) + "\n")

    def event(self, t: float, kind: str, **kw) -> None:
        self._write({"t": round(t, 3), "event": kind, **kw})
        if self._f:
            self._f.flush()

    def tick(self, ctx, mission, safety_events=None) -> None:
        t = ctx.now
        if mission is not None and mission.state != self._state:
            self._state = mission.state
            self.event(t, "state", state=mission.state, why=mission.history[-1][2] if mission.history else "")
        if mission is not None and len(mission.said) > self._n_said:
            for ts, s in mission.said[self._n_said:]:
                self.event(ts, "say", text=s)
            self._n_said = len(mission.said)
        if t - self._last < self.every_s:
            return
        self._last = t
        rec = {"t": round(t, 3), "wall": round(time.time(), 3)}
        if mission is not None:
            rec["state"], rec["status"] = mission.state, mission.status
        p = ctx.odom.pose
        rec["pose"] = [round(p.x, 3), round(p.y, 3), round(p.heading_deg, 1)]
        tel = ctx.tel
        if tel is not None:
            rec["tel"] = {"alt": tel.floor_altitude_m(), "tof": tel.tof_m, "yaw": tel.yaw_deg, "pitch": tel.pitch_deg,
                          "bat": tel.battery_pct}
        r = ctx.res
        if r is not None:
            rec["seq"], rec["latency_ms"] = r.seq, round(r.latency_ms, 1)
            if r.person is not None:
                rec["person"] = {"range": r.person.range_m, "lo": r.person.range_lo_m, "bearing": round(r.person.bearing_deg, 1),
                                 "facing": r.person.facing_deg, "conf": round(r.person.facing_conf, 2), "src": r.person.range_src}
            if r.target is not None:
                rec["target"] = {"cls": r.target.cls, "range": r.target.range_m, "bearing": round(r.target.bearing_deg, 1),
                                 "confirmed": r.target.confirmed}
            rec["n_det"] = len(r.detections)
        self._write(rec)

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
