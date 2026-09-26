"""ReachGlass main program.

  python -m reachglass.app sim                           # whole mission on the simulator (dashboard window)
  python -m reachglass.app sim --headless --query "find my water bottle" --at 20 --seconds 150 --record sim.mp4
  python -m reachglass.app tello --dry-run               # REAL video + telemetry, NO motion commands sent
                                                         # (hold the drone at head height behind someone)
  python -m reachglass.app tello                         # fly

Queries: type a sentence in this terminal and press Enter (e.g. "find my water bottle", "follow me",
"what's around me", "stop", "land"), or send UDP datagrams to --udp (default 5005) from the voice app.

On the real drone nothing happens until you type "takeoff" (or press t): check the dashboard first.

Keys (dashboard window focused):  t takeoff   SPACE hold / resume   l land   e EMERGENCY (motors off, it falls)
                                  q land and quit                    Ctrl+C in the terminal: land and quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .behaviors import Ctx, sense
from .config import load_config
from .mission import Mission, MultiInbox, ScriptedInbox, StdinInbox, UdpInbox
from .query import KeywordQueryParser
from .runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def say(text: str) -> None:
    print(f"\n>>> {text}\n", flush=True)


class Window:
    """OpenCV window + optional video recorder (main thread only)."""

    def __init__(self, headless: bool, record: str | None, name: str = "ReachGlass"):
        self.headless, self.name = headless, name
        self.writer = None
        self.record = record
        self._t = []

    def show(self, img) -> int:
        import cv2

        if self.record:
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            self.writer.write(img)
        if self.headless:
            return -1
        cv2.imshow(self.name, img)
        return cv2.waitKey(1) & 0xFF

    def fps(self) -> float | None:
        now = time.time()
        self._t = [t for t in self._t if now - t < 2.0] + [now]
        return (len(self._t) - 1) / 2.0 if len(self._t) > 1 else None

    def close(self) -> None:
        import cv2

        if self.writer is not None:
            self.writer.release()
        if not self.headless:
            cv2.destroyAllWindows()


def handle_key(key: int, mission: Mission, drone) -> bool:
    """Returns False to quit."""
    if key in (-1, 255):
        return True
    ch = chr(key).lower()
    if ch == " ":
        if mission.state == "HOLD":
            mission.resume()
        else:
            mission.hold()
    elif ch == "l":
        mission.force_land("operator")
    elif ch == "e":
        drone.emergency()
        mission.force_land("EMERGENCY")
    elif ch == "t" and mission.state == "IDLE":
        mission.start()
    elif ch == "q":
        mission.force_land("quit")
        return False
    return True


def run_sim(args, cfg) -> int:
    from .sim.runner import SimRunner

    items = [(args.at, q) for q in args.query]
    inbox = MultiInbox(ScriptedInbox(items), StdinInbox() if not args.headless and sys.stdin.isatty() else None)
    r = SimRunner(cfg, inbox=inbox, seed=args.seed, announce=say)
    win = Window(args.headless, args.record, "ReachGlass (simulator)")
    log = RunLog(args.log)
    from .dashboard import render

    try:
        while r.t < args.seconds:
            r.step()
            log.tick(r.ctx, r.mission)
            if not args.headless or args.record:
                key = win.show(render(r.ctx, r.mission, extra=f"sim t={r.t:5.1f}s  collisions={r.sim.drone.collisions}"))
                if not handle_key(key, r.mission, r.drone):
                    break
            if args.realtime:
                time.sleep(r.dt)
            if r.mission.state == "LANDED" and r.t > 5:
                break
    finally:
        win.close()
        log.close()
    print(f"sim done: t={r.t:.1f}s state={r.mission.state} collisions={r.sim.drone.collisions}")
    for t, s, why in r.mission.history:
        print(f"  {t:6.1f}s {s:10s} {why}")
    return 0


def run_tello(args, cfg) -> int:
    from .drone import SafetyGovernor, make_tello
    from .perception import Perception

    tello = make_tello(cfg, dry_run=args.dry_run)
    dry = tello.dry_run  # the flag OR drone.kind: dry_run in the config
    if args.no_takeoff and not dry:
        print("--no-takeoff is only allowed with --dry-run: on a live drone it would switch off the landing and "
              "safety logic that depends on knowing the drone took off")
        return 4
    drone = SafetyGovernor(tello, cfg.safety)
    print("connecting to the Tello (join its Wi-Fi first)...")
    drone.connect()
    tel = drone.telemetry()
    print(f"battery {tel.battery_pct}%  {'DRY RUN: no motion commands will be sent' if dry else 'LIVE: it will fly'}")
    if not dry and tel.battery_pct is not None and tel.battery_pct < cfg.safety.min_battery_pct + 10:
        print("battery too low to start a mission")
        drone.close()
        return 2
    print("loading models...")
    perception = Perception.from_config(cfg)
    print(f"can look for: {', '.join(perception.vocabulary())}")
    source = drone.frame_source()
    if source.wait_first(10.0) is None:
        print("no video from the drone after 10 s")
        drone.close()
        return 3
    ctx = Ctx(cfg, drone, perception)
    mission = Mission(ctx, KeywordQueryParser(), announce=say)
    udp = UdpInbox(args.udp) if args.udp else None
    inbox = MultiInbox(StdinInbox(), udp, ScriptedInbox([(time.time() + args.at, q) for q in args.query]))
    win = Window(args.headless, args.record, "ReachGlass (Tello)" + (" DRY RUN" if dry else ""))
    log = RunLog(args.log)
    from .dashboard import render

    period = 1.0 / args.hz
    mission.start(wait_for_operator=not args.autostart)
    print("type 'takeoff' (or press t in the window), then a request, e.g.: find my water bottle")
    try:
        while True:
            t0 = time.time()
            sense(ctx, source, t0)
            last_frame_t = ctx.frame.t if ctx.frame is not None else None
            if drone.check(t0, last_frame_t) == "land":
                mission.force_land("safety: " + (drone.events[-1][1] if drone.events else ""))
            for text in inbox.poll(t0):
                print(f"heard: {text}")
                log.event(t0, "query", text=text)
                mission.query(text)
            mission.step()
            log.tick(ctx, mission)
            key = win.show(render(ctx, mission, win.fps(), "DRY RUN" if dry else ""))
            if not handle_key(key, mission, drone):
                # land, wait for it, then leave
                t_end = time.time() + 12
                while (drone.busy() or drone.flying) and time.time() < t_end:
                    sense(ctx, source, time.time())
                    mission.step()
                    win.show(render(ctx, mission))
                    time.sleep(0.05)
                break
            if mission.state == "LANDED" and not drone.flying and args.quit_on_land:
                break
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing")
    finally:
        try:
            if drone.flying:
                drone.land()
                t_end = time.time() + 10
                while drone.busy() and time.time() < t_end:
                    time.sleep(0.05)
        finally:
            win.close()
            log.close()
            if udp is not None:
                udp.close()
            drone.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["sim", "tello"])
    p.add_argument("--config", help="YAML overrides (see reachglass/config.py)")
    p.add_argument("--dry-run", action="store_true", help="tello: video + telemetry only, motion commands NOT sent")
    p.add_argument("--no-takeoff", action="store_true", help="start directly in FOLLOW (drone held / already flying)")
    p.add_argument("--autostart", action="store_true", help="tello: take off immediately (default: wait for 'takeoff')")
    p.add_argument("--headless", action="store_true", help="no window")
    p.add_argument("--record", help="save the dashboard to this .mp4")
    p.add_argument("--query", action="append", default=[], help="send this request automatically (repeatable)")
    p.add_argument("--at", type=float, default=20.0, help="seconds after start to send --query")
    p.add_argument("--seconds", type=float, default=240.0, help="sim: stop after this much simulated time")
    p.add_argument("--realtime", action="store_true", help="sim: run at wall-clock speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--udp", type=int, default=5005, help="UDP port for queries (0 = off)")
    p.add_argument("--hz", type=float, default=20.0, help="tello: control loop rate")
    p.add_argument("--quit-on-land", action="store_true")
    p.add_argument("--log", help="JSONL run log (default runs/<mode>_<time>.jsonl)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")
    overrides = {"mission": {"takeoff": False}} if args.no_takeoff else None
    cfg = load_config(args.config, overrides)
    if args.log is None:
        args.log = str(ROOT / "runs" / f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    return run_sim(args, cfg) if args.mode == "sim" else run_tello(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
