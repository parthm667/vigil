"""Live viewer for the glasses hardware: video window + ToF/haptic overlay.

This is the GUI bench tool. It deliberately contains NO protocol code of its own -- it
drives GlassesLink and GlassesVideoSource from reachglass_glasses.py, which is the single
implementation of protocol v1.

An earlier version of this file spoke the wire protocol directly and drifted out of sync
with the firmware: it sent four-servo raw-angle commands, which bypass the Nano's
press-travel safety clamp, and parsed a four-field telemetry line that no longer exists.
Do not reintroduce that. If you need something the hardware can do, add a method to
GlassesLink so there stays exactly one place the protocol is written down.

    pip install opencv-python numpy
    python rover_client.py                      # auto-discover the CAM
    python rover_client.py --cam-ip 192.168.4.1 --nano-ip 192.168.4.50

Keys
    a       cue GO LEFT   (D,-1 -> presses the RIGHT pad)
    d       cue GO RIGHT  (D,+1 -> presses the LEFT pad)
    space   no cue / release both (D,0)
    p       pulse both pads twice -- the "arrived" cue
    q       quit (releases the pads on the way out)

The sides are crossed on purpose: the pad opposite the turn presses, so the wearer
is nudged from the far side toward where they should go.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reachglass_glasses import GlassesLink, GlassesVideoSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Glasses hardware live viewer")
    # Two boards, two addresses: the CAM serves video, the Nano serves data.
    # Do NOT pass one to the other -- the Nano has no web server and the CAM
    # has no telemetry.
    ap.add_argument("--cam-ip", default=None, help="ESP32-CAM (video), default 192.168.4.1")
    ap.add_argument("--nano-ip", default=None, help="Nano ESP32 (data), default 192.168.4.50")
    args = ap.parse_args()

    with GlassesLink(host=args.nano_ip) as link:
        cam = GlassesVideoSource(host=args.cam_ip).start()
        print(f"data: {link.host}   video: {cam.url}")

        if cam.wait_first(10.0) is None:
            print(f"no video after 10 s -- check {cam.url.replace('/stream', '/health')}")
            cam.stop()
            return 1

        win = "glasses"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cue = 0          # -1 left, 0 none, +1 right
        last_seq = -1

        try:
            while True:
                frame = cam.read()
                if frame is not None and frame.seq != last_seq:
                    last_seq = frame.seq
                    img = frame.image.copy()   # frames are read-only by contract
                    st = link.state()          # state() is a method: it returns a snapshot

                    rows = (
                        f"ToF  L {st.tof_left_mm:>5} mm    R {st.tof_right_mm:>5} mm",
                        f"pads L {st.press_left:>3} %     R {st.press_right:>3} %",
                        f"cue {('LEFT ' if cue < 0 else 'RIGHT' if cue > 0 else 'none ')}"
                        f"  link {st.link}   lost {st.packets_lost}"
                        + ("   STALE" if st.stale else ""),
                    )
                    colour = (0, 165, 255) if st.stale else (0, 255, 0)
                    for i, row in enumerate(rows):
                        cv2.putText(img, row, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, colour, 1, cv2.LINE_AA)
                    cv2.imshow(win, img)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("a"):
                    cue = -1
                    link.direction(cue)
                elif k == ord("d"):
                    cue = 1
                    link.direction(cue)
                elif k == ord(" "):
                    cue = 0
                    link.direction(cue)
                elif k == ord("p"):
                    link.pulse("B", count=2, on_ms=120, off_ms=120, depth=80)
        finally:
            cam.stop()   # GlassesLink's context manager releases the pads
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
