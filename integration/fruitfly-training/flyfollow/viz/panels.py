"""2D panels drawn with numpy + PIL: drone camera, fly's-eye strip, top-down map, trace strips.

Used by the MP4 composite and logged as images into Rerun. All take a frame dict (flyfollow.viz.frames).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from flyfollow.interfaces import IMG_H, IMG_W, N_AZIMUTH_BINS
from flyfollow.viz import draw
from flyfollow.viz.frames import default_bin_centers_deg, lc10a_bin_rates, look_bearing

NB = N_AZIMUTH_BINS


def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]


# --------------------------------------------------------------------------- drone camera
class CameraPanel:
    """The drone's view: the real frame when the frame carries one, else a synthetic scene with the tracked box."""

    def __init__(self, w: int = 480, h: int = 360):
        self.w, self.h = w, h
        bg = draw.vgradient(h, w, (20, 24, 33), (9, 11, 15)).copy()
        # perspective floor grid for a sense of space
        hy = int(0.46 * h)
        acc = np.zeros((h, w, 3), np.float32)
        for k in range(-12, 13):
            xs = np.linspace(w / 2 + k * 8, w / 2 + k * 90, 200)
            ys = np.linspace(hy, h, 200)
            draw.splat(acc, xs, ys, np.asarray(draw.GRID, np.float32) * 0.5)
        for j in range(1, 9):
            y = hy + (h - hy) * (j / 8) ** 1.8
            draw.splat(acc, np.linspace(0, w, 400), np.full(400, y), np.asarray(draw.GRID, np.float32) * 0.5)
        self.bg = np.clip(bg + acc, 0, 255).astype(np.uint8)

    def render(self, frame: dict) -> np.ndarray:
        w, h = self.w, self.h
        img = frame.get("image")
        sx, sy = w / IMG_W, h / IMG_H
        out = _resize_nearest(np.asarray(img, np.uint8), h, w).copy() if img is not None else self.bg.copy()
        tg = frame.get("target") or {}
        with draw.TextBatch(out) as tb:
            # crosshair at the image center
            cx0, cy0 = w / 2, h / 2
            tb.line([(cx0 - 10, cy0), (cx0 + 10, cy0)], draw.MUTED, alpha=0.6)
            tb.line([(cx0, cy0 - 10), (cx0, cy0 + 10)], draw.MUTED, alpha=0.6)
            if tg.get("valid") and tg.get("h", 0) > 0:
                bx, by, bh = tg["cx"] * sx, tg["cy"] * sy, tg["h"] * sy
                bw = 0.8 * bh
                if img is None:  # stylized person under the head box
                    tb.ellipse((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2), fill=(60, 70, 88), alpha=0.95)
                    tb.rect((bx - 1.1 * bw, by + 0.6 * bh, bx + 1.1 * bw, by + 3.6 * bh), fill=(44, 52, 66), radius=int(0.4 * bw) + 1)
                tb.rect((bx - bw / 2 - 3, by - bh / 2 - 3, bx + bw / 2 + 3, by + bh / 2 + 3), outline=draw.TARGET, width=2)
                lab = "person"
                if tg.get("range_m") is not None:
                    lab += f"  {tg['range_m']:.1f} m"
                tb.text((bx - bw / 2 - 3, by - bh / 2 - 8), lab, size=12, color=draw.TARGET, anchor="ls", shadow=True)
            elif tg:
                tb.text((w / 2, h / 2 + 30), "target lost", size=14, color=(255, 120, 120), anchor="mm", shadow=True)
            tb.text((12, 10), "DRONE CAMERA", size=13, bold=True, shadow=True)
            src = (frame.get("meta") or {}).get("source", "")
            sub = "Tello video" if img is not None else ("simulated view of the tracked box" if src != "drone" else "no video")
            tb.text((12, 28), sub, size=11, color=draw.MUTED, shadow=True)
            if tg.get("bearing_deg") is not None:
                tb.text((w - 12, 10), f"bearing {tg['bearing_deg']:+5.1f}°", size=12, color=draw.TEXT, anchor="ra", kind="mono", shadow=True)
        return out


# --------------------------------------------------------------------------- fly's-eye strip
class FlyEyePanel:
    """Compound-eye readout: LC10a activity per azimuth bin, left eye on top, right eye below.

    Cell x position = the encoder's bin center (the azimuth each bin is tuned to). The green marker is the
    target bearing the camera measured; the white marker is the LC10a activity centroid (where the fly "sees" it).
    """

    def __init__(self, w: int = 480, h: int = 150, span_deg: float = 55.0):
        self.w, self.h, self.span = w, h, span_deg

    def x_of(self, deg: float) -> float:
        return self.w / 2 + (deg / self.span) * (self.w / 2 - 30)

    def render(self, frame: dict, bins: dict | None = None, rate_ref_hz: float = 60.0) -> np.ndarray:
        w, h = self.w, self.h
        out = np.empty((h, w, 3), np.uint8)
        out[:] = draw.PANEL
        rates = lc10a_bin_rates(frame, bins)
        centers = np.asarray(frame.get("bin_centers_deg", default_bin_centers_deg()), float)
        lb = look_bearing(rates, centers)
        tg = frame.get("target") or {}
        r = max(8.0, min(16.0, (w - 60) / 40))
        y_first = max(62.0, 0.5 * h - r - 16)
        rows = {0: y_first, 1: y_first + 2 * r + 6}
        with draw.TextBatch(out) as tb:
            tb.text((12, 10), "FLY'S-EYE VIEW", size=13, bold=True)
            tb.text((12, 28), "LC10a spikes per azimuth bin (top: left eye, bottom: right eye)", size=11, color=draw.MUTED)
            for s_i, (row_y, col) in enumerate(((rows[0], draw.LEFT), (rows[1], draw.LC10A))):
                for k in range(NB):
                    v = float(np.clip(rates[s_i * NB + k] / rate_ref_hz, 0, 1))
                    x = self.x_of(centers[s_i * NB + k])
                    c = draw.lerp_color((26, 32, 44), draw.LC10A, v) + (255 - draw.lerp_color((26, 32, 44), draw.LC10A, v)) * (0.6 * v * v)
                    hexpts = [(x + r * math.cos(math.pi / 6 + j * math.pi / 3), row_y + r * math.sin(math.pi / 6 + j * math.pi / 3)) for j in range(6)]
                    tb.polygon(hexpts, fill=tuple(int(q) for q in c), outline=(50, 60, 78))
                tb.text((w - 10, row_y), "L" if s_i == 0 else "R", size=11, color=draw.MUTED, anchor="rm")
            ya = rows[1] + r + 14
            for deg in (-45, -30, -15, 0, 15, 30, 45):
                x = self.x_of(deg)
                tb.line([(x, ya - 3), (x, ya + 1)], draw.GRID)
                tb.text((x, ya + 4), f"{deg:+d}°" if deg else "0°", size=10, color=draw.MUTED, anchor="ma")
            if tg.get("valid") and tg.get("bearing_deg") is not None:
                x = self.x_of(float(np.clip(tg["bearing_deg"], -self.span, self.span)))
                tb.polygon([(x, rows[0] - r - 4), (x - 6, rows[0] - r - 13), (x + 6, rows[0] - r - 13)], fill=draw.TARGET)
            if lb is not None:
                x = self.x_of(float(np.clip(math.degrees(lb), -self.span, self.span)))
                tb.line([(x, rows[0] - r), (x, rows[1] + r)], (255, 255, 255), width=2, alpha=0.8)
            if h >= 200:
                yb = h - 22
                cam_s = f"{tg['bearing_deg']:+5.1f}\u00b0" if tg.get("valid") and tg.get("bearing_deg") is not None else "  --  "
                fly_s = f"{math.degrees(lb):+5.1f}\u00b0" if lb is not None else "  --  "
                tb.polygon([(14, yb - 5), (8, yb - 12), (20, yb - 12)], fill=draw.TARGET)
                tb.text((26, yb - 8), f"camera bearing {cam_s}", size=11, color=draw.TEXT, anchor="lm", kind="mono")
                tb.line([(w / 2 + 10, yb - 14), (w / 2 + 10, yb - 2)], (255, 255, 255), width=2)
                tb.text((w / 2 + 20, yb - 8), f"LC10a centroid {fly_s}", size=11, color=draw.TEXT, anchor="lm", kind="mono")
        return out


# --------------------------------------------------------------------------- top-down map
class TopDownPanel:
    """Top view of the simulated drone, its camera wedge, the person and the standoff ring (sim frames only)."""

    def __init__(self, w: int = 480, h: int = 360, trail_s: float = 12.0):
        self.w, self.h = w, h
        self.drone_trail: deque = deque(maxlen=int(trail_s * 20))
        self.person_trail: deque = deque(maxlen=int(trail_s * 20))
        self.center = None
        self.bg = draw.radial_vignette(h, w, (16, 20, 28), (8, 10, 14))

    def reset(self) -> None:
        self.drone_trail.clear()
        self.person_trail.clear()
        self.center = None

    def push(self, frame: dict) -> None:
        """Advance the trails and the (smoothed) view center with one tick."""
        sim = frame.get("sim")
        if not sim:
            return
        dx, dy = sim["drone_xy"]
        px, py = sim["person_xy"]
        self.drone_trail.append((dx, dy))
        self.person_trail.append((px, py))
        mid = np.array([(dx + px) / 2, (dy + py) / 2])
        self.center = mid if self.center is None else self.center + 0.08 * (mid - self.center)

    def render(self, frame: dict) -> np.ndarray:
        w, h = self.w, self.h
        out = self.bg.copy()
        sim = frame.get("sim")
        with draw.TextBatch(out) as tb:
            if not sim:
                tb.text((12, 10), "TOP VIEW", size=13, bold=True)
                tb.text((12, 28), "simulation only", size=11, color=draw.MUTED)
                return out
            dx, dy = sim["drone_xy"]
            px, py = sim["person_xy"]
            if self.center is None:
                self.center = np.array([(dx + px) / 2, (dy + py) / 2])
            scale = min(w, h) / 6.0  # px per meter, 6 m across

            def P(x, y):  # world x forward (up on screen), y left (left on screen)
                return (w / 2 - (y - self.center[1]) * scale, h / 2 + 18 - (x - self.center[0]) * scale)

            # 1 m grid
            gx0 = math.floor(self.center[0] - 4)
            gy0 = math.floor(self.center[1] - 5)
            for k in range(12):
                a, b = P(gx0 + k, gy0), P(gx0 + k, gy0 + 11)
                tb.line([a, b], draw.GRID, alpha=0.5)
                a, b = P(gx0, gy0 + k), P(gx0 + 11, gy0 + k)
                tb.line([a, b], draw.GRID, alpha=0.5)
            z_ref = (frame.get("target") or {}).get("z_ref_m")
            if z_ref:
                cxp, cyp = P(px, py)
                rr = z_ref * scale
                tb.ellipse((cxp - rr, cyp - rr, cxp + rr, cyp + rr), outline=draw.TARGET, alpha=0.25)
            if len(self.person_trail) > 1:
                tb.line([P(*q) for q in self.person_trail], draw.TARGET, width=2, alpha=0.45)
            if len(self.drone_trail) > 1:
                tb.line([P(*q) for q in self.drone_trail], draw.DRONE, width=2, alpha=0.35)
            psi = math.radians(sim.get("drone_psi_deg", 0.0))
            hf = math.radians(sim.get("hfov_deg", 55.0) / 2)
            L = 2.2
            wedge = [P(dx, dy), P(dx + L * math.cos(psi + hf), dy + L * math.sin(psi + hf)),
                     P(dx + L * math.cos(psi - hf), dy + L * math.sin(psi - hf))]
            tb.polygon(wedge, fill=(120, 160, 220), alpha=0.12)
            tri = [P(dx + 0.25 * math.cos(psi), dy + 0.25 * math.sin(psi)),
                   P(dx + 0.14 * math.cos(psi + 2.5), dy + 0.14 * math.sin(psi + 2.5)),
                   P(dx + 0.14 * math.cos(psi - 2.5), dy + 0.14 * math.sin(psi - 2.5))]
            tb.polygon(tri, fill=draw.DRONE)
            cxp, cyp = P(px, py)
            tb.ellipse((cxp - 7, cyp - 7, cxp + 7, cyp + 7), fill=draw.TARGET)
            tb.rect((0, 0, w, 44), fill=draw.PANEL, alpha=0.85)
            tb.text((12, 10), "TOP VIEW", size=13, bold=True)
            tb.text((12, 28), "drone (white) following the person (green)", size=11, color=draw.MUTED)
            rng = (frame.get("target") or {}).get("range_m")
            if rng is not None:
                tb.text((w - 12, 10), f"range {rng:4.2f} m", size=12, color=draw.TEXT, anchor="ra", kind="mono")
                if z_ref:
                    tb.text((w - 12, 28), f"standoff {z_ref:.2f} m", size=11, color=draw.MUTED, anchor="ra", kind="mono")
        return out


# --------------------------------------------------------------------------- trace strip
class TracePanel:
    """Scrolling traces: DNa02 L / R rates (the brain's output) and the yaw stick (what the drone was told)."""

    def __init__(self, w: int = 480, h: int = 220, window_s: float = 10.0):
        self.w, self.h, self.window = w, h, window_s
        self.hist: deque = deque(maxlen=int(window_s * 20) + 2)

    def reset(self) -> None:
        self.hist.clear()

    def push(self, frame: dict) -> None:
        dn = frame.get("dn") or {}
        st = frame.get("sticks") or {}
        self.hist.append((frame["t"], dn.get("DNa02_L", 0.0), dn.get("DNa02_R", 0.0), st.get("yaw", 0.0), st.get("fb", 0.0)))

    def render(self) -> np.ndarray:
        w, h = self.w, self.h
        out = np.empty((h, w, 3), np.uint8)
        out[:] = draw.PANEL
        H = np.array(self.hist) if self.hist else np.zeros((0, 5))
        top, mid, bot = 44, 44 + (h - 60) * 0.58, h - 14
        x0, x1 = 50, w - 12
        with draw.TextBatch(out) as tb:
            tb.text((12, 10), "BRAIN OUTPUT -> DRONE", size=13, bold=True)
            tb.text((12, 28), "DNa02 left / right spike rate, and the yaw stick sent", size=11, color=draw.MUTED)
            tb.line([(x0, mid - 6), (x1, mid - 6)], draw.GRID)
            ymid = (mid + 6 + bot) / 2
            tb.line([(x0, ymid), (x1, ymid)], draw.GRID)
            tb.text((x0 - 6, top), "400", size=9, color=draw.MUTED, anchor="ra", kind="mono")
            tb.text((x0 - 6, mid - 6), "0 Hz", size=9, color=draw.MUTED, anchor="rs", kind="mono")
            tb.text((x0 - 6, ymid), "yaw", size=9, color=draw.MUTED, anchor="rm", kind="mono")
            if len(H) > 1:
                t1 = H[-1, 0]
                xs = x0 + (H[:, 0] - (t1 - self.window)) / self.window * (x1 - x0)

                def ydn(v):
                    return (mid - 6) - np.clip(v, 0, 400) / 400.0 * (mid - 6 - top)

                def yyaw(v):
                    return ymid - np.clip(v, -60, 60) / 60.0 * ((bot - mid - 6) / 2)

                tb.line(list(zip(xs, ydn(H[:, 1]))), draw.LEFT, width=2)
                tb.line(list(zip(xs, ydn(H[:, 2]))), draw.RIGHT, width=2)
                tb.line(list(zip(xs, yyaw(H[:, 3]))), (240, 240, 240), width=2)
                last = H[-1]
                tb.text((x1, top - 2), f"L {last[1]:3.0f} Hz", size=11, color=draw.LEFT, anchor="rs", kind="mono")
                tb.text((x1 - 90, top - 2), f"R {last[2]:3.0f} Hz", size=11, color=draw.RIGHT, anchor="rs", kind="mono")
                arrow = "turn right" if last[3] > 3 else ("turn left" if last[3] < -3 else "straight")
                tb.text((x1, bot), f"yaw {last[3]:+4.0f}  {arrow}", size=11, color=draw.TEXT, anchor="rs", kind="mono")
        return out
