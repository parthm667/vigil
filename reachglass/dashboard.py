"""Operator dashboard: camera view with overlays | top-down mission map | status line.

render(ctx, mission) -> BGR image. Draws on a COPY (frames are shared and read-only). Call from the main
thread only (macOS requires GUI calls there).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

CAM_W, CAM_H = 640, 480
MAP_W = 480
BAR_H = 70
COLORS = {"person": (255, 200, 0), "target": (60, 220, 60), "cand": (0, 140, 255), "context": (0, 215, 255),
          "text": (240, 240, 240), "dim": (150, 150, 150), "warn": (0, 80, 255)}


def _txt(img, s, org, scale=0.5, color=COLORS["text"], thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def camera_panel(ctx) -> np.ndarray:
    if ctx.frame is None:
        img = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        _txt(img, "waiting for video...", (20, 40), 0.8)
        return img
    src = ctx.frame.image
    sx, sy = CAM_W / src.shape[1], CAM_H / src.shape[0]
    img = cv2.resize(src, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)  # a copy: safe to draw on
    res = ctx.res
    cv2.drawMarker(img, (CAM_W // 2, CAM_H // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    if res is None:
        return img

    def box(d, color, thick=1):
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), color, thick)

    for o in res.context:
        box(o.det, COLORS["context"])
        _txt(img, o.cls, (int(o.det.bbox[0] * sx), int(o.det.bbox[1] * sy) - 4), 0.4, COLORS["context"])
    for p in res.persons:
        box(p.det, COLORS["person"], 3 if (res.person is not None and p.det.track_id == res.person.det.track_id) else 1)
    if res.person is not None:
        p = res.person
        s = f"person {p.range_m:.1f}m" if p.range_m else "person"
        if p.facing_deg is not None:
            s += f" facing {p.facing_deg:+.0f} ({p.facing_conf:.1f})"
        _txt(img, s, (int(p.det.bbox[0] * sx), max(14, int(p.det.bbox[1] * sy) - 6)), 0.5, COLORS["person"])
        # keypoints
        if p.det.keypoints is not None:
            for x, y, c in p.det.keypoints:
                if c > 0.4:
                    cv2.circle(img, (int(x * sx), int(y * sy)), 3, COLORS["person"], -1)
    for t in res.targets:
        locked = res.target is not None and t.det.track_id == res.target.det.track_id
        color = COLORS["target"] if (locked and t.confirmed) else COLORS["cand"]
        box(t.det, color, 3 if locked else 1)
        s = f"{t.cls} {t.range_m:.2f}m {t.bearing_deg:+.0f}deg" if t.range_m else t.cls
        _txt(img, s + (" OK" if locked and t.confirmed else ""), (int(t.det.bbox[0] * sx), int(t.det.bbox[3] * sy) + 16), 0.5, color)
    return img


def map_panel(ctx, mission, scale: float = 45.0) -> np.ndarray:
    img = np.full((CAM_H, MAP_W, 3), 30, np.uint8)
    g = ctx.grid
    pose = ctx.odom.pose
    cx, cy = MAP_W // 2, CAM_H // 2

    def px(x, y):  # mission frame (x forward, y right) -> image, x up, y right, centred on the drone
        return int(cx + (y - pose.y) * scale), int(cy - (x - pose.x) * scale)

    n = g.n
    step = max(1, int(round(1.0 / (g.cell * scale / 4))))
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            if g.seen[i, j] == 0 and g.free[i, j] == 0 and g.blocked[i, j] < 1:
                continue
            x, y = g.center(i, j)
            u, v = px(x, y)
            if not (0 <= u < MAP_W and 0 <= v < CAM_H):
                continue
            h = int(g.cell * scale / 2) + 1
            color = (60, 60, 60)
            if g.free[i, j] > 0:
                color = (60, 110, 60)
            if g.blocked[i, j] >= 1:
                color = (40, 40, 160) if np.isinf(g.top[i, j]) else (40, 90, 150)
            cv2.rectangle(img, (u - h, v - h), (u + h, v + h), color, -1)
    del step
    # 1 m grid ticks
    for k in range(-10, 11):
        u, _ = px(pose.x, pose.y + k)
        _, v = px(pose.x + k, pose.y)
        if 0 <= u < MAP_W:
            cv2.line(img, (u, 0), (u, CAM_H), (45, 45, 45), 1)
        if 0 <= v < CAM_H:
            cv2.line(img, (0, v), (MAP_W, v), (45, 45, 45), 1)
    # path and vantages
    pts = [px(x, y) for x, y in ctx.odom.path]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (200, 200, 200), 1)
    for vx, vy in ctx.vantages:
        cv2.circle(img, px(vx, vy), 5, (200, 200, 200), 1)
    # memory objects
    for o in ctx.memory.objects:
        x, y = o.xy
        if o.cls == "person":
            continue
        color = COLORS["target"] if o.cls == ctx.target_cls else COLORS["context"]
        cv2.circle(img, px(x, y), 6 if o.cls == ctx.target_cls else 4, color, -1 if o.confirmed else 1)
        _txt(img, o.cls, (px(x, y)[0] + 6, px(x, y)[1] - 6), 0.4, color)
    # person when the query came
    if ctx.person_origin is not None:
        u, v = px(*ctx.person_origin)
        cv2.circle(img, (u, v), 8, COLORS["person"], 2)
        if ctx.person_heading is not None:
            h = math.radians(ctx.person_heading)
            cv2.arrowedLine(img, (u, v), px(ctx.person_origin[0] + 0.6 * math.cos(h), ctx.person_origin[1] + 0.6 * math.sin(h)),
                            COLORS["person"], 2, tipLength=0.3)
    # guidance: person -> target
    gd = getattr(mission, "guidance", None)
    if gd is not None and gd.person_xy is not None:
        cv2.arrowedLine(img, px(*gd.person_xy), px(*gd.target_xy), (60, 220, 60), 2, tipLength=0.08)
    # drone
    u, v = px(pose.x, pose.y)
    h = math.radians(pose.heading_deg)
    tip = px(pose.x + 0.35 * math.cos(h), pose.y + 0.35 * math.sin(h))
    left = px(pose.x + 0.15 * math.cos(h + 2.4), pose.y + 0.15 * math.sin(h + 2.4))
    right = px(pose.x + 0.15 * math.cos(h - 2.4), pose.y + 0.15 * math.sin(h - 2.4))
    cv2.fillPoly(img, [np.array([tip, left, right])], (255, 255, 255))
    _txt(img, "map (1 m grid): green free, red blocked, grey seen", (8, CAM_H - 10), 0.4, COLORS["dim"])
    return img


def status_bar(ctx, mission, fps: float | None = None, extra: str = "") -> np.ndarray:
    bar = np.full((BAR_H, CAM_W + MAP_W, 3), 18, np.uint8)
    st = mission.status if mission is not None else ""
    _txt(bar, st[:110], (10, 22), 0.55, (255, 255, 255))
    tel = ctx.tel
    parts = []
    if tel is not None:
        if tel.floor_altitude_m() is not None:
            parts.append(f"alt {tel.floor_altitude_m():.2f} m")
        if tel.battery_pct is not None:
            parts.append(f"bat {tel.battery_pct:.0f}%")
        if tel.yaw_deg is not None:
            parts.append(f"yaw {tel.yaw_deg:+.0f}")
    if ctx.res is not None:
        parts.append(f"perception {ctx.res.latency_ms:.0f} ms")
    if fps:
        parts.append(f"{fps:.0f} fps")
    if ctx.target_cls:
        parts.append(f"target: {ctx.target_cls}")
    gd = getattr(mission, "guidance", None)
    _txt(bar, "   ".join(parts) + ("   " + extra if extra else ""), (10, 44), 0.5, COLORS["dim"])
    if gd is not None:
        _txt(bar, gd.text[:120], (10, 64), 0.5, (60, 220, 60))
    elif mission is not None and mission.said:
        _txt(bar, "said: " + mission.said[-1][1][:110], (10, 64), 0.5, (200, 200, 120))
    return bar


def render(ctx, mission, fps: float | None = None, extra: str = "") -> np.ndarray:
    top = np.hstack([camera_panel(ctx), map_panel(ctx, mission)])
    return np.vstack([top, status_bar(ctx, mission, fps, extra)])
