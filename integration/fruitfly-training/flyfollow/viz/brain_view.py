"""Brain view: MaleCNS somata as a point cloud, the pursuit subgraph lit by live spikes, and the
LC10a -> AOTU025 / AOTU012 / AOTU019 -> DNa02 push-pull pathway drawn from the connectome.

Positions are real: each subgraph neuron sits at its MaleCNS soma location (somaLocation, 8 nm voxels,
shown in micrometers). The faint cloud is a random subsample of all MaleCNS brain somata. Neurons
without a soma location use their type's mean (per side when possible).

Display frame (right-handed, micrometers, centered on the brain): +X = the fly's RIGHT, +Y = anterior,
+Z = dorsal. The default camera looks from behind and above, so the fly's right is on screen right,
like the drone camera and the fly body view.

Two renderers share one BrainGeometry and one BrainActivity:
- log_static_rerun / log_frame_rerun: native Rerun 3D entities (interactive).
- BrainRenderer.render: a fast numpy splat renderer with bloom, for the MP4 composite and PNG stills.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyfollow.interfaces import OUTPUT_GROUPS, data_root
from flyfollow.viz import draw

VOXEL_UM = 0.008  # MaleCNS voxel size (8 nm)
ANNOTATIONS = "malecns_v1/body-annotations-male-cns-v1.0-minconf-0.5.feather"

ROLE_NAMES = ("other", "LC10a", "LC9/LC11", "P1 arousal", "AOTU025/012 (excit.)", "AOTU019 (inhib.)", "DNa02", "other steering DNs")
R_OTHER, R_LC10A, R_AUX, R_AROUSAL, R_AOTU_EXC, R_AOTU_INH, R_DNA02, R_DN = range(8)
ROLE_COLORS = np.array([draw.RELAY, draw.LC10A, draw.LC_AUX, draw.AROUSAL, draw.AOTU_EXC, draw.AOTU_INH, draw.DNA02, draw.DN_OTHER], np.float32)
ROLE_BASE_RADIUS_UM = np.array([2.2, 2.6, 2.2, 2.2, 7.0, 7.0, 9.0, 5.0], np.float32)
ROLE_REF_SPIKES = np.array([2.0, 2.5, 2.0, 2.0, 3.0, 3.0, 5.0, 4.0], np.float32)  # decayed spikes that glow at 63 %

RELAY_EXC_TYPES = ("AOTU025", "AOTU012")
RELAY_INH_TYPES = ("AOTU019",)
EDGE_EXC, EDGE_INH = 0, 1  # sign of the synapse (excitatory ACh / inhibitory GABA)


@dataclass
class PathwayEdges:
    """Strongest connectome edges of the push-pull pathway (subgraph indices)."""

    pre: np.ndarray  # (E,)
    post: np.ndarray
    weight: np.ndarray  # signed synapse counts
    route: np.ndarray  # 0: through AOTU025/012 (excitatory route), 1: through AOTU019 (inhibitory route)
    stage: np.ndarray  # 0: LC10a -> AOTU, 1: AOTU -> DNa02
    pts: np.ndarray  # (E, S, 3) points along a gentle arc, display frame


@dataclass
class BrainGeometry:
    name: str
    path: str
    pos: np.ndarray  # (n, 3) display frame, micrometers
    has_soma: np.ndarray  # (n,) bool, False where the type mean was used
    types: np.ndarray
    sides: np.ndarray
    role: np.ndarray  # (n,) role code
    context: np.ndarray  # (m, 3) subsample of all MaleCNS brain somata
    edges: PathwayEdges
    key: dict[str, int]  # "AOTU019_L" -> neuron index, for single-cell pathway neurons
    groups: dict[str, np.ndarray]
    lc10a_bins: dict[str, list[np.ndarray]]
    labels: list[tuple[str, np.ndarray, tuple[int, int, int]]] = field(default_factory=list)
    center_em: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    # ------------------------------------------------------------------ building
    @classmethod
    def load(cls, brain_path: str | Path, annotations: str | Path | None = None, n_context: int = 20000,
             include_vnc: bool = False, seed: int = 0, edges_per_relay: int = 14) -> BrainGeometry:
        import pandas as pd

        from flyfollow.brain.build import ensure_flydrones

        ensure_flydrones()
        from flydrones.brain.connectome import Connectome

        from flyfollow.pilot.fly_brain import lc10a_bins, resolve_brain_path

        p = resolve_brain_path(brain_path)
        conn = Connectome.load(p)
        ann_path = Path(annotations) if annotations else data_root() / ANNOTATIONS
        df = pd.read_feather(ann_path, columns=["bodyId", "type", "somaLocation", "tosomaLocation", "superclass", "somaSide"])
        has = df["somaLocation"].notna().to_numpy()
        allpos = np.full((len(df), 3), np.nan, np.float64)
        allpos[has] = np.stack(df.loc[has, "somaLocation"].to_numpy()).astype(np.float64)
        sc = df["superclass"].astype(str).to_numpy()
        in_brain = has & ~np.char.startswith(sc.astype("U32"), "vnc") & (allpos[:, 2] < 55000)
        in_brain &= ~np.isin(sc, ("ascending_neuron", "efferent_ascending"))
        ctx_mask = has if include_vnc else in_brain
        center = np.nanmean(allpos[in_brain], axis=0)

        # subgraph soma positions: own soma, else toward-soma point, else type (+side) mean
        by_id = pd.Series(np.arange(len(df)), index=df["bodyId"].to_numpy())
        rows = by_id.reindex(conn.body_ids).to_numpy()
        n = conn.n
        em = np.full((n, 3), np.nan)
        ok = ~np.isnan(rows)
        ri = rows[ok].astype(np.int64)
        em[ok] = allpos[ri]
        miss = np.isnan(em).any(axis=1)
        if miss.any():
            tos = df["tosomaLocation"].to_numpy()
            for i in np.flatnonzero(miss & ok):
                v = tos[int(rows[i])]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    em[i] = np.asarray(v, np.float64)
        has_soma = ~np.isnan(em).any(axis=1)
        types = conn.types.astype(str)
        sides = conn.sides.astype(str)
        for i in np.flatnonzero(~has_soma):
            sel = (df["type"].to_numpy() == types[i]) & has
            same_side = sel & (df["somaSide"].astype(str).to_numpy() == sides[i])
            use = same_side if same_side.any() else sel
            em[i] = allpos[use].mean(axis=0) if use.any() else center
        pos = to_display(em, center)

        rng = np.random.default_rng(seed)
        ctx_idx = np.flatnonzero(ctx_mask)
        if ctx_idx.size > n_context:
            ctx_idx = rng.choice(ctx_idx, n_context, replace=False)
        context = to_display(allpos[ctx_idx], center)

        role = np.zeros(n, np.int8)
        groups = {k: np.asarray(v, np.int64) for k, v in conn.groups.items()}
        for s in "LR":
            role[groups.get(f"LC9_{s}", [])] = R_AUX
            role[groups.get(f"LC11_{s}", [])] = R_AUX
            role[groups.get(f"AROUSAL_{s}", [])] = R_AROUSAL
            role[groups.get(f"LC10a_{s}", [])] = R_LC10A
        for g in OUTPUT_GROUPS:
            role[groups.get(g, [])] = R_DN
        role[np.isin(types, RELAY_EXC_TYPES)] = R_AOTU_EXC
        role[np.isin(types, RELAY_INH_TYPES)] = R_AOTU_INH
        role[types == "DNa02"] = R_DNA02

        key = {}
        for t in RELAY_EXC_TYPES + RELAY_INH_TYPES + ("DNa02",):
            for s in "LR":
                ii = np.flatnonzero((types == t) & (sides == s))
                if ii.size:
                    key[f"{t}_{s}"] = int(ii[0])

        edges = _pathway_edges(conn, pos, groups, key, edges_per_relay)
        bins = {g: lc10a_bins(conn, g) for g in ("LC10a_L", "LC10a_R")}
        geo = cls(name=conn.name, path=str(p), pos=pos.astype(np.float32), has_soma=has_soma, types=types, sides=sides,
                  role=role, context=context.astype(np.float32), edges=edges, key=key, groups=groups, lc10a_bins=bins,
                  center_em=center)
        geo.labels = geo._labels()
        return geo

    def _labels(self) -> list[tuple[str, np.ndarray, tuple[int, int, int]]]:
        out = []
        for s, side in (("L", "left"), ("R", "right")):
            g = self.groups.get(f"LC10a_{s}")
            if g is not None and g.size:
                out.append((f"LC10a {s}", self.pos[g].mean(axis=0), draw.LC10A))
            for t, col in (("AOTU019", draw.AOTU_INH), ("DNa02", draw.DNA02)):
                if f"{t}_{s}" in self.key:
                    out.append((f"{t} {s}", self.pos[self.key[f'{t}_{s}']], col))
        return out

    def pathway_nodes(self) -> np.ndarray:
        return np.array(sorted(set(self.edges.pre.tolist()) | set(self.edges.post.tolist())), np.int64)


def to_display(em: np.ndarray, center: np.ndarray) -> np.ndarray:
    """EM voxels (x = fly left, y = ventral, z = posterior) -> display micrometers (X right, Y anterior, Z dorsal)."""
    d = (np.asarray(em, np.float64) - center) * VOXEL_UM
    return np.stack([-d[:, 0], -d[:, 2], -d[:, 1]], axis=1)


def _arc(a: np.ndarray, b: np.ndarray, n: int = 24, lift: float = 0.18) -> np.ndarray:
    """Quadratic Bezier from a to b bowing dorsally (up) by lift * length."""
    mid = 0.5 * (a + b)
    L = float(np.linalg.norm(b - a))
    ctrl = mid + np.array([0.0, 0.0, lift * L])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t**2 * b


def _pathway_edges(conn, pos: np.ndarray, groups: dict, key: dict, k: int) -> PathwayEdges:
    W = conn.weights.tocsr()  # (post, pre)
    lc = np.concatenate([groups.get("LC10a_L", np.zeros(0, np.int64)), groups.get("LC10a_R", np.zeros(0, np.int64))])
    dna02 = [key[f"DNa02_{s}"] for s in "LR" if f"DNa02_{s}" in key]
    pre, post, w, route, stage = [], [], [], [], []
    for name, relay in key.items():
        if name.startswith("DNa02"):
            continue
        r = 1 if name.startswith(RELAY_INH_TYPES) else 0
        row = W[relay].toarray().ravel()
        win = row[lc]
        top = np.argsort(-win)[:k]
        for j in top:
            if win[j] > 0:
                pre.append(int(lc[j])); post.append(relay); w.append(float(win[j])); route.append(r); stage.append(0)
        for dn in dna02:
            v = float(W[dn, relay])
            if abs(v) >= 5:
                pre.append(relay); post.append(dn); w.append(v); route.append(r); stage.append(1)
    pre_a, post_a = np.array(pre, np.int64), np.array(post, np.int64)
    pts = np.stack([_arc(pos[a], pos[b], lift=0.12 if s == 0 else 0.35) for a, b, s in zip(pre_a, post_a, stage)]) if pre else np.zeros((0, 24, 3))
    return PathwayEdges(pre_a, post_a, np.array(w), np.array(route, np.int8), np.array(stage, np.int8), pts.astype(np.float32))


# --------------------------------------------------------------------------- activity
class BrainActivity:
    """Per-neuron decaying spike trace: a <- a * exp(-dt / tau) + counts. glow = 1 - exp(-a / ref(role))."""

    def __init__(self, geo: BrainGeometry, tau_s: float = 0.2):
        self.geo = geo
        self.tau = tau_s
        self.a = np.zeros(geo.n, np.float32)
        self.ref = ROLE_REF_SPIKES[geo.role]
        self.last_t: float | None = None

    def update(self, counts: np.ndarray | None, dt: float) -> None:
        self.a *= np.float32(math.exp(-dt / self.tau))
        self.dt = dt
        if counts is not None:
            c = np.asarray(counts, np.float32)
            if c.shape == self.a.shape:
                self.a += c

    def rate_hz(self, idx) -> float:
        """Smoothed firing rate (Hz) of a neuron or the mean over a group, from the decaying trace."""
        dt = getattr(self, "dt", 0.05)
        a = float(np.mean(self.a[np.atleast_1d(idx)])) if np.size(idx) else 0.0
        return a * (1.0 - math.exp(-dt / self.tau)) / dt

    def label_rates(self) -> dict[str, float]:
        g = self.geo
        out = {}
        for s in "LR":
            if f"LC10a_{s}" in g.groups:
                out[f"LC10a {s}"] = self.rate_hz(g.groups[f"LC10a_{s}"])
            for t in ("AOTU019", "DNa02", "AOTU025", "AOTU012"):
                if f"{t}_{s}" in g.key:
                    out[f"{t} {s}"] = self.rate_hz(g.key[f"{t}_{s}"])
        return out

    def update_from_rates(self, frame: dict, dt: float) -> None:
        """Fallback when a frame carries no spike counts: expected counts from input and DN rates."""
        c = np.zeros(self.geo.n, np.float32)
        for g, r in (frame.get("inputs_per_neuron") or {}).items():
            idx = self.geo.groups.get(g)
            if idx is not None and idx.size:
                c[idx] = np.broadcast_to(np.asarray(r, np.float32), idx.shape) * dt
        for g, r in (frame.get("dn") or {}).items():
            idx = self.geo.groups.get(g)
            if idx is not None and idx.size:
                c[idx] = float(r) * dt
        self.update(c, dt)

    def glow(self) -> np.ndarray:
        return 1.0 - np.exp(-self.a / self.ref)


def neuron_colors(geo: BrainGeometry, glow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB float (n, 3) 0..255 and radii (n,) um for the subgraph given glow in 0..1."""
    base = ROLE_COLORS[geo.role]
    dim = np.where(geo.role[:, None] == R_OTHER, 0.22, 0.38).astype(np.float32)
    col = base * (dim + (1 - dim) * glow[:, None]) + (255 - base) * (0.55 * glow[:, None] ** 2)
    rad = ROLE_BASE_RADIUS_UM[geo.role] * (1.0 + 0.9 * glow)
    return np.clip(col, 0, 255), rad


def edge_colors(geo: BrainGeometry, glow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = geo.edges
    base = np.where(e.route[:, None] == 1, np.asarray(draw.AOTU_INH, np.float32), np.asarray(draw.AOTU_EXC, np.float32))
    act = glow[e.pre]
    col = base * (0.18 + 0.82 * act[:, None])
    wn = np.abs(e.weight) / max(1.0, float(np.abs(e.weight).max()))
    rad = (0.6 + 1.6 * wn) * (1.0 + 1.2 * act)
    return col, rad


# --------------------------------------------------------------------------- rerun
def log_static_rerun(geo: BrainGeometry, root: str = "brain") -> None:
    import rerun as rr

    rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(f"{root}/context", rr.Points3D(geo.context, radii=1.3, colors=np.array([*draw.CONTEXT, 70], np.uint8)), static=True)
    lab_pos = np.array([p for _, p, _ in geo.labels], np.float32)
    rr.log(f"{root}/labels", rr.Points3D(lab_pos + np.array([0, 0, 18], np.float32), radii=0.1,
                                          colors=np.array([c for _, _, c in geo.labels], np.uint8),
                                          labels=[t for t, _, _ in geo.labels], show_labels=True), static=True)


def log_frame_rerun(geo: BrainGeometry, glow: np.ndarray, root: str = "brain") -> None:
    import rerun as rr

    col, rad = neuron_colors(geo, glow)
    order = np.argsort(geo.role, kind="stable")  # draw pathway cells last
    rr.log(f"{root}/neurons", rr.Points3D(geo.pos[order], colors=col[order].astype(np.uint8), radii=rad[order]))
    ecol, erad = edge_colors(geo, glow)
    rr.log(f"{root}/pathway", rr.LineStrips3D(list(geo.edges.pts), colors=ecol.astype(np.uint8), radii=erad))


# --------------------------------------------------------------------------- numpy renderer (MP4 / stills)
@dataclass
class BrainCamera:
    azimuth_deg: float = 0.0  # 0 = from behind (fly right on screen right); + orbits toward the fly's right
    elevation_deg: float = 28.0  # above the horizontal
    distance_um: float = 1250.0
    fov_deg: float = 30.0
    target: tuple[float, float, float] = (0.0, 20.0, 10.0)


class BrainRenderer:
    """Numpy splat renderer: additive points and arcs with bloom over a dark vignette. ~20 ms at 960x720."""

    def __init__(self, geo: BrainGeometry, width: int = 960, height: int = 720, camera: BrainCamera | None = None,
                 title: bool = True, legend: bool = True):
        self.geo = geo
        self.W, self.H = width, height
        self.cam = camera or BrainCamera()
        self.title = title
        self.legend = legend
        self.bg = draw.radial_vignette(height, width, (16, 20, 30), (4, 5, 8), cy=0.48)
        self._ctx_cache: tuple | None = None
        self.t = 0.0

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self.cam
        az, el = math.radians(c.azimuth_deg), math.radians(c.elevation_deg)
        # camera sits behind (-Y) and above; azimuth rotates about +Z
        d = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
        tgt = np.asarray(c.target, np.float64)
        eye = tgt + c.distance_um * d
        f = tgt - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return eye, r, u, f

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye, r, u, f = self._basis()
        Q = np.asarray(P, np.float64) - eye
        x, y, z = Q @ r, Q @ u, Q @ f
        foc = 0.5 * self.H / math.tan(math.radians(self.cam.fov_deg) / 2)
        z = np.maximum(z, 1.0)
        return (self.W / 2 + foc * x / z).astype(np.float32), (self.H / 2 - foc * y / z).astype(np.float32), z.astype(np.float32)

    def _depth_cue(self, z: np.ndarray) -> np.ndarray:
        d = self.cam.distance_um
        return np.clip(1.25 - 0.9 * (z - d) / 700.0, 0.35, 1.3).astype(np.float32)

    def _fat(self, acc, x, y, rgb, radius_px) -> None:
        """Splat points as soft discs; radius per point, grouped into size classes, one splat call."""
        radius_px = np.asarray(radius_px, np.float32)
        classes = np.clip(np.round(radius_px), 1, 16)
        XS, YS, CS = [], [], []
        for rc in np.unique(classes):
            m = classes == rc
            ox, oy, w = draw.disc_offsets(float(rc))
            XS.append((x[m][:, None] + ox[None, :]).ravel())
            YS.append((y[m][:, None] + oy[None, :]).ravel())
            CS.append((rgb[m][:, None, :] * w[None, :, None]).reshape(-1, 3))
        if XS:
            draw.splat(acc, np.concatenate(XS), np.concatenate(YS), np.concatenate(CS))

    def render(self, glow: np.ndarray, t: float | None = None, rates: dict[str, float] | None = None) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        geo = self.geo
        if t is not None:
            self.t = t
        H, W = self.H, self.W
        acc = np.zeros((H, W, 3), np.float32)
        foc = 0.5 * H / math.tan(math.radians(self.cam.fov_deg) / 2)

        # context cloud (cached per camera pose)
        key = (self.cam.azimuth_deg, self.cam.elevation_deg, self.cam.distance_um)
        if self._ctx_cache is None or self._ctx_cache[0] != key:
            ctx = np.zeros((H, W, 3), np.float32)
            x, y, z = self.project(geo.context)
            cue = self._depth_cue(z)
            draw.splat(ctx, x, y, np.asarray(draw.CONTEXT, np.float32) * 0.55 * cue[:, None])
            self._ctx_cache = (key, ctx)
        acc += self._ctx_cache[1]

        # pathway arcs with travelling pulses
        e = geo.edges
        if e.pre.size:
            ecol, _ = edge_colors(geo, glow)
            S = e.pts.shape[1]
            P = e.pts.reshape(-1, 3)
            x, y, z = self.project(P)
            cue = self._depth_cue(z)
            wn = (np.abs(e.weight) / max(1.0, float(np.abs(e.weight).max())))
            base = (ecol * (0.35 + 0.65 * wn[:, None]))
            cols = np.repeat(base, S, axis=0) * cue[:, None] * 0.55
            # densify: splat between samples too
            xm = 0.5 * (x.reshape(-1, S)[:, :-1] + x.reshape(-1, S)[:, 1:]).ravel()
            ym = 0.5 * (y.reshape(-1, S)[:, :-1] + y.reshape(-1, S)[:, 1:]).ravel()
            cm = np.repeat(base, S - 1, axis=0) * 0.55
            draw.splat(acc, x, y, cols)
            draw.splat(acc, xm, ym, cm)
            act = glow[e.pre]
            speed = np.where(e.stage == 0, 0.9, 1.3)
            ph = (self.t * speed + (np.arange(e.pre.size) * 0.137) % 1.0) % 1.0
            k = np.clip((ph * (S - 1)).astype(int), 0, S - 1)
            pp = e.pts[np.arange(e.pre.size), k]
            px, py, _ = self.project(pp)
            pcol = (np.where(e.route[:, None] == 1, np.asarray(draw.AOTU_INH, np.float32), np.asarray(draw.AOTU_EXC, np.float32)) * 0.6
                    + 100.0) * (act[:, None] ** 1.5) * 1.4
            self._fat(acc, px, py, pcol, np.full(px.size, 2.0))

        # subgraph neurons
        col, rad = neuron_colors(geo, glow)
        x, y, z = self.project(geo.pos)
        cue = self._depth_cue(z)
        rpx = rad * foc / z * 0.55
        order = np.argsort(geo.role, kind="stable")
        weight = np.where(geo.role == R_OTHER, 0.5, 1.0).astype(np.float32)
        self._fat(acc, x[order], y[order], col[order] * (cue * weight)[order][:, None] * 0.55, np.maximum(rpx[order], 0.7))

        # DNa02 halos: a ring that swells with the steering neuron's activity (the output of the circuit)
        ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        for s in "LR":
            i = geo.key.get(f"DNa02_{s}")
            if i is None:
                continue
            g = float(glow[i])
            rr_px = 9.0 + 12.0 * g
            draw.splat(acc, x[i] + rr_px * np.cos(ang), y[i] + rr_px * np.sin(ang),
                       np.asarray(draw.DNA02, np.float32) * (0.12 + 0.9 * g))

        # bloom: blur at half resolution, cheap nearest upsample (the blur hides the blockiness)
        small = acc[::2, ::2]
        blur = gaussian_filter(small, sigma=(5, 5, 0))
        bloom = np.repeat(np.repeat(blur, 2, axis=0), 2, axis=1)[:H, :W]
        acc = acc + 1.6 * bloom
        img = np.clip(self.bg.astype(np.float32) + draw.tone_map(acc, exposure=1.5).astype(np.float32), 0, 255).astype(np.uint8)
        with draw.TextBatch(img) as tb:
            self._labels(tb, glow, rates)
            if self.title:
                tb.text((18, 14), "FLY BRAIN", size=17, bold=True)
                tb.text((18, 38), f"MaleCNS connectome: {geo.n:,}-neuron pursuit circuit at real soma positions", size=12, color=draw.MUTED)
            if self.legend:
                self._legend(tb)
        return img

    def _labels(self, tb: draw.TextBatch, glow: np.ndarray, rates: dict[str, float] | None) -> None:
        """Labels in two outer columns (fly left / fly right), stacked by height, with leader lines and live rates."""
        geo = self.geo
        rates = rates or {}
        H, W = self.H, self.W
        per_side: dict[str, list] = {"L": [], "R": []}
        for text, p, col in geo.labels:
            sx, sy, _ = self.project(p[None, :])
            per_side[text[-1]].append([float(sy[0]), float(sx[0]), text, col])
        for side, items in per_side.items():
            items.sort()
            y_prev = -1e9
            col_x = 0.17 * W if side == "L" else 0.83 * W
            placed = []
            for sy, sx, text, col in items:
                ly = max(sy - 40, y_prev + 44, 80)
                y_prev = ly
                placed.append((ly, sy, sx, text, col))
            overflow = placed[-1][0] - (H - 150) if placed else 0
            for ly, sy, sx, text, col in placed:
                ly -= max(0.0, overflow)
                hz = rates.get(text)
                active = min(1.0, (hz or 0.0) / (60.0 if text.startswith("LC10a") else 120.0))
                a = 0.5 + 0.5 * active
                colv = tuple(int(c * a + 30 * (1 - a)) for c in col)
                ex = col_x + (8 if side == "L" else -8)
                tb.line([(sx, sy), (ex, ly - 5)], colv, alpha=0.35 + 0.4 * active)
                anchor = "rs" if side == "L" else "ls"
                tb.text((col_x, ly), text, size=15, color=colv, anchor=anchor, bold=True, shadow=True)
                if hz is not None:
                    tb.text((col_x, ly + 16), f"{hz:5.0f} Hz", size=12, color=colv if active > 0.2 else draw.MUTED,
                            anchor=anchor, kind="mono", shadow=True)

    def _legend(self, tb: draw.TextBatch) -> None:
        x0, y0 = 18, self.H - 108
        items = [(draw.LC10A, "LC10a visual target neurons (input)"), (draw.AOTU_EXC, "AOTU025/012: excite same-side DNa02"),
                 (draw.AOTU_INH, "AOTU019 (GABA): inhibits opposite DNa02"), (draw.DNA02, "DNa02 steering descending neurons (output)")]
        for i, (c, s) in enumerate(items):
            yy = y0 + i * 20
            tb.rect((x0, yy - 4, x0 + 8, yy + 4), fill=c)
            tb.text((x0 + 16, yy), s, size=12, color=draw.TEXT, anchor="lm")


def _line(img: np.ndarray, x0, y0, x1, y1, col, alpha: float = 0.6) -> None:
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)
    ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[ok], ys[ok]
    img[ys, xs] = (img[ys, xs] * (1 - alpha) + np.asarray(col) * alpha).astype(np.uint8)


# --------------------------------------------------------------------------- CLI: stills
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render brain-view stills driven by a scripted spot (LIF on the subgraph)")
    ap.add_argument("--brain", default="data/brains/pursuit_core1.npz")
    ap.add_argument("--out", default="runs/viz_check")
    a = ap.parse_args(argv)
    import imageio.v2 as imageio

    from flyfollow.viz.frames import synthetic_frames

    t0 = time.perf_counter()
    geo = BrainGeometry.load(a.brain)
    print(f"geometry: {geo.n} neurons ({(~geo.has_soma).sum()} without soma), {geo.context.shape[0]} context somata, "
          f"{geo.edges.pre.size} pathway edges, {time.perf_counter() - t0:.2f} s")
    act = BrainActivity(geo)
    ren = BrainRenderer(geo)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    shots = {30: "brain_spot_right", 70: "brain_spot_left"}
    ms = []
    for i, fr in enumerate(synthetic_frames(a.brain, seconds=4.0, seed=0, script="left_right")):
        act.update(fr["counts"], 0.05)
        if i in shots:
            t1 = time.perf_counter()
            img = ren.render(act.glow(), t=fr["t"], rates=act.label_rates())
            ms.append(1000 * (time.perf_counter() - t1))
            imageio.imwrite(Path(a.out) / f"{shots[i]}.png", img)
            print(shots[i], "bearing", round(fr["target"]["bearing_deg"], 1), "DNa02 L/R", round(fr["dn"]["DNa02_L"]), round(fr["dn"]["DNa02_R"]))
    for _ in range(3):
        t1 = time.perf_counter()
        ren.render(act.glow(), t=1.0)
        ms.append(1000 * (time.perf_counter() - t1))
    print(f"render {np.mean(ms[1:]):.1f} ms per frame")


if __name__ == "__main__":
    main()
