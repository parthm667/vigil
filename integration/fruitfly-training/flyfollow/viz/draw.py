"""Small drawing helpers shared by the visualization panels (fonts, text, colors, splatting).

Everything draws into numpy RGB uint8 images, so the same panels feed Rerun, the MP4 recorder and
PNG stills. PIL (a matplotlib dependency) does the text; numpy does the additive point splats.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Palette (RGB 0..255). Dark theme; warm = excitation, violet = inhibition.
BG = (7, 9, 13)
PANEL = (12, 15, 21)
GRID = (34, 40, 52)
TEXT = (226, 232, 240)
MUTED = (132, 145, 162)
LC10A = (56, 189, 248)  # sky blue: LC10a visual inputs
AOTU_EXC = (251, 176, 64)  # amber: excitatory AOTU relay (AOTU025, AOTU012)
AOTU_INH = (167, 139, 250)  # violet: inhibitory AOTU019 (GABA)
DNA02 = (255, 77, 109)  # hot pink: DNa02 steering descending neurons
DN_OTHER = (244, 114, 182)
LC_AUX = (45, 212, 191)  # teal: LC9 / LC11
AROUSAL = (250, 204, 21)  # yellow: P1 (pC1) arousal
RELAY = (148, 163, 184)  # other subgraph neurons
CONTEXT = (70, 86, 110)  # all-MaleCNS somata cloud
TARGET = (74, 222, 128)  # green: the person / target box
DRONE = (226, 232, 240)
LEFT = (56, 189, 248)
RIGHT = (255, 77, 109)

_FONT_CANDIDATES = {
    "sans": [
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
}


@lru_cache(maxsize=64)
def font(size: int, kind: str = "sans", bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A TrueType font of `size` px; system fonts first, then matplotlib's DejaVu, then PIL's bitmap font."""
    paths = list(_FONT_CANDIDATES.get(kind, _FONT_CANDIDATES["sans"]))
    try:
        import matplotlib.font_manager as fm

        paths.append(fm.findfont("DejaVu Sans Mono" if kind == "mono" else "DejaVu Sans"))
    except Exception:  # noqa: BLE001, S110 (matplotlib is optional here)
        pass
    want = ("demi bold", "bold", "semibold") if bold else ("medium", "regular", "book", "roman")
    for p in paths:
        if not Path(p).exists():
            continue
        try:
            if p.endswith(".ttc"):  # collections: pick the face by its style name, never an italic
                faces = {}
                for idx in range(24):
                    try:
                        f = ImageFont.truetype(p, size, index=idx)
                    except OSError:
                        break
                    faces.setdefault(f.getname()[1].lower(), f)
                for style in want:
                    if style in faces:
                        return faces[style]
                upright = [f for s, f in faces.items() if "italic" not in s and "oblique" not in s]
                if upright:
                    return upright[0]
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text(img: np.ndarray, xy: tuple[float, float], s: str, size: int = 14, color=TEXT, kind: str = "sans",
         bold: bool = False, anchor: str = "la", shadow: bool = False, alpha: float = 1.0) -> np.ndarray:
    """Draw text onto an RGB uint8 image in place (and return it). anchor follows PIL ('la', 'mm', 'rs', ...)."""
    pil = Image.fromarray(img)
    if alpha < 1.0 or shadow:
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f = font(size, kind, bold)
        if shadow:
            d.text((xy[0] + 1, xy[1] + 1), s, font=f, fill=(0, 0, 0, int(200 * alpha)), anchor=anchor)
        d.text(xy, s, font=f, fill=(*color, int(255 * alpha)), anchor=anchor)
        pil = Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB")
    else:
        ImageDraw.Draw(pil).text(xy, s, font=font(size, kind, bold), fill=tuple(color), anchor=anchor)
    img[...] = np.asarray(pil)
    return img


class TextBatch:
    """Many text / line draws with one numpy <-> PIL round trip (text() converts the whole image per call).

        with TextBatch(img) as tb:
            tb.text((10, 10), "hello", size=14, color=TEXT, shadow=True)
    """

    def __init__(self, img: np.ndarray):
        self.img = img
        self.layer = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)

    def text(self, xy, s: str, size: int = 14, color=TEXT, kind: str = "sans", bold: bool = False, anchor: str = "la",
             shadow: bool = False, alpha: float = 1.0) -> None:
        f = font(size, kind, bold)
        if shadow:
            self.d.text((xy[0] + 1, xy[1] + 1), s, font=f, fill=(0, 0, 0, int(210 * alpha)), anchor=anchor)
        self.d.text(xy, s, font=f, fill=(*[int(c) for c in color], int(255 * alpha)), anchor=anchor)

    def line(self, pts, color, width: int = 1, alpha: float = 1.0) -> None:
        self.d.line([tuple(map(float, p)) for p in pts], fill=(*[int(c) for c in color], int(255 * alpha)), width=width)

    def rect(self, box, fill=None, outline=None, width: int = 1, radius: int = 0, alpha: float = 1.0) -> None:
        f = (*[int(c) for c in fill], int(255 * alpha)) if fill is not None else None
        o = (*[int(c) for c in outline], int(255 * alpha)) if outline is not None else None
        if radius:
            self.d.rounded_rectangle(box, radius=radius, fill=f, outline=o, width=width)
        else:
            self.d.rectangle(box, fill=f, outline=o, width=width)

    def ellipse(self, box, fill=None, outline=None, width: int = 1, alpha: float = 1.0) -> None:
        f = (*[int(c) for c in fill], int(255 * alpha)) if fill is not None else None
        o = (*[int(c) for c in outline], int(255 * alpha)) if outline is not None else None
        self.d.ellipse(box, fill=f, outline=o, width=width)

    def polygon(self, pts, fill=None, outline=None, alpha: float = 1.0) -> None:
        f = (*[int(c) for c in fill], int(255 * alpha)) if fill is not None else None
        o = (*[int(c) for c in outline], int(255 * alpha)) if outline is not None else None
        self.d.polygon([tuple(map(float, p)) for p in pts], fill=f, outline=o)

    def commit(self) -> np.ndarray:
        a = np.asarray(self.layer, np.float32)
        al = a[..., 3:4] / 255.0
        self.img[...] = (self.img.astype(np.float32) * (1 - al) + a[..., :3] * al + 0.5).astype(np.uint8)
        return self.img

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.commit()


def text_width(s: str, size: int = 14, kind: str = "sans", bold: bool = False) -> int:
    f = font(size, kind, bold)
    l, _, r, _ = f.getbbox(s)
    return int(r - l)


def vgradient(h: int, w: int, top, bottom) -> np.ndarray:
    """Vertical color gradient, uint8 (h, w, 3)."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    g = (1 - t) * np.asarray(top, np.float32) + t * np.asarray(bottom, np.float32)
    return np.broadcast_to(g, (h, w, 3)).astype(np.uint8)


def radial_vignette(h: int, w: int, center_rgb, edge_rgb, cx: float = 0.5, cy: float = 0.45, power: float = 1.6) -> np.ndarray:
    """Radial gradient background, uint8 (h, w, 3)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - cx) * (w / max(h, w))) ** 2 + ((yy / h - cy) * (h / max(h, w))) ** 2) / 0.75
    t = np.clip(r, 0, 1)[..., None] ** power
    g = (1 - t) * np.asarray(center_rgb, np.float32) + t * np.asarray(edge_rgb, np.float32)
    return g.astype(np.uint8)


def splat(acc: np.ndarray, x: np.ndarray, y: np.ndarray, rgb: np.ndarray, weight: np.ndarray | float = 1.0) -> None:
    """Additively splat points into a float (H, W, 3) accumulator with bilinear weights (anti-aliased)."""
    H, W, _ = acc.shape
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (x.size, 3))
    wgt = np.broadcast_to(np.asarray(weight, np.float32), x.shape)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    xi = np.concatenate([x0, x0 + 1, x0, x0 + 1])
    yi = np.concatenate([y0, y0, y0 + 1, y0 + 1])
    w = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * np.tile(wgt, 4)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (w > 0)
    if not ok.any():
        return
    idx = yi[ok] * W + xi[ok]
    ww = w[ok][:, None] * np.tile(rgb, (4, 1))[ok]
    # bincount over the touched bounding range only (much cheaper than the full image for local splats)
    lo, hi = int(idx.min()), int(idx.max()) + 1
    flat = acc.reshape(-1, 3)
    for c in range(3):
        flat[lo:hi, c] += np.bincount(idx - lo, weights=ww[:, c], minlength=hi - lo)


def disc_offsets(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel offsets and soft weights of a disc with a Gaussian falloff, for splatting fat points."""
    r = max(radius, 0.5)
    k = int(np.ceil(2 * r))
    oy, ox = np.mgrid[-k : k + 1, -k : k + 1].astype(np.float32)
    w = np.exp(-(ox**2 + oy**2) / (2 * (r * 0.6) ** 2))
    keep = w > 0.02
    return ox[keep], oy[keep], w[keep]


def tone_map(acc: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Soft-saturating map of an additive float accumulator (0..inf) to uint8, so hot spots bloom to white."""
    x = 1.0 - np.exp(-np.maximum(acc, 0.0) * (exposure / 255.0))
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def blend_over(dst: np.ndarray, src: np.ndarray, alpha: float) -> None:
    dst[...] = (dst.astype(np.float32) * (1 - alpha) + src.astype(np.float32) * alpha).astype(np.uint8)


def lerp_color(a, b, t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.float32)[..., None]
    return (1 - t) * np.asarray(a, np.float32) + t * np.asarray(b, np.float32)


def rounded_panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill=PANEL, outline=GRID, radius: int = 10) -> None:
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=tuple(fill), outline=tuple(outline), width=1)
    img[...] = np.asarray(pil)
