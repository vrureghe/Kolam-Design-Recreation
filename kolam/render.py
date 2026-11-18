"""kolam.render -- draw a KolamModel as an anti-aliased image (BGR uint8)."""
from __future__ import annotations

import cv2
import numpy as np

from .curves import bspline_closed
from .model import KolamModel

# BGR colours for 'colour each loop separately'
PALETTE = [(60, 180, 255), (120, 220, 120), (255, 160, 90), (200, 120, 255),
           (90, 230, 230), (130, 130, 255), (230, 200, 90), (150, 255, 190)]

THEMES = {
    "rice flour on dark floor": dict(fg=(235, 235, 235), bg=(34, 28, 24)),
    "ink on paper": dict(fg=(30, 30, 30), bg=(245, 245, 245)),
    "red on cream": dict(fg=(40, 40, 190), bg=(225, 240, 250)),
    "turmeric on dark": dict(fg=(40, 190, 250), bg=(40, 30, 30)),
}


def render_model(model: KolamModel, u: int = 28, fg=(235, 235, 235), bg=(34, 28, 24),
                 line_frac: float = 0.17, dot_frac: float = 0.24, margin: float = 1.4,
                 color_by_loop: bool = False, show_dots: bool = True,
                 ss: int = 3, samples: int = 12) -> np.ndarray:
    """u = pixels per node-unit (dot spacing = 2u).  Returns BGR image."""
    W = int(round((2 * model.C + 2 * margin) * u))
    H = int(round((2 * model.R + 2 * margin) * u))
    img = np.full((H * ss, W * ss, 3), bg, np.uint8)
    thick = max(1, int(round(line_frac * u * ss)))
    SH = 4
    k = 1 << SH

    def to_px(pts):
        return np.round((np.asarray(pts, float) + margin) * u * ss * k).astype(np.int64)

    for i, loop in enumerate(model.loops()):
        col = PALETTE[i % len(PALETTE)] if color_by_loop else fg
        pts = bspline_closed(np.array(loop, float), samples)
        cv2.polylines(img, [to_px(pts).astype(np.int32)], True, col, thick, cv2.LINE_AA, SH)
    if show_dots:
        r = max(1, int(round(dot_frac * u * ss * k)))
        for (x, y) in model.dots:
            c = to_px([(x, y)])[0]
            cv2.circle(img, (int(c[0]), int(c[1])), r, fg, -1, cv2.LINE_AA, SH)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
