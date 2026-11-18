"""
kolam.rules -- identify the design rule used at every crossing point.

PURPOSE
-------
Given the dot lattice and the curve skeleton, decide for each lattice
"gate" (the midpoint between two neighbouring dots) whether the strands
  X  cross straight through,
  H  bounce back along the pairing (NW-NE / SW-SE), or
  V  bounce back along the pairing (NW-SW / NE-SE).
This turns a picture into a KolamModel (see kolam.model) whose loops can be
traced, compared with the picture, and re-used by the generator.

METHOD (connectivity test, robust to line thickness and curve shape)
-------------------------------------------------------------------
Around a gate we look at a small disc (radius ~0.8 node unit).  Inside it the
skeleton is split into connected pieces.  The four strands leave the disc in
the diagonal directions NW, NE, SE, SW; we find which piece each strand
belongs to:
    all four in one piece, or NW~SE and NE~SW  -> X
    NW~NE and SW~SE (not NW~SW)                -> H
    NW~SW and NE~SE (not NW~NE)                -> V
anything else is reported as 'unknown' (and defaults to X, flagged).

LIMITATIONS: needs a clean dot lattice (module 3), curves that really follow
the dot-gate geometry, and enough resolution (>= ~12 px per node unit).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .dots import GridInfo
from .model import KolamModel, DIR_NW, DIR_NE, DIR_SE, DIR_SW


@dataclass
class NodeStates:
    model: KolamModel
    confidence: dict          # node -> 1.0 (clear) | 0.0 (unknown, defaulted to X)
    node_px: dict             # node -> (x, y) pixel position of the gate
    unknown_fraction: float
    counts: dict


def node_pixel(grid: GridInfo, node) -> np.ndarray:
    """Image position of node coordinates (x, y); dot (c, r) sits at node (2c+1, 2r+1)."""
    return grid.origin + (node[0] - 1) * grid.a1 / 2 + (node[1] - 1) * grid.a2 / 2


def _classify(skel: np.ndarray, centre: np.ndarray, ex: np.ndarray, ey: np.ndarray):
    u = min(np.linalg.norm(ex), np.linalg.norm(ey))
    R = int(np.ceil(0.8 * u)) + 1
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    H, W = skel.shape
    if cx - R < 0 or cy - R < 0 or cx + R >= W or cy + R >= H:
        return "X", 0.0
    win = skel[cy - R:cy + R + 1, cx - R:cx + R + 1] > 0
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    win &= (xx ** 2 + yy ** 2) <= (0.8 * u) ** 2
    n, lab = cv2.connectedComponents(win.astype(np.uint8), connectivity=8)
    if n <= 1:
        return "X", 0.0
    pts = np.argwhere(lab > 0)
    labels = {}
    for name, (dx, dy) in (("NW", DIR_NW), ("NE", DIR_NE), ("SE", DIR_SE), ("SW", DIR_SW)):
        nominal = np.array([R, R], float) + 0.45 * (dx * ex[::-1] + dy * ey[::-1])  # (row, col) order
        d = np.hypot(pts[:, 0] - nominal[0], pts[:, 1] - nominal[1])
        j = int(np.argmin(d))
        labels[name] = int(lab[tuple(pts[j])]) if d[j] <= 0.3 * u else None

    def same(a, b):
        return labels[a] is not None and labels[a] == labels[b]
    present = [v for v in labels.values() if v is not None]
    if len(present) >= 3 and len(set(present)) == 1:
        return "X", 1.0
    nwse, nesw = same("NW", "SE"), same("NE", "SW")
    h_ = same("NW", "NE") or same("SW", "SE")
    v_ = same("NW", "SW") or same("NE", "SE")
    if (nwse or nesw) and not h_ and not v_:
        return "X", 1.0
    if h_ and not v_ and not nwse and not nesw:
        return "H", 1.0
    if v_ and not h_ and not nwse and not nesw:
        return "V", 1.0
    return "X", 0.0


def identify_node_states(curve_skeleton: np.ndarray, grid: GridInfo) -> NodeStates:
    model = KolamModel(grid.occupancy)
    ex, ey = grid.a1 / 2.0, grid.a2 / 2.0
    conf, px = {}, {}
    for n in model.free_nodes:
        c = node_pixel(grid, n)
        s, cf = _classify(curve_skeleton, c, ex, ey)
        model.states[n] = s
        conf[n] = cf
        px[n] = (float(c[0]), float(c[1]))
    for n in model.seg_dirs:
        if n not in px:
            c = node_pixel(grid, n)
            px[n] = (float(c[0]), float(c[1]))
    unk = (sum(1 for v in conf.values() if v == 0.0) / len(conf)) if conf else 0.0
    return NodeStates(model, conf, px, unk, model.state_counts())
