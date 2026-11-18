"""
kolam.dots -- Module 3: find the dots (pulli) and reconstruct the dot lattice.

DOT DETECTION
-------------
A dot is a small, filled, round, isolated blob.  For every connected component
we test: area window, bounding-box aspect ratio, fill ratio (area / bbox area,
pi/4 ~ 0.785 for a disc), circularity 4*pi*A/P^2 (=1 for a circle), and "no
holes" (a ring-shaped curve is not a dot).  Survivors are filtered again by
size consistency (all dots in one kolam have about the same area).  If dots
touch the curve we fall back to a morphological opening whose disc is wider than
the stroke, which erases strokes but keeps the thicker dots.

LATTICE FITTING (the maths)
---------------------------
Dot positions are modelled as   p = o + u*a1 + v*a2   with integer (u, v).
1. Collect vectors to each dot's nearest neighbours; fold v ~ -v and histogram
   their directions; the two strongest, non-parallel peaks give the lattice
   directions; the median vectors in those clusters give a1, a2.
2. Round (p - o) expressed in the basis to integers (u, v).
3. Refit o, a1, a2 by linear least squares  [1 u v] -> (x, y); repeat.
The angle between a1 and a2 and the ratio |a1|/|a2| classify the lattice
(square, rectangular, triangular/hexagonal, oblique).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .preprocess import estimate_stroke_width, skeletonize_mask


@dataclass
class DotDetection:
    centers: np.ndarray            # (N, 2) float (x, y)
    radii: np.ndarray              # (N,)
    method: str
    curve_mask: np.ndarray         # binary with dots erased (curves only)
    n_candidates: int = 0


@dataclass
class GridInfo:
    ok: bool
    message: str = ""
    lattice_type: str = "unknown"
    a1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    a2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))   # pixel position of dot (col 0, row 0)
    spacing: float = 0.0
    orientation_deg: float = 0.0     # image angle of a1
    angle_between_deg: float = 0.0
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # (col,row) per dot
    rows: int = 0
    cols: int = 0
    occupancy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    row_profile: list = field(default_factory=list)
    rms_error: float = 0.0
    rel_error: float = 0.0
    collisions: int = 0
    n_dots: int = 0

    def to_dict(self) -> dict:
        return dict(ok=self.ok, message=self.message, lattice_type=self.lattice_type,
                    spacing_px=round(float(self.spacing), 2), orientation_deg=round(float(self.orientation_deg), 2),
                    angle_between_deg=round(float(self.angle_between_deg), 2), rows=self.rows, cols=self.cols,
                    n_dots=self.n_dots, row_profile=self.row_profile,
                    rms_error_px=round(float(self.rms_error), 3), rel_error=round(float(self.rel_error), 4),
                    collisions=self.collisions)


# ------------------------------------------------------------------ detection
def _compact_blobs(binary: np.ndarray, stroke: float, min_area: float, max_area: float):
    n, lab, stats, cents = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_area <= a <= max_area):
            continue
        asp = w / h
        if not (0.6 <= asp <= 1.67) or a / (w * h) < 0.55:
            continue
        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, hier = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(cnts) != 1:                       # has a hole (ring) or fragments
            continue
        per = cv2.arcLength(cnts[0], True)
        circ = 4 * np.pi * a / (per ** 2 + 1e-9)
        if circ < 0.55:
            continue
        out.append((i, a, cents[i], np.sqrt(a / np.pi)))
    return out, lab


def _size_filter(blobs):
    if len(blobs) < 3:
        return blobs
    med = np.median([b[1] for b in blobs])
    return [b for b in blobs if 0.35 * med <= b[1] <= 2.8 * med]


def detect_dots(binary: np.ndarray, stroke_width: float | None = None) -> DotDetection:
    binary = (binary > 0).astype(np.uint8) * 255
    if stroke_width is None:
        stroke_width = estimate_stroke_width(binary, skeletonize_mask(binary))
    area_img = binary.size
    min_a = max(8.0, 0.5 * stroke_width ** 2)
    max_a = 0.02 * area_img
    blobs, lab = _compact_blobs(binary, stroke_width, min_a, max_a)
    blobs = _size_filter(blobs)
    method = "connected-components"
    curve = binary.copy()
    if len(blobs) >= 4:
        for b in blobs:
            curve[lab == b[0]] = 0
        centers = np.array([b[2] for b in blobs], float)
        radii = np.array([b[3] for b in blobs], float)
        return DotDetection(centers, radii, method, curve, len(blobs))

    # ---- fallback: opening with a disc wider than the stroke removes strokes, keeps dots
    r = max(2, int(round(0.85 * stroke_width)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    blobs2, lab2 = _compact_blobs(opened, stroke_width, min_a, max_a)
    blobs2 = _size_filter(blobs2)
    if len(blobs2) > len(blobs):
        centers = np.array([b[2] for b in blobs2], float)
        radii = np.array([b[3] for b in blobs2], float)
        erase = cv2.dilate((lab2 > 0).astype(np.uint8) * 255, k)
        curve = cv2.bitwise_and(binary, cv2.bitwise_not(erase))
        return DotDetection(centers, radii, "opening-fallback", curve, len(blobs2))
    centers = np.array([b[2] for b in blobs], float).reshape(-1, 2)
    radii = np.array([b[3] for b in blobs], float)
    return DotDetection(centers, radii, method, binary, len(blobs))


# ------------------------------------------------------------ lattice fitting
def _circ_smooth(h: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([h[-r:], h, h[:r]])
    return np.convolve(ext, k, mode="same")[r:-r]


def _aligned_median(vecs: np.ndarray, ang_deg: float, tol: float = 14.0) -> np.ndarray:
    a = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) % 180
    d = np.abs((a - ang_deg + 90) % 180 - 90)
    sel = vecs[d < tol]
    u = np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])
    sel = np.where((sel @ u)[:, None] < 0, -sel, sel)
    return np.median(sel, axis=0)


def _row_profile(centers: np.ndarray, tilt_deg: float, d_nn: float) -> list:
    th = np.radians(-tilt_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    y = np.sort((centers @ R.T)[:, 1])
    rows, cur = [], [y[0]]
    for v in y[1:]:
        if v - cur[-1] > 0.3 * d_nn:
            rows.append(len(cur)); cur = [v]
        else:
            cur.append(v)
    rows.append(len(cur))
    return rows


def fit_lattice(centers: np.ndarray) -> GridInfo:
    P = np.asarray(centers, float)
    N = len(P)
    if N < 4:
        return GridInfo(False, f"only {N} dots found; need at least 4 to fit a lattice")
    k = min(N, 7)
    dists, idx = cKDTree(P).query(P, k=k)
    d_nn = float(np.median(dists[:, 1]))
    vecs = [P[j] - P[i] for i in range(N) for j, dj in zip(idx[i, 1:], dists[i, 1:]) if dj <= 1.3 * d_nn]
    vecs = np.array(vecs)
    ang = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) % 180
    h, _ = np.histogram(ang, bins=180, range=(0, 180))
    hs = _circ_smooth(h.astype(float), 1.5)
    t1 = int(np.argmax(hs))
    dist_to_t1 = np.abs((np.arange(180) - t1 + 90) % 180 - 90)
    hs2 = np.where(dist_to_t1 < 25, -1, hs)
    t2 = int(np.argmax(hs2))
    if hs2[t2] < 0.2 * hs[t1]:
        return GridInfo(False, "dots look collinear (only one lattice direction found)")
    a1, a2 = _aligned_median(vecs, t1 + 0.5), _aligned_median(vecs, t2 + 0.5)

    # integer coordinates + least-squares refit (3 rounds)
    o = P[np.argmin(((P - P.mean(0)) ** 2).sum(1))]
    for _ in range(4):
        A = np.stack([a1, a2], 1)
        uv = np.round(np.linalg.solve(A, (P - o).T).T)
        D = np.column_stack([np.ones(N), uv])
        if np.linalg.matrix_rank(D) < 3:
            return GridInfo(False, "degenerate lattice fit")
        coef, *_ = np.linalg.lstsq(D, P, rcond=None)
        o, a1, a2 = coef[0], coef[1], coef[2]
    A = np.stack([a1, a2], 1)
    uvf = np.linalg.solve(A, (P - o).T).T
    uv = np.round(uvf).astype(int)
    resid = np.linalg.norm(P - (o + uv @ np.stack([a1, a2])), axis=1)

    # canonical basis: a1 = the more horizontal one, pointing right; a2 pointing down
    def ang_h(v):
        return abs(((np.degrees(np.arctan2(v[1], v[0])) + 90) % 180) - 90)
    if ang_h(a2) < ang_h(a1) - 1e-6:
        a1, a2 = a2, a1
        uv = uv[:, ::-1]
    if a1[0] < 0 or (abs(a1[0]) < 1e-6 and a1[1] < 0):
        a1, uv = -a1, uv * np.array([-1, 1])
    if a2[1] < 0 or (abs(a2[1]) < 1e-6 and a2[0] < 0):
        a2, uv = -a2, uv * np.array([1, -1])
    umin, vmin = uv.min(0)
    uv = uv - np.array([umin, vmin])
    origin = o + umin * a1 + vmin * a2
    cols, rows = int(uv[:, 0].max() + 1), int(uv[:, 1].max() + 1)
    occ = np.zeros((rows, cols), bool)
    occ[uv[:, 1], uv[:, 0]] = True
    collisions = int(N - occ.sum())

    L1, L2 = np.linalg.norm(a1), np.linalg.norm(a2)
    cosang = np.clip(abs(a1 @ a2) / (L1 * L2), 0, 1)
    between = float(np.degrees(np.arccos(a1 @ a2 / (L1 * L2))))
    acute = float(np.degrees(np.arccos(cosang)))
    ratio = max(L1, L2) / min(L1, L2)
    if abs(acute - 90) < 12:
        ltype = "square" if ratio < 1.12 else "rectangular"
    elif abs(acute - 60) < 10 and ratio < 1.15:
        ltype = "triangular/hexagonal"
    else:
        ltype = "oblique"
    orient = float(np.degrees(np.arctan2(a1[1], a1[0])))
    tilt = ((orient + 22.5) % 45) - 22.5
    return GridInfo(True, "ok", ltype, a1, a2, origin, float((L1 + L2) / 2), orient, between, uv, rows, cols,
                    occ, _row_profile(P, tilt, d_nn), float(np.sqrt((resid ** 2).mean())),
                    float(np.sqrt((resid ** 2).mean()) / ((L1 + L2) / 2)), collisions, N)
