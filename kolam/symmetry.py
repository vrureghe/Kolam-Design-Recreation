"""
kolam.symmetry -- Module 4 (part): mirror, rotational and translational structure.

HOW SYMMETRY IS MEASURED
------------------------
Let A be the binary drawing, centred on a square canvas.  For a candidate
symmetry T (a reflection about a line through the centre, or a rotation by
360/n degrees) form B = T(A).  The *tolerant overlap score* is

        S(T) = ( |A ∩ dil(B)| + |B ∩ dil(A)| ) / ( |A| + |B| )

where dil() thickens the strokes by a few pixels so that hand-drawn wobble
does not count as asymmetry.  S = 1 means perfectly symmetric.

Reflection about a line at angle phi through the centre c:
        x' = R(phi) (x - c) + c ,   R = [[cos 2phi,  sin 2phi],[sin 2phi, -cos 2phi]]
We scan phi = 0..179 degrees to locate every mirror axis, and test rotations
by 360/n for n in {2,3,4,5,6,8}.

A "chance baseline" (median score of unrelated angles) is reported so a high
score can be told apart from a dense pattern that overlaps itself by accident.

TRANSLATIONAL REPETITION is read from the 2-D autocorrelation (via FFT):
peaks away from the origin are repeat vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SymmetryReport:
    center: tuple
    mirror_angles: np.ndarray
    mirror_scores: np.ndarray
    mirror_axes: list                    # [{'angle_deg','score'}]
    rotation_scores: dict                # {n: score}
    rotation_order: int
    baseline_mirror: float
    baseline_rotation: float
    group: str                           # inferred: C1, D1, C2, D2, C4, D4 ...
    threshold: float
    canvas_scale: float = 1.0
    group_strict: str = "C1"
    strict_threshold: float = 0.985

    def to_dict(self) -> dict:
        return dict(center=[round(float(c), 1) for c in self.center],
                    mirror_axes=[{k: round(float(v), 3) for k, v in a.items()} for a in self.mirror_axes],
                    rotation_scores={int(k): round(float(v), 3) for k, v in self.rotation_scores.items()},
                    rotation_order=self.rotation_order,
                    baseline_mirror=round(float(self.baseline_mirror), 3),
                    baseline_rotation=round(float(self.baseline_rotation), 3),
                    inferred_group_approx=self.group, inferred_group_strict=self.group_strict,
                    threshold=self.threshold, strict_threshold=self.strict_threshold)


def _make_canvas(binary: np.ndarray, center, size: int = 384):
    """Resample the drawing onto a square canvas whose centre is EXACTLY `center`
    (sub-pixel accurate, anti-aliased) so that rotations/reflections about the
    canvas centre are not spoiled by rounding."""
    ys, xs = np.nonzero(binary)
    cx, cy = float(center[0]), float(center[1])
    R = float(np.max(np.hypot(xs - cx, ys - cy))) + 3.0
    s = (size / 2 - 3) / R
    f = cv2.GaussianBlur((binary > 0).astype(np.float32), (0, 0), max(0.5, 0.5 / s))
    M = np.array([[s, 0, (size - 1) / 2 - s * cx], [0, s, (size - 1) / 2 - s * cy]], np.float32)
    canvas = cv2.warpAffine(f, M, (size, size), flags=cv2.INTER_LINEAR)
    return (canvas > 0.35).astype(np.uint8), s


def _tol_score(A, dA, B, k):
    dB = cv2.dilate(B, k)
    den = A.sum() + B.sum()
    return float((np.count_nonzero(A & dB) + np.count_nonzero(B & dA)) / den) if den else 0.0


def _reflect(A, phi_deg, c):
    p = np.radians(phi_deg)
    R = np.array([[np.cos(2 * p), np.sin(2 * p)], [np.sin(2 * p), -np.cos(2 * p)]])
    M = np.hstack([R, (np.array(c) - R @ np.array(c))[:, None]])
    return cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_NEAREST)


def _rotate(A, deg, c):
    M = cv2.getRotationMatrix2D(tuple(c), deg, 1.0)
    return cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_NEAREST)


def _candidate_centers(binary, dot_centers):
    ys, xs = np.nonzero(binary)
    h, w = binary.shape
    cands = [(xs.mean(), ys.mean()), ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)]
    if dot_centers is not None and len(dot_centers) >= 3:
        cands.append(tuple(np.mean(dot_centers, axis=0)))
    return cands


def analyze_symmetry(binary: np.ndarray, stroke_width: float, dot_centers=None,
                     threshold: float = 0.93, size: int = 384, tol_factor: float = 0.4,
                     strict_threshold: float = 0.985) -> SymmetryReport:
    binary = (binary > 0).astype(np.uint8) * 255
    # pick the centre giving the best 180-degree rotation score
    best = None
    for cen in _candidate_centers(binary, dot_centers):
        A, sc = _make_canvas(binary, cen, size)
        tol = max(1, int(round(stroke_width * sc * tol_factor)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        c = ((size - 1) / 2, (size - 1) / 2)
        s180 = _tol_score(A, cv2.dilate(A, k), _rotate(A, 180, c), k)
        if best is None or s180 > best[0]:
            best = (s180, cen, A, sc, k, c)
    _, cen, A, sc, k, c = best
    dA = cv2.dilate(A, k)

    angles = np.arange(180)
    mscores = np.array([_tol_score(A, dA, _reflect(A, a, c), k) for a in angles])
    rscores = {n: _tol_score(A, dA, _rotate(A, 360.0 / n, c), k) for n in (2, 3, 4, 5, 6, 8)}
    base_m = float(np.median(mscores))
    off_angles = [13, 23, 37, 53, 67, 79, 101, 113, 127, 139, 151, 163, 173]
    base_r = float(np.median([_tol_score(A, dA, _rotate(A, a, c), k) for a in off_angles]))

    def classify(thr):
        axes = []
        for a in angles:
            win = [mscores[(a + d) % 180] for d in range(-4, 5)]
            if mscores[a] >= max(win) and mscores[a] >= thr and mscores[a] - base_m >= 0.10:
                if not axes or (a - axes[-1]["angle_deg"]) > 4:
                    axes.append(dict(angle_deg=float(a), score=float(mscores[a])))
        if len(axes) > 1 and (axes[0]["angle_deg"] + 180 - axes[-1]["angle_deg"]) <= 4:
            axes.pop()
        order = 1
        for n in (2, 3, 4, 5, 6, 8):
            if rscores[n] >= thr and rscores[n] - base_r >= 0.10:
                order = n
        if order > 1:
            grp = f"D{order}" if len(axes) >= order else f"C{order}"
        else:
            grp = "D1" if axes else "C1"
        return axes, order, grp

    axes, order, group = classify(threshold)
    _, _, group_strict = classify(strict_threshold)
    return SymmetryReport((cen[0], cen[1]), angles, mscores, axes, rscores, order, base_m, base_r, group,
                          threshold, sc, group_strict, strict_threshold)


# --------------------------------------------------------- discrete (grid) level
def discrete_symmetries(model) -> dict:
    """Exact symmetry test on the reconstructed model: fraction of free nodes whose
    state agrees with the state of their image under each operation (0 if the dot
    mask itself is not symmetric under that operation)."""
    from .groups import all_ops, mask_symmetric
    out = {}
    for name, op in all_ops(model.R, model.C).items():
        if not mask_symmetric(model.mask, op):
            out[name] = 0.0
            continue
        agree = total = 0
        for p in model.free_nodes:
            q = op(p)
            if q in model.states:
                total += 1
                s = model.states[p]
                if op.swaps:
                    s = {"X": "X", "H": "V", "V": "H"}[s]
                agree += model.states[q] == s
        out[name] = agree / total if total else 1.0
    return out


def state_periodicity(model, min_agree: float = 0.92):
    """Smallest translation (in node units, even numbers) under which the node
    states repeat; returns list of dicts sorted by period."""
    found = []
    xs = [n[0] for n in model.free_nodes]
    ys = [n[1] for n in model.free_nodes]
    if not xs:
        return found
    for axis, span in ((0, max(xs) - min(xs)), (1, max(ys) - min(ys))):
        for p in range(2, span, 2):
            agree = tot = 0
            for n in model.free_nodes:
                m = (n[0] + p, n[1]) if axis == 0 else (n[0], n[1] + p)
                if m in model.states:
                    tot += 1
                    agree += model.states[m] == model.states[n]
            if tot >= 8 and agree / tot >= min_agree:
                found.append(dict(axis="x" if axis == 0 else "y", period_dots=p // 2,
                                  agreement=round(agree / tot, 3), overlap_nodes=tot))
                break
    return found


def autocorrelation_peaks(binary: np.ndarray, max_peaks: int = 6, size: int = 256) -> list:
    """Translational repetition from the FFT autocorrelation of the drawing."""
    m = cv2.resize((binary > 0).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    m -= m.mean()
    F = np.fft.rfft2(m, s=(2 * size, 2 * size))
    ac = np.fft.irfft2(F * np.conj(F))
    ac = np.fft.fftshift(ac) / (ac.max() + 1e-9)
    ac = cv2.GaussianBlur(ac.astype(np.float32), (0, 0), 1.2)
    cy = cx = size
    dil = cv2.dilate(ac, np.ones((9, 9), np.uint8))
    peaks = np.argwhere((ac >= dil - 1e-9) & (ac > 0.25))
    out = []
    for (y, x) in peaks:
        dy, dx = y - cy, x - cx
        r = np.hypot(dx, dy)
        if r < 6:
            continue
        out.append(dict(dx=float(dx), dy=float(dy), distance=float(r), strength=float(ac[y, x])))
    out.sort(key=lambda d: -d["strength"])
    return out[:max_peaks]
