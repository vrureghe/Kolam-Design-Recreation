"""
kolam.preprocess -- Module 2: grayscale, denoise, binarise, edges, contours, skeleton.

KEY STEPS AND THE MATH BEHIND THEM
----------------------------------
* Gaussian blur: convolution with a 2-D Gaussian removes high-frequency noise.
* Otsu threshold: choose t that maximises the between-class variance
      sigma_B^2(t) = w0(t) w1(t) (mu0(t) - mu1(t))^2
  We also report  sigma_B^2 / sigma_T^2  ("separability", 0..1).  If it is low
  (uneven lighting, shadows) we switch to a *local* adaptive threshold.
* Polarity: kolams are white-on-dark OR dark-on-white.  The border of a photo
  is almost always background, so the class that dominates the border is
  background and the other class is the drawing (foreground).
* Skeletonisation (Zhang-Suen style thinning, skimage) reduces strokes to
  1-pixel centre-lines so we can analyse topology.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Preprocessed:
    original: np.ndarray        # BGR, working scale, cropped
    gray: np.ndarray
    denoised: np.ndarray
    binary: np.ndarray          # uint8 0/255, foreground = 255
    skeleton: np.ndarray        # uint8 0/255
    edges: np.ndarray           # uint8 Canny
    contours: list
    hierarchy: object
    scale: float                # working scale / input scale
    crop: tuple                 # (x0, y0, x1, y1) in working-scale coordinates
    polarity: str               # 'light-on-dark' | 'dark-on-light'
    method: str                 # 'otsu' | 'adaptive'
    separability: float
    stroke_width: float         # median stroke width in px (working scale)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def otsu_separability(gray: np.ndarray, thr: float) -> float:
    g = gray.astype(np.float64).ravel()
    lo, hi = g[g <= thr], g[g > thr]
    if len(lo) == 0 or len(hi) == 0 or g.var() < 1e-9:
        return 0.0
    w0, w1 = len(lo) / len(g), len(hi) / len(g)
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / g.var())


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[lab] * 255).astype(np.uint8)


def estimate_stroke_width(binary: np.ndarray, skel: np.ndarray) -> float:
    dt = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt[skel > 0]
    return float(2 * np.median(vals)) if len(vals) else 2.0


def preprocess(img_bgr: np.ndarray, max_side: int = 800, min_side: int = 400,
               margin_frac: float = 0.04) -> Preprocessed:
    h, w = img_bgr.shape[:2]
    s = 1.0
    if max(h, w) > max_side:
        s = max_side / max(h, w)
    elif max(h, w) < min_side:
        s = min_side / max(h, w)
    work = img_bgr if s == 1.0 else cv2.resize(
        img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    den = cv2.GaussianBlur(cv2.medianBlur(gray, 3), (5, 5), 0)

    # --- global Otsu + polarity from the image border
    thr, bright = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sep = otsu_separability(den, thr)
    b = max(2, int(0.03 * min(den.shape)))
    border = np.concatenate([bright[:b].ravel(), bright[-b:].ravel(),
                             bright[:, :b].ravel(), bright[:, -b:].ravel()])
    bright_bg = (border > 0).mean() > 0.5
    polarity = "dark-on-light" if bright_bg else "light-on-dark"
    method = "otsu"
    fg = (255 - bright) if bright_bg else bright

    # --- local threshold when lighting is uneven / contrast is poor
    if sep < 0.5:
        block = _odd(max(31, min(den.shape) // 8))
        if bright_bg:
            fg = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, 8)
        else:
            fg = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, -8)
        method = "adaptive"

    fg = cv2.medianBlur(fg, 3)
    fg = remove_small_components(fg, max(6, int(1e-5 * fg.size)))
    if not fg.any():
        raise ValueError("No foreground found -- is the image blank?")

    # --- crop to the drawing
    ys, xs = np.nonzero(fg)
    m = int(margin_frac * max(fg.shape))
    x0, x1 = max(0, xs.min() - m), min(fg.shape[1], xs.max() + 1 + m)
    y0, y1 = max(0, ys.min() - m), min(fg.shape[0], ys.max() + 1 + m)
    fg, work, gray, den = fg[y0:y1, x0:x1], work[y0:y1, x0:x1], gray[y0:y1, x0:x1], den[y0:y1, x0:x1]

    skel = (skeletonize(fg > 0) * 255).astype(np.uint8)
    edges = cv2.Canny(den, 50, 150)
    contours, hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sw = estimate_stroke_width(fg, skel)
    return Preprocessed(work, gray, den, fg, skel, edges, list(contours), hier, s,
                        (int(x0), int(y0), int(x1), int(y1)), polarity, method, sep, sw)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return (skeletonize(mask > 0) * 255).astype(np.uint8)
