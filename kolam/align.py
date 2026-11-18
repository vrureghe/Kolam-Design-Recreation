"""kolam.align -- compare two models up to the 8 symmetries of the square (evaluation helper)."""
from __future__ import annotations

from .model import swap_state

_LINEAR = {
    "id": (1, 0, 0, 1), "rot90": (0, -1, 1, 0), "rot180": (-1, 0, 0, -1), "rot270": (0, 1, -1, 0),
    "flip_x": (-1, 0, 0, 1), "flip_y": (1, 0, 0, -1), "transpose": (0, 1, 1, 0), "antitranspose": (0, -1, -1, 0),
}


def compare_models(gt, det):
    """Best state agreement between ground-truth model gt and detected model det.

    Returns dict(aligned, op, accuracy, n_compared, loops_equal) -- aligned False when the dot
    masks cannot be matched by any of the 8 operations."""
    det_dots = set(det.dots)
    best = dict(aligned=False, op=None, accuracy=0.0, n_compared=0)
    for name, (a, b, c, d) in _LINEAR.items():
        mapped = [(a * x + b * y, c * x + d * y) for (x, y) in gt.dots]
        mx = min(p[0] for p in mapped); my = min(p[1] for p in mapped)
        sx, sy = 1 - mx, 1 - my
        if {(p[0] + sx, p[1] + sy) for p in mapped} != det_dots:
            continue
        swaps = b != 0 or c != 0
        agree = tot = 0
        for n, s in gt.states.items():
            q = (a * n[0] + b * n[1] + sx, c * n[0] + d * n[1] + sy)
            if q in det.states:
                tot += 1
                agree += det.states[q] == (swap_state(s) if swaps else s)
        acc = agree / tot if tot else 1.0
        if not best["aligned"] or acc > best["accuracy"]:
            best = dict(aligned=True, op=name, accuracy=acc, n_compared=tot)
    return best
