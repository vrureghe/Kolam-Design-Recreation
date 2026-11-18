"""
kolam.curves -- turning a loop of nodes into a smooth parametric curve.

MATH
----
A closed uniform *quadratic B-spline* with control points P_0..P_{n-1} has,
on the i-th piece (t in [0, 1)),

    C_i(t) = 1/2 (1-t)^2 P_{i-1} + (-t^2 + t + 1/2) P_i + 1/2 t^2 P_{i+1}

It is C^1-continuous (no kinks), stays inside the convex hull of its control
points and passes through the midpoints of the control polygon's edges.  It is
also the limit of Chaikin's corner-cutting.  We use the loop's node
coordinates as control points: straight-through ('X') nodes give straight
crossings, bounce nodes ('H'/'V') give smooth rounded turns that stay
~0.25 node-units away from the node -- the look of a hand-drawn kolam curve.
"""
from __future__ import annotations

import numpy as np


def bspline_closed(ctrl, samples: int = 12) -> np.ndarray:
    """Sample a closed quadratic B-spline.  ctrl: (n,2) -> (n*samples, 2)."""
    P = np.asarray(ctrl, dtype=float)
    n = len(P)
    if n < 3:
        return P
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    prev, cur, nxt = np.roll(P, 1, 0), P, np.roll(P, -1, 0)
    w0 = 0.5 * (1 - t) ** 2
    w1 = -t ** 2 + t + 0.5
    w2 = 0.5 * t ** 2
    # shape (n, samples, 2)
    pts = (w0[None] * prev[:, None, :] + w1[None] * cur[:, None, :] + w2[None] * nxt[:, None, :])
    return pts.reshape(-1, 2)


def curve_length(points: np.ndarray, closed: bool = True) -> float:
    d = np.diff(np.vstack([points, points[:1]]) if closed else points, axis=0)
    return float(np.sqrt((d ** 2).sum(1)).sum())


def symbolic_bspline():
    """Return the symbolic basis functions with SymPy (used in the docs/notebook)."""
    import sympy as sp
    t = sp.symbols("t")
    basis = [sp.Rational(1, 2) * (1 - t) ** 2, -t ** 2 + t + sp.Rational(1, 2), sp.Rational(1, 2) * t ** 2]
    total = sp.simplify(sum(basis))          # partition of unity -> 1
    return {"t": t, "basis": [sp.expand(b) for b in basis], "sum": total}
