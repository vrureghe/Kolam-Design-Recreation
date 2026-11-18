"""
kolam.groups -- symmetry operations on the node-coordinate plane.

Each symmetry is an affine map  p' = L p + t  with L an integer 2x2 matrix
(entries 0/+-1), acting on node coordinates (x, y) of a model with R rows and
C columns of dots (node domain [0, 2C] x [0, 2R], centre at (C, R)).

A map that exchanges the x and y axes (a 90-degree rotation or a diagonal
mirror) also exchanges the 'H' and 'V' node states, because H pairs
directions that share dy while V pairs directions that share dx.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Affine:
    L: tuple          # (a, b, c, d)  ->  [[a, b], [c, d]]
    t: tuple          # (tx, ty)
    name: str = ""

    def __call__(self, p):
        x, y = p
        a, b, c, d = self.L
        return (a * x + b * y + self.t[0], c * x + d * y + self.t[1])

    @property
    def swaps(self) -> bool:
        """True if the map exchanges the x and y axes (so H <-> V)."""
        return self.L[1] != 0 or self.L[2] != 0


def all_ops(R: int, C: int) -> dict:
    """Every non-trivial D4 operation that maps the node box onto itself."""
    ops = {
        "flip_x": Affine((-1, 0, 0, 1), (2 * C, 0), "flip_x"),      # vertical mirror axis
        "flip_y": Affine((1, 0, 0, -1), (0, 2 * R), "flip_y"),      # horizontal mirror axis
        "rot180": Affine((-1, 0, 0, -1), (2 * C, 2 * R), "rot180"),
    }
    if R == C:
        n = R
        ops["rot90"] = Affine((0, -1, 1, 0), (2 * n, 0), "rot90")
        ops["rot270"] = Affine((0, 1, -1, 0), (0, 2 * n), "rot270")
        ops["transpose"] = Affine((0, 1, 1, 0), (0, 0), "transpose")          # main diagonal mirror
        ops["antitranspose"] = Affine((0, -1, -1, 0), (2 * n, 2 * n), "antitranspose")
    return ops


GROUP_NAMES = ("none", "mirror_v", "mirror_h", "mirror_vh", "rot2", "rot4", "diag", "D4")

GROUP_DESCRIPTIONS = {
    "none": "no imposed symmetry",
    "mirror_v": "left-right mirror (vertical axis)",
    "mirror_h": "top-bottom mirror (horizontal axis)",
    "mirror_vh": "both mirrors (dihedral D2, includes 180 deg rotation)",
    "rot2": "180-degree rotation (cyclic C2)",
    "rot4": "90-degree rotation (cyclic C4)",
    "diag": "diagonal mirror",
    "D4": "full square symmetry (4 rotations + 4 mirrors)",
}


def generators(name: str, R: int, C: int) -> list[Affine]:
    ops = all_ops(R, C)
    table = {
        "none": [],
        "mirror_v": ["flip_x"],
        "mirror_h": ["flip_y"],
        "mirror_vh": ["flip_x", "flip_y"],
        "rot2": ["rot180"],
        "rot4": ["rot90"],
        "diag": ["transpose"],
        "D4": ["rot90", "flip_x"],
    }
    if name not in table:
        raise ValueError(f"unknown group {name!r}; choose from {GROUP_NAMES}")
    need = table[name]
    if any(n not in ops for n in need):
        raise ValueError(f"group {name!r} needs a square dot box (got {R}x{C})")
    return [ops[n] for n in need]


def mask_symmetric(mask: np.ndarray, op: Affine) -> bool:
    R, C = mask.shape
    dots = {(2 * c + 1, 2 * r + 1) for r in range(R) for c in range(C) if mask[r, c]}
    return all(op(p) in dots for p in dots)


def compatible(mask: np.ndarray, name: str) -> bool:
    """Is the dot mask itself invariant under the group's generators?"""
    R, C = mask.shape
    try:
        gens = generators(name, R, C)
    except ValueError:
        return False
    return all(mask_symmetric(mask, g) for g in gens)


def largest_compatible_group(mask: np.ndarray) -> str:
    for name in ("D4", "rot4", "mirror_vh", "diag", "rot2", "mirror_v", "mirror_h"):
        if compatible(mask, name):
            return name
    return "none"


def infer_group_from_ops(scores: dict, thr: float = 0.95) -> str:
    """Map a dict {op name: agreement score} onto the closest named group."""
    ok = {k for k, v in scores.items() if v >= thr}
    if {"rot90", "flip_x"} <= ok or {"rot90", "transpose"} <= ok:
        return "D4"
    if "rot90" in ok:
        return "rot4"
    if {"flip_x", "flip_y"} <= ok:
        return "mirror_vh"
    if "transpose" in ok or "antitranspose" in ok:
        return "diag"
    if "rot180" in ok:
        return "rot2"
    if "flip_x" in ok:
        return "mirror_v"
    if "flip_y" in ok:
        return "mirror_h"
    return "none"
