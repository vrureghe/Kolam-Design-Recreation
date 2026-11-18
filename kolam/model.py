"""
kolam.model -- the combinatorial "loop model" of a dot (pulli) kolam.

IDEA (beginner-friendly)
------------------------
A pulli kolam is drawn around a lattice of dots.  We use *node coordinates*
that are twice as fine as the dot lattice:

        dot in column c, row r      ->  node coordinate (2c+1, 2r+1)   (odd, odd)
        midpoint between two dots   ->  a "node" with x+y odd

Every dot is surrounded by a small diamond made of 4 segments that join the
midpoints of its four sides (W, N, E, S).  At a node shared by two neighbouring
dots, 4 segments meet (2 from each dot) and the artist decides how the curve
passes through that node.  There are exactly three ways to pair 4 diagonal
directions (NW, NE, SE, SW):

    "X"  straight through   NW<->SE and NE<->SW   (two strands CROSS)
    "H"  bounce, same y     NW<->NE and SW<->SE   (strands turn back; 'horizontal' pairing)
    "V"  bounce, same x     NW<->SW and NE<->SE   (strands turn back; 'vertical' pairing)

Once every node has a state, following the pairings splits the whole drawing
into closed loops.  A "one-stroke" (sikku / kambi) kolam is the special case
with exactly ONE loop.  This is the same idea as the mirror/ray models used in
the kolam literature (Ascher; Siromoney et al.), written as a loop model.

Nodes on the outside of the pattern only have 2 segments, so the curve simply
bounces there; it has no choice (state irrelevant).
"""
from __future__ import annotations

import numpy as np

DIR_NW, DIR_NE, DIR_SE, DIR_SW = (-1, -1), (1, -1), (1, 1), (-1, 1)
DIRS = (DIR_NW, DIR_NE, DIR_SE, DIR_SW)
STATES = ("X", "H", "V")
_SWAP = {"X": "X", "H": "V", "V": "H"}


def swap_state(s: str) -> str:
    """H <-> V (what a 90-degree rotation or a diagonal mirror does)."""
    return _SWAP[s]


def partner(state: str, d_in):
    """Given the direction we ARRIVED from (pointing from the node back to the
    previous node), return the direction we LEAVE along, for a free node."""
    dx, dy = d_in
    if state == "X":
        return (-dx, -dy)
    if state == "H":
        return (-dx, dy)
    if state == "V":
        return (dx, -dy)
    raise ValueError(f"bad state {state!r}")


class KolamModel:
    """A dot mask plus a state (X/H/V) for every free node.

    mask : 2-D bool array, shape (rows, cols); True where a dot exists.
    """

    def __init__(self, mask, states: dict | None = None):
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.ndim != 2 or not self.mask.any():
            raise ValueError("mask must be a 2-D array with at least one dot")
        self.R, self.C = self.mask.shape
        self.dots = [(2 * c + 1, 2 * r + 1)
                     for r in range(self.R) for c in range(self.C) if self.mask[r, c]]
        self.seg_dirs: dict[tuple, set] = {}
        for (x, y) in self.dots:
            ring = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]  # W N E S
            for i in range(4):
                a, b = ring[i], ring[(i + 1) % 4]
                d = (b[0] - a[0], b[1] - a[1])
                self.seg_dirs.setdefault(a, set()).add(d)
                self.seg_dirs.setdefault(b, set()).add((-d[0], -d[1]))
        self.free_nodes = sorted(n for n, s in self.seg_dirs.items() if len(s) == 4)
        self.states = {n: "X" for n in self.free_nodes}
        if states:
            self.set_states(states)

    # ------------------------------------------------------------------ utils
    def set_states(self, states: dict):
        for n, s in states.items():
            if n in self.states:
                if s not in STATES:
                    raise ValueError(f"bad state {s!r}")
                self.states[n] = s

    def copy(self) -> "KolamModel":
        return KolamModel(self.mask.copy(), dict(self.states))

    @property
    def n_dots(self) -> int:
        return len(self.dots)

    @property
    def n_segments(self) -> int:
        return 4 * self.n_dots

    # ---------------------------------------------------------------- tracing
    def _next_dir(self, node, d_in):
        dirs = self.seg_dirs[node]
        if len(dirs) == 2:                      # outer node: forced bounce
            for d in dirs:
                if d != d_in:
                    return d
            return d_in
        return partner(self.states[node], d_in)

    def loops(self) -> list[list[tuple]]:
        """Closed loops as ordered lists of nodes (each loop is a closed polygon).

        Walk along a segment, arrive at a node, look up the state, leave along
        the paired segment, repeat until we return to the start.
        """
        visited = set()
        out = []
        for node in sorted(self.seg_dirs):
            for d in sorted(self.seg_dirs[node]):
                if (node, d) in visited:
                    continue
                loop, p, dd = [], node, d
                while (p, dd) not in visited:
                    visited.add((p, dd))
                    q = (p[0] + dd[0], p[1] + dd[1])
                    visited.add((q, (-dd[0], -dd[1])))
                    loop.append(p)
                    dd = self._next_dir(q, (-dd[0], -dd[1]))
                    p = q
                out.append(loop)
        return out

    def n_loops(self) -> int:
        """Same as len(self.loops()) but cheaper (no node lists)."""
        visited = set()
        count = 0
        for node in self.seg_dirs:
            for d in self.seg_dirs[node]:
                if (node, d) in visited:
                    continue
                count += 1
                p, dd = node, d
                while (p, dd) not in visited:
                    visited.add((p, dd))
                    q = (p[0] + dd[0], p[1] + dd[1])
                    visited.add((q, (-dd[0], -dd[1])))
                    dd = self._next_dir(q, (-dd[0], -dd[1]))
                    p = q
        return count

    # --------------------------------------------------------------- reporting
    def state_counts(self) -> dict:
        out = {s: 0 for s in STATES}
        for s in self.states.values():
            out[s] += 1
        return out

    def ascii(self) -> str:
        """Small text picture: 'o' dots, X/H/V free nodes, '+' forced boundary nodes."""
        W, Hh = 2 * self.C + 1, 2 * self.R + 1
        rows = []
        dots = set(self.dots)
        for y in range(Hh):
            line = []
            for x in range(W):
                if (x, y) in dots:
                    line.append("o")
                elif (x, y) in self.states:
                    line.append(self.states[(x, y)])
                elif (x, y) in self.seg_dirs:
                    line.append("+")
                else:
                    line.append(" ")
            rows.append("".join(line))
        return "\n".join(rows)
