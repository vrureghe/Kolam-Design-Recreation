"""
kolam.generator -- Module 6: rule-based kolam generation.

GENERATION RULE (the "design principle" being used)
---------------------------------------------------
1. Choose a dot arrangement (mask): square, rectangle, diamond, cross, octagon.
2. Choose a symmetry group G (e.g. D4).  Node states are tied together along
   G-orbits, so the whole design is symmetric *by construction*.
3. Choose the state (X / H / V) of each orbit so that the number of closed
   loops is as small as possible (target 1 = one-stroke / sikku kolam).
   This is a stochastic local search (simulated-annealing style) because the
   number of loops is a global property of all node states.
4. Optionally add translation constraints (repetition of a motif).

Because the loops always run around the dots, the result is a closed-curve
pulli kolam, not an arbitrary geometric figure.  Limits: square dot lattices
only; one-stroke designs are not always reachable for a given symmetry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .groups import Affine, compatible, generators, GROUP_NAMES
from .model import KolamModel, STATES, swap_state


class InconsistentSymmetry(ValueError):
    """Raised when the requested symmetry forces a node to be its own H/V swap."""


# ----------------------------------------------------------------------- masks
def make_mask(shape: str = "square", n: int = 5, m: int | None = None) -> np.ndarray:
    """Boolean dot masks (rows x cols)."""
    if shape == "square":
        return np.ones((n, n), bool)
    if shape == "rect":
        return np.ones((n, m if m else n + 2), bool)
    if shape in ("diamond", "cross", "octagon"):
        if n % 2 == 0:
            n += 1
        k = n // 2
        r, c = np.mgrid[0:n, 0:n]
        if shape == "diamond":
            return (abs(r - k) + abs(c - k)) <= k
        if shape == "cross":
            w = 1 if n >= 5 else 0
            return (abs(r - k) <= w) | (abs(c - k) <= w)
        cut = max(1, n // 3)
        mask = np.ones((n, n), bool)
        for (rr, cc) in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)):
            for i in range(cut):
                for j in range(cut - i):
                    mask[abs(rr - i), abs(cc - j)] = False
        return mask
    raise ValueError(f"unknown shape {shape!r}")


MASK_SHAPES = ("square", "rect", "diamond", "cross", "octagon")


# ---------------------------------------------------------------------- orbits
def compute_orbits(model: KolamModel, relations: list[Affine]):
    """Group free nodes into orbits under the given maps.

    Returns a list of orbits; each orbit is a list of (node, flag) where flag=1
    means 'this node uses the H<->V swapped state of the orbit representative'.
    """
    free = set(model.free_nodes)
    adj = {n: [] for n in free}
    for g in relations:
        flag = 1 if g.swaps else 0
        for p in model.free_nodes:
            q = g(p)
            if q in free:
                adj[p].append((q, flag))
                adj[q].append((p, flag))
    seen: dict = {}
    orbits = []
    for p in model.free_nodes:
        if p in seen:
            continue
        seen[p] = 0
        stack, members = [p], [(p, 0)]
        while stack:
            a = stack.pop()
            for b, f in adj[a]:
                nf = seen[a] ^ f
                if b in seen:
                    if seen[b] != nf:
                        raise InconsistentSymmetry("symmetry forces H and V to coincide")
                else:
                    seen[b] = nf
                    stack.append(b)
                    members.append((b, nf))
        orbits.append(members)
    return orbits


def _assign(model: KolamModel, orbit, s: str):
    t = swap_state(s)
    for n, f in orbit:
        model.states[n] = s if f == 0 else t


def translation_relations(model: KolamModel, period_x: int | None, period_y: int | None):
    rel = []
    if period_x:
        rel.append(Affine((1, 0, 0, 1), (2 * period_x, 0), "tx"))
    if period_y:
        rel.append(Affine((1, 0, 0, 1), (0, 2 * period_y), "ty"))
    return rel


# ---------------------------------------------------------------------- search
def _weights_array(weights) -> np.ndarray:
    if isinstance(weights, dict):
        w = np.array([weights.get(s, 0.0) for s in STATES], float)
    else:
        w = np.array(weights, float)
    if w.sum() <= 0:
        w = np.ones(3)
    return w / w.sum()


def search_states(model: KolamModel, orbits, rng, target_loops: int = 1, iters: int = 4000,
                  weights=(0.4, 0.3, 0.3), restarts: int = 6, temp: float = 0.35):
    """Simulated-annealing-like search that minimises the loop count.

    Returns (loops_found, evaluations_used).
    """
    w = _weights_array(weights)
    n = len(orbits)
    if n == 0:
        return model.n_loops(), 1
    best_l, best_states, evals = 10 ** 9, None, 0
    per = max(1, iters // restarts)
    for _ in range(restarts):
        cur_states = [STATES[int(rng.choice(3, p=w))] for _ in range(n)]
        for o, s in zip(orbits, cur_states):
            _assign(model, o, s)
        cur = model.n_loops()
        evals += 1
        if cur < best_l:
            best_l, best_states = cur, dict(model.states)
        for _ in range(per):
            if cur <= target_loops:
                break
            i = int(rng.integers(n))
            old = cur_states[i]
            new = STATES[int(rng.choice([k for k in range(3) if STATES[k] != old]))]
            _assign(model, orbits[i], new)
            l = model.n_loops()
            evals += 1
            if l <= cur or rng.random() < math.exp(-(l - cur) / temp):
                cur, cur_states[i] = l, new
                if cur < best_l:
                    best_l, best_states = cur, dict(model.states)
            else:
                _assign(model, orbits[i], old)
        if best_l <= target_loops:
            break
    model.states.update(best_states)
    return best_l, evals


# ---------------------------------------------------------------------- result
@dataclass
class GeneratedKolam:
    model: KolamModel
    group: str
    shape: str
    size: int
    seed: int | None
    target_loops: int
    n_loops: int
    achieved: bool
    method: str
    n_orbits: int = 0
    evaluations: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.shape} {self.size}  group={self.group}  dots={self.model.n_dots}  "
                f"loops={self.n_loops}  one-stroke={'yes' if self.n_loops == 1 else 'no'}  "
                f"({self.method})")


# --------------------------------------------------------------------- generate
def generate_kolam(shape: str = "square", size: int = 5, group: str = "D4", target_loops: int = 1,
                   seed: int | None = None, iters: int = 4000, weights=(0.4, 0.3, 0.3),
                   mask: np.ndarray | None = None, cols: int | None = None) -> GeneratedKolam:
    """Dot-grid + symmetry + rule-based generation of a new kolam."""
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = make_mask(shape, size, cols)
    notes = []
    if group not in GROUP_NAMES:
        raise ValueError(f"unknown group {group!r}")
    if not compatible(mask, group):
        raise ValueError(f"dot arrangement {shape} {mask.shape} is not invariant under '{group}'")
    model = KolamModel(mask)
    try:
        orbits = compute_orbits(model, generators(group, model.R, model.C))
    except InconsistentSymmetry:
        notes.append("symmetry inconsistent; fell back to 'none'")
        group = "none"
        orbits = compute_orbits(model, [])
    l, ev = search_states(model, orbits, rng, target_loops, iters, weights)
    return GeneratedKolam(model, group, shape, size, seed, target_loops, l, l <= target_loops,
                          "symmetric orbit search", len(orbits), ev, notes)


def generate_repeated(motif: int = 3, reps: tuple = (2, 2), group: str = "none", target_loops: int = 1,
                      seed: int | None = None, iters: int = 4000, weights=(0.4, 0.3, 0.3)) -> GeneratedKolam:
    """Repetition-based generation: a motif x motif block of node states is tiled reps[0] x reps[1] times."""
    rng = np.random.default_rng(seed)
    rows, cols = motif * reps[1], motif * reps[0]
    mask = np.ones((rows, cols), bool)
    if not compatible(mask, group):
        raise ValueError(f"{rows}x{cols} block not invariant under '{group}'")
    model = KolamModel(mask)
    trans = translation_relations(model, motif if reps[0] > 1 else None, motif if reps[1] > 1 else None)
    notes = []
    try:
        orbits = compute_orbits(model, generators(group, rows, cols) + trans)
    except InconsistentSymmetry:
        notes.append("translation + symmetry inconsistent; used translation only")
        group = "none"
        orbits = compute_orbits(model, trans)
    l, ev = search_states(model, orbits, rng, target_loops, iters, weights)
    return GeneratedKolam(model, group, f"repeat {motif}x{motif} tiled {reps[0]}x{reps[1]}", motif, seed,
                          target_loops, l, l <= target_loops, "motif repetition", len(orbits), ev, notes)


def mutate(g: GeneratedKolam, n_flips: int = 2, seed: int | None = None, max_loops: int | None = None,
           tries: int = 300) -> GeneratedKolam:
    """Randomised variation that PRESERVES the symmetry group and does not
    increase the loop count above max_loops (default: current loop count)."""
    rng = np.random.default_rng(seed)
    limit = g.n_loops if max_loops is None else max_loops
    base = g.model
    orbits = compute_orbits(base, generators(g.group, base.R, base.C))
    for _ in range(tries):
        m = base.copy()
        idx = rng.choice(len(orbits), size=min(n_flips, len(orbits)), replace=False) if orbits else []
        for i in idx:
            rep, flag = orbits[i][0]
            cur = m.states[rep] if flag == 0 else swap_state(m.states[rep])
            new = STATES[int(rng.choice([k for k in range(3) if STATES[k] != cur]))]
            _assign(m, orbits[i], new)
        l = m.n_loops()
        if l <= limit and any(m.states[n] != base.states[n] for n in m.states):
            return GeneratedKolam(m, g.group, g.shape, g.size, seed, g.target_loops, l, l <= g.target_loops,
                                  "mutation of " + g.method, len(orbits), tries, list(g.notes))
    out = GeneratedKolam(base.copy(), g.group, g.shape, g.size, seed, g.target_loops, g.n_loops, g.achieved,
                         "mutation failed (returned original)", len(orbits), tries, list(g.notes))
    return out
