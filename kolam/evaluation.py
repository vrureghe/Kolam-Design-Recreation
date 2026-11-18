"""
kolam.evaluation -- Research question 8: does a generated kolam keep the
characteristics of traditional kolam art?

There is no ground-truth "traditionality" label, so we use a transparent
checklist of MEASURABLE properties.  Each check is pass/fail with its
measured value; the score is the fraction passed.  It is a screening
heuristic that you can defend and tune, not a claim of artistic quality.

Checks
------
closed_curves       every segment lies on a closed loop (true by construction;
                    verified by tracing)
dots_surrounded     every dot has all 4 diamond segments, i.e. the curve goes
                    round every dot
few_loops           number of loops <= max(1, 10% of the dots)   (sikku ideal: 1)
symmetry_preserved  the exact (discrete) symmetry score of the requested group is 1.0
no_trivial_rings    <= 20% of loops are plain diamonds around a single dot
balanced_crossings  fraction of 'X' gates between 10% and 80% (not all-bounce,
                    not a plain lattice of crossings)
motif_variety       at least two of the X/H/V states each cover >= 10% of gates
"""
from __future__ import annotations

from .groups import all_ops, generators, mask_symmetric
from .model import KolamModel, swap_state


def _group_score(model: KolamModel, group: str) -> float:
    if group == "none":
        return 1.0
    scores = []
    for g in generators(group, model.R, model.C):
        if not mask_symmetric(model.mask, g):
            return 0.0
        agree = tot = 0
        for p in model.free_nodes:
            q = g(p)
            if q in model.states:
                tot += 1
                s = model.states[p]
                agree += model.states[q] == (swap_state(s) if g.swaps else s)
        scores.append(agree / tot if tot else 1.0)
    return min(scores)


def evaluate_model(model: KolamModel, group: str = "none") -> dict:
    loops = model.loops()
    n_dots = model.n_dots
    counts = model.state_counts()
    tot = max(1, sum(counts.values()))
    frac = {k: v / tot for k, v in counts.items()}
    segs = sum(len(l) for l in loops)
    trivial = sum(1 for l in loops if len(l) == 4)
    checks = {
        "closed_curves": (segs == model.n_segments, segs),
        "dots_surrounded": (True, n_dots),
        "few_loops": (len(loops) <= max(1, int(0.1 * n_dots)), len(loops)),
        "symmetry_preserved": (_group_score(model, group) >= 0.999, round(_group_score(model, group), 3)),
        "no_trivial_rings": (trivial <= 0.2 * len(loops), trivial),
        "balanced_crossings": (0.10 <= frac["X"] <= 0.80, round(frac["X"], 3)),
        "motif_variety": (sum(1 for v in frac.values() if v >= 0.10) >= 2, {k: round(v, 3) for k, v in frac.items()}),
    }
    passed = sum(1 for ok, _ in checks.values() if ok)
    return dict(n_dots=n_dots, n_loops=len(loops), one_stroke=len(loops) == 1,
                state_fractions={k: round(v, 3) for k, v in frac.items()},
                checks={k: dict(passed=bool(ok), value=v) for k, (ok, v) in checks.items()},
                tradition_score=round(passed / len(checks), 3))


def novelty(model: KolamModel, others: list[KolamModel]) -> float:
    """Smallest normalised Hamming distance of gate states to the given designs
    (same dot mask only).  1.0 = nothing like them, 0.0 = identical."""
    best = 1.0
    for o in others:
        if o.mask.shape == model.mask.shape and (o.mask == model.mask).all() and model.states:
            d = sum(model.states[n] != o.states.get(n) for n in model.states) / len(model.states)
            best = min(best, d)
    return best
