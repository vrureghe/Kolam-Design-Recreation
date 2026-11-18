"""
kolam.pipeline -- glue: image -> measurements -> inferred rules -> structured
representation -> new designs.

The result separates
  DETECTED  = things measured directly from pixels (dots, spacing, symmetry
              scores, crossings, end points, ...)
  INFERRED  = conclusions that rest on a model (gate states, symmetry group,
              loop count, design rule).  These are only as good as the
              detection and are reported with confidence / warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import generator as gen
from .dots import DotDetection, GridInfo, detect_dots, fit_lattice
from .evaluation import evaluate_model
from .groups import compatible, infer_group_from_ops, largest_compatible_group
from .model import KolamModel, STATES
from .preprocess import Preprocessed, preprocess, skeletonize_mask
from .representation import rule_descriptor
from .rules import NodeStates, identify_node_states
from .symmetry import (SymmetryReport, analyze_symmetry, autocorrelation_peaks, discrete_symmetries,
                       state_periodicity)
from .topology import box_counting_dimension, topology_report


@dataclass
class AnalysisResult:
    pre: Preprocessed
    dots: DotDetection
    grid: GridInfo
    topology: dict
    symmetry: SymmetryReport
    repetition: list
    fractal_dimension: float
    node_states: NodeStates | None
    model: KolamModel | None
    descriptor: dict | None
    detected: dict
    inferred: dict
    warnings: list = field(default_factory=list)

    # ---------------------------------------------------------------- text
    def report_text(self) -> str:
        d, i = self.detected, self.inferred
        L = ["=== DETECTED (measured from the image) ==="]
        for k, v in d.items():
            L.append(f"  {k}: {v}")
        L.append("=== INFERRED (model-based conclusions) ===")
        for k, v in i.items():
            L.append(f"  {k}: {v}")
        if self.warnings:
            L.append("=== WARNINGS ===")
            L += [f"  - {w}" for w in self.warnings]
        return "\n".join(L)


def _lattice_relative_axes(sym: SymmetryReport, grid: GridInfo) -> list:
    out = []
    for a in sym.mirror_axes:
        if not grid.ok:
            out.append(dict(a, relative="unknown"))
            continue
        psi = (a["angle_deg"] - grid.orientation_deg) % 180
        if min(psi, 180 - psi) < 6:
            rel = "along rows (a1)"
        elif abs(psi - 90) < 6:
            rel = "along columns (a2)"
        elif abs(psi - 45) < 6 or abs(psi - 135) < 6:
            rel = "lattice diagonal"
        else:
            rel = f"{psi:.0f} deg to a1"
        out.append(dict(a, relative=rel))
    return out


def analyze_image(img_bgr: np.ndarray, symmetry_threshold: float = 0.93) -> AnalysisResult:
    warnings = []
    pre = preprocess(img_bgr)
    if pre.method == "adaptive":
        warnings.append(f"uneven lighting / low contrast (separability {pre.separability:.2f}); local threshold used")
    dots = detect_dots(pre.binary, pre.stroke_width)
    centers = dots.centers if len(dots.centers) else np.zeros((0, 2))
    grid = fit_lattice(centers) if len(centers) >= 4 else GridInfo(False, f"{len(centers)} dots detected")
    curve_skel = skeletonize_mask(dots.curve_mask)
    topo = topology_report(curve_skel, pre.stroke_width)
    sym = analyze_symmetry(pre.binary, pre.stroke_width, centers if len(centers) >= 3 else None,
                           threshold=symmetry_threshold)
    ac = autocorrelation_peaks(pre.binary)
    fd = box_counting_dimension(topo["skeleton"])

    ns = model = desc = None
    if grid.ok:
        if grid.collisions:
            warnings.append(f"{grid.collisions} dots mapped to the same lattice site (lattice fit unreliable)")
        if grid.rel_error > 0.08:
            warnings.append(f"dots deviate from a lattice by {100 * grid.rel_error:.1f}% of the spacing")
        ns = identify_node_states(topo["skeleton"], grid)
        model = ns.model
        if ns.unknown_fraction > 0.10:
            warnings.append(f"{100 * ns.unknown_fraction:.0f}% of gates could not be classified (defaulted to X)")
        desc = rule_descriptor(model, sym.group)
    else:
        warnings.append("dot lattice not recovered: " + grid.message + " -- structural rules not available")
    if grid.ok and grid.lattice_type not in ("square",):
        warnings.append(f"lattice type '{grid.lattice_type}': gate-state model and generator assume a square lattice")

    detected = {
        "foreground_polarity": pre.polarity,
        "threshold_method": pre.method,
        "stroke_width_px": round(pre.stroke_width, 1),
        "dots_found": int(len(centers)),
        "dot_method": dots.method,
        "lattice": grid.to_dict() if grid.ok else grid.message,
        "mirror_axes": _lattice_relative_axes(sym, grid),
        "rotation_scores": {int(n): round(float(s), 3) for n, s in sym.rotation_scores.items()},
        "symmetry_baseline(chance)": {"mirror": round(sym.baseline_mirror, 3), "rotation": round(sym.baseline_rotation, 3)},
        "crossings(deg-4 junctions)": topo["crossings_deg4"],
        "t_junctions(deg-3)": topo["t_junctions_deg3"],
        "end_points": topo["end_points"],
        "independent_cycles(faces)": topo["cyclomatic_number"],
        "curve_components": topo["n_components"],
        "box_counting_dimension": round(fd, 3) if fd == fd else None,
        "translation_peaks(top3)": [dict(dx=p["dx"], dy=p["dy"], strength=round(p["strength"], 2)) for p in ac[:3]],
    }
    inferred = {"image_symmetry_group_strict(>=0.985)": sym.group_strict,
                "image_symmetry_group_approx(>=0.93)": sym.group,
                "closed_curve_system": topo["closed_curve_system"]}
    if model is not None:
        disc = discrete_symmetries(model)
        g = infer_group_from_ops(disc, 0.95)
        g = g if compatible(model.mask, g) else "none"
        inferred.update({
            "gate_states(X/H/V)": model.state_counts(),
            "unclassified_gate_fraction": round(ns.unknown_fraction, 3),
            "number_of_closed_loops": model.n_loops(),
            "one_stroke_design": model.n_loops() == 1,
            "discrete_symmetry_group": g,
            "discrete_symmetry_scores": {k: round(float(v), 3) for k, v in disc.items()},
            "state_periodicity": state_periodicity(model),
        })
    return AnalysisResult(pre, dots, grid, topo, sym, ac, fd, ns, model, desc, detected, inferred, warnings)


# ------------------------------------------------------------------ generation
def generate_from_analysis(res: AnalysisResult, seed: int | None = None, group: str | None = None,
                           target_loops: int | None = None, iters: int = 4000) -> gen.GeneratedKolam:
    """Generate a NEW kolam on the SAME dot arrangement, using the rules that were identified:
    the dot mask, the discrete symmetry group, the state proportions and the loop count."""
    if res.model is None:
        raise ValueError("no structural model available -- the dot lattice was not recovered")
    m = res.model
    disc = discrete_symmetries(m)
    g = group or infer_group_from_ops(disc, 0.95)
    if not compatible(m.mask, g):
        g = largest_compatible_group(m.mask) if group is None else g
        if not compatible(m.mask, g):
            g = "none"
    counts = m.state_counts()
    w = np.array([counts[s] for s in STATES], float) + 2.0     # smoothing so no state is impossible
    tgt = target_loops if target_loops is not None else max(1, m.n_loops())
    best = None
    for k in range(6):                                          # make sure the result differs from the input
        s = None if seed is None else seed + k
        out = gen.generate_kolam(shape="from-analysis", size=max(m.R, m.C), group=g, target_loops=tgt,
                                 seed=s, iters=iters, weights=w, mask=m.mask)
        out.method = "rules identified from image"
        same = all(out.model.states[n] == m.states[n] for n in m.states)
        if best is None or (not same and out.n_loops <= (best.n_loops if best else 10 ** 9)):
            best = out
        if not same and out.achieved:
            break
    return best
