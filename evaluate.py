"""Batch evaluation on synthetic kolams with known ground truth.  Usage: python evaluate.py"""
import csv, sys, time
import numpy as np
from kolam.generator import generate_kolam
from kolam.render import render_model
from kolam.synth import perturb
from kolam.pipeline import analyze_image
from kolam.align import compare_models
from kolam.groups import GROUP_NAMES

CONFIGS = [("square", 5, "rot4"), ("square", 7, "rot4"), ("diamond", 7, "rot2"), ("rect", 4, "mirror_vh"),
           ("cross", 7, "D4"), ("octagon", 7, "rot2"), ("square", 5, "D4"), ("square", 6, "none")]
CONDITIONS = {
    "clean": dict(),
    "rot+noise": dict(angle=17, noise=10),
    "blur+shade": dict(blur=1.2, shade=0.35, noise=6),
    "handdrawn-warp": dict(warp=1.5, blur=0.8, noise=6, angle=-8),
}
STRICT = {"none": "C1", "mirror_v": "D1", "mirror_h": "D1", "mirror_vh": "D2", "rot2": "C2", "rot4": "C4", "D4": "D4"}

def main(seeds=3, out="outputs/evaluation.csv"):
    rows = []
    for cond, kw in CONDITIONS.items():
        for shape, size, grp in CONFIGS:
            for s in range(seeds):
                g = generate_kolam(shape, size, grp, seed=s)
                dark = s % 2 == 0
                fg, bg = ((235, 235, 235), (34, 28, 24)) if dark else ((30, 30, 30), (245, 245, 245))
                img = perturb(render_model(g.model, u=28, fg=fg, bg=bg), bg, seed=s, **kw)
                try:
                    r = analyze_image(img)
                except Exception as e:
                    rows.append(dict(cond=cond, shape=shape, size=size, group=grp, seed=s, error=str(e))); continue
                cmp = compare_models(g.model, r.model) if r.model is not None else dict(aligned=False, accuracy=0.0)
                rows.append(dict(cond=cond, shape=shape, size=size, group=grp, seed=s,
                    dots_ok=int(r.dots.centers.shape[0] == g.model.n_dots),
                    lattice_ok=int(r.grid.ok and r.model is not None and cmp["aligned"]),
                    state_acc=round(cmp["accuracy"], 4) if cmp["aligned"] else 0.0,
                    loops_ok=int(r.model is not None and r.model.n_loops() == g.n_loops),
                    sym_strict_ok=int(r.symmetry.group_strict == STRICT[grp]),
                    discrete_group=r.inferred.get("discrete_symmetry_group", "n/a"),
                    sec=0))
    import os; os.makedirs("outputs", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{'condition':16s} n   dots  lattice  stateAcc  loops  symStrict")
    for cond in CONDITIONS:
        R = [r for r in rows if r["cond"] == cond and "error" not in r]
        E = [r for r in rows if r["cond"] == cond and "error" in r]
        n = len(R) + len(E)
        m = lambda k: np.mean([r[k] for r in R]) if R else 0
        print(f"{cond:16s} {n:<3d} {m('dots_ok'):.2f}  {m('lattice_ok'):.2f}     {m('state_acc'):.3f}     {m('loops_ok'):.2f}   {m('sym_strict_ok'):.2f}   errors={len(E)}")
    return rows

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
