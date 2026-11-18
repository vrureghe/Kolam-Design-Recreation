"""
kolam.representation -- Module 5: turn a kolam into structured mathematics.

Five complementary representations of ONE kolam (all derived from a KolamModel):

1. COORDINATE  - dots as integer lattice points + a state per gate.
                 Lets us rebuild the kolam exactly (the dot lattice and a
                 small table of X/H/V choices is the whole design).
2. GRAPH       - the 'loop graph': nodes = gates, edges = curve segments
                 (networkx).  Connected components of the graph = the closed
                 loops; graph measures (degree, cycles) describe topology.
3. MATRIX      - adjacency matrix A (A[i,j]=1 if a curve segment joins gate i
                 and gate j).  Spectral/graph tools apply, and it is a compact
                 'fingerprint' for comparing designs.
4. PARAMETRIC  - each loop as a closed uniform quadratic B-spline
                 C_i(t); the control points are the gate coordinates.
5. RULE-BASED  - the design principle in words/numbers: dot mask, symmetry
                 group, state proportions, number of loops, repetition.
                 This is what the generator consumes to make new designs.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

from .curves import bspline_closed, curve_length
from .groups import largest_compatible_group, infer_group_from_ops, compatible
from .model import KolamModel
from .symmetry import discrete_symmetries, state_periodicity


# 1 -------------------------------------------------------------- coordinates
def coordinate_representation(model: KolamModel) -> dict:
    return dict(
        rows=model.R, cols=model.C,
        dots=[[(x - 1) // 2, (y - 1) // 2] for (x, y) in model.dots],      # (col, row)
        mask=model.mask.astype(int).tolist(),
        gate_states={f"{x},{y}": s for (x, y), s in sorted(model.states.items())},
    )


# 2 -------------------------------------------------------------------- graph
def loop_graph(model: KolamModel) -> nx.Graph:
    G = nx.Graph()
    loops = model.loops()
    for k, loop in enumerate(loops):
        for i, p in enumerate(loop):
            q = loop[(i + 1) % len(loop)]
            G.add_node(p, state=model.states.get(p, "edge"))
            G.add_node(q, state=model.states.get(q, "edge"))
            G.add_edge(p, q, loop=k)
    return G


# 3 -------------------------------------------------------------------- matrix
def adjacency_matrix(model: KolamModel):
    G = loop_graph(model)
    nodes = sorted(G.nodes)
    return nx.to_numpy_array(G, nodelist=nodes, dtype=int), nodes


# 4 ------------------------------------------------------------------ parametric
def parametric_curves(model: KolamModel, samples: int = 24) -> list[np.ndarray]:
    return [bspline_closed(np.array(l, float), samples) for l in model.loops()]


# 5 ------------------------------------------------------------------ rule based
def rule_descriptor(model: KolamModel, detected_group_img: str | None = None) -> dict:
    loops = model.loops()
    counts = model.state_counts()
    tot = max(1, sum(counts.values()))
    disc = discrete_symmetries(model)
    group = infer_group_from_ops(disc, 0.95)
    lengths = [curve_length(bspline_closed(np.array(l, float), 8)) for l in loops]
    return dict(
        dot_mask=model.mask.astype(int).tolist(),
        n_dots=model.n_dots,
        rows=model.R, cols=model.C,
        n_loops=len(loops),
        one_stroke=(len(loops) == 1),
        loop_lengths_node_units=[round(float(x), 2) for x in lengths],
        state_proportions={k: round(v / tot, 3) for k, v in counts.items()},
        state_symmetry_scores={k: round(float(v), 3) for k, v in disc.items()},
        symmetry_group=group if compatible(model.mask, group) else "none",
        image_level_group=detected_group_img,
        periodicity=state_periodicity(model),
        gate_states={f"{x},{y}": s for (x, y), s in sorted(model.states.items())},
    )


def model_from_descriptor(desc: dict) -> KolamModel:
    mask = np.array(desc["dot_mask"], bool)
    states = {tuple(int(v) for v in k.split(",")): s for k, s in desc["gate_states"].items()}
    return KolamModel(mask, states)


def save_json(obj, path: str):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    return path
