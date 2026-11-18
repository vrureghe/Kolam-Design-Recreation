"""
kolam.topology -- Module 4 (part): line intersections, curve continuity, connectivity.

The 1-pixel skeleton of the curves is turned into a multigraph:
    graph nodes  = junction clusters (>=3 skeleton neighbours) and free end points
    graph edges  = skeleton branches between them (a closed ring with no
                   junction becomes a self-loop on its own node)

MATH
----
For a planar graph with V nodes, E edges and k connected components the number
of independent cycles (= bounded faces of the drawing) is the cyclomatic number

        beta_1 = E - V + k          (Euler: V - E + F = 1 + k)

A kolam curve is *closed* when it has no end points (degree-1 nodes); a
crossing of two strands shows up as a junction of degree 4.

The box-counting dimension D = lim log N(s) / log(1/s) (N(s) = number of
s x s boxes touched) summarises how space-filling the curve is.
"""
from __future__ import annotations

import cv2
import networkx as nx
import numpy as np

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    s = (skel > 0).astype(np.uint8)
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return (cv2.filter2D(s, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT).round().astype(np.int32)) * s


def prune_spurs(skel: np.ndarray, min_len: int, passes: int = 2) -> np.ndarray:
    """Remove short side branches (skeletonisation artefacts) shorter than min_len pixels."""
    sk = (skel > 0).copy()
    H, W = sk.shape
    for _ in range(passes):
        nb = neighbor_count(sk)
        changed = False
        for (y, x) in np.argwhere(sk & (nb == 1)):
            path, prev, cur, hit = [(y, x)], None, (y, x), False
            while len(path) <= min_len:
                cy, cx = cur
                if nb[cy, cx] >= 3:
                    hit = True
                    path.pop()
                    break
                nxt = [(cy + dy, cx + dx) for dy, dx in _OFFS
                       if 0 <= cy + dy < H and 0 <= cx + dx < W and sk[cy + dy, cx + dx]
                       and (cy + dy, cx + dx) != prev and (cy + dy, cx + dx) not in path]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            if hit and path:
                for (py, px) in path:
                    sk[py, px] = False
                changed = True
        if not changed:
            break
    return (sk * 255).astype(np.uint8)


def skeleton_graph(skel: np.ndarray, merge_radius: int = 2):
    """Build the multigraph described in the module docstring."""
    sk = skel > 0
    nb = neighbor_count(sk)
    junc = (sk & (nb >= 3)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_radius + 1, 2 * merge_radius + 1))
    jd = cv2.dilate(junc, k) > 0
    jd &= sk
    nj, jlab = cv2.connectedComponents(jd.astype(np.uint8), connectivity=8)
    branches = sk & ~jd
    nbr, blab, stats, _ = cv2.connectedComponentsWithStats(branches.astype(np.uint8), connectivity=8)
    endpoints = sk & (nb == 1)
    G = nx.MultiGraph()
    for j in range(1, nj):
        G.add_node(("J", j))
    ring_id = 0
    for b in range(1, nbr):
        x, y, w, h, area = stats[b]
        if area < 3:
            continue
        y0, y1, x0, x1 = max(0, y - 2), min(sk.shape[0], y + h + 2), max(0, x - 2), min(sk.shape[1], x + w + 2)
        comp = (blab[y0:y1, x0:x1] == b).astype(np.uint8)
        touch = cv2.dilate(comp, np.ones((3, 3), np.uint8)) > 0
        contact = touch & (jlab[y0:y1, x0:x1] > 0)
        ends = []
        if contact.any():
            nc, clab = cv2.connectedComponents(contact.astype(np.uint8), connectivity=8)
            for c in range(1, nc):
                ends.append(("J", int(jlab[y0:y1, x0:x1][clab == c][0])))
        ep = np.argwhere((comp > 0) & endpoints[y0:y1, x0:x1])
        for i, _ in enumerate(ep):
            ends.append(("E", b, i))
        if len(ends) == 0:
            ring_id += 1
            n = ("R", ring_id)
            G.add_edge(n, n, length=int(area))
        elif len(ends) == 1:
            n = ("E", b, "x")
            G.add_edge(ends[0], n, length=int(area))
        else:
            G.add_edge(ends[0], ends[1], length=int(area))
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return G


def topology_report(curve_skel: np.ndarray, stroke_width: float) -> dict:
    sk = prune_spurs(curve_skel, max(4, int(1.5 * stroke_width)))
    G = skeleton_graph(sk)
    V, E, k = G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G) if G.number_of_nodes() else 0
    deg = dict(G.degree())
    junction_deg = [d for n, d in deg.items() if n[0] == "J"]
    ends = sum(1 for n in G.nodes if n[0] == "E")
    n_labels, _ = cv2.connectedComponents((sk > 0).astype(np.uint8), connectivity=8)
    return dict(
        skeleton=sk,
        graph=G,
        n_nodes=V, n_edges=E, n_components=k,
        cyclomatic_number=E - V + k,
        crossings_deg4=sum(1 for d in junction_deg if d == 4),
        t_junctions_deg3=sum(1 for d in junction_deg if d == 3),
        high_deg_junctions=sum(1 for d in junction_deg if d > 4),
        end_points=ends,
        closed_curve_system=(ends == 0),
        curve_pixels=int((sk > 0).sum()),
        skeleton_components=n_labels - 1,
        degree_histogram={int(d): junction_deg.count(d) for d in sorted(set(junction_deg))},
    )


def box_counting_dimension(mask: np.ndarray) -> float:
    m = mask > 0
    h, w = m.shape
    sizes = [s for s in (2, 4, 8, 16, 32, 64) if s < min(h, w) / 4]
    if len(sizes) < 3:
        return float("nan")
    counts = []
    for s in sizes:
        hh, ww = (h // s) * s, (w // s) * s
        blocks = m[:hh, :ww].reshape(hh // s, s, ww // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    slope = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(counts), 1)[0]
    return float(slope)
