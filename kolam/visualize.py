"""kolam.visualize -- Module 7: figures for every stage of the analysis."""
from __future__ import annotations

import io

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .render import render_model
from .rules import node_pixel

STATE_COLORS = {"X": "#e4572e", "H": "#17bebb", "V": "#ffc914"}


def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _axes(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def plot_original(ax, res):
    ax.imshow(_rgb(res.pre.original)); _axes(ax, "1. Original (cropped, working scale)")


def plot_binary(ax, res):
    ax.imshow(res.pre.binary, cmap="gray")
    ax.contour(res.pre.skeleton > 0, levels=[0.5], colors="#e4572e", linewidths=0.5)
    _axes(ax, f"2. Binary ({res.pre.method}, {res.pre.polarity}); skeleton in red")


def plot_dots(ax, res):
    ax.imshow(res.pre.binary, cmap="gray", alpha=0.5)
    c = res.dots.centers
    if len(c):
        ax.scatter(c[:, 0], c[:, 1], s=14, c="#e4572e")
    g = res.grid
    if g.ok:
        for r in range(g.rows):
            p0, p1 = g.origin + r * g.a2, g.origin + r * g.a2 + (g.cols - 1) * g.a1
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="#17bebb", lw=0.7)
        for k in range(g.cols):
            p0, p1 = g.origin + k * g.a1, g.origin + k * g.a1 + (g.rows - 1) * g.a2
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="#17bebb", lw=0.7)
        _axes(ax, f"3. Dots: {len(c)}  lattice {g.rows}x{g.cols} ({g.lattice_type}, {g.orientation_deg:.1f} deg)")
    else:
        _axes(ax, f"3. Dots: {len(c)} (lattice not recovered)")
    ax.set_xlim(0, res.pre.binary.shape[1]); ax.set_ylim(res.pre.binary.shape[0], 0)


def plot_structure(ax, res):
    ax.imshow(res.pre.binary, cmap="gray", alpha=0.45)
    G = res.topology["graph"]
    ax.imshow(np.ma.masked_where(res.topology["skeleton"] == 0, res.topology["skeleton"]), cmap="autumn", alpha=0.9)
    if res.node_states is not None:
        for n, s in res.model.states.items():
            x, y = res.node_states.node_px[n]
            unk = res.node_states.confidence[n] == 0.0
            ax.scatter([x], [y], s=34, facecolors="none" if unk else STATE_COLORS[s],
                       edgecolors="k" if unk else STATE_COLORS[s], linewidths=1.0, zorder=5)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f"gate {s}") for s, c in STATE_COLORS.items()],
                  loc="lower right", fontsize=7, framealpha=0.8)
    t = res.topology
    _axes(ax, f"4. Structure: {t['crossings_deg4']} crossings, {t['end_points']} end points, "
              f"{t['cyclomatic_number']} cycles")
    ax.set_xlim(0, res.pre.binary.shape[1]); ax.set_ylim(res.pre.binary.shape[0], 0)


def plot_symmetry(ax, res):
    s = res.symmetry
    ax.imshow(res.pre.binary, cmap="gray", alpha=0.35)
    cx, cy = s.center
    L = max(res.pre.binary.shape) * 0.55
    for a in s.mirror_axes:
        t = np.radians(a["angle_deg"])
        ax.plot([cx - L * np.cos(t), cx + L * np.cos(t)], [cy - L * np.sin(t), cy + L * np.sin(t)],
                c="#e4572e", lw=1.2, ls="--")
    ax.scatter([cx], [cy], c="#17bebb", s=40, zorder=5)
    ax.set_xlim(0, res.pre.binary.shape[1]); ax.set_ylim(res.pre.binary.shape[0], 0)
    rot = ", ".join(f"{n}:{v:.2f}" for n, v in s.rotation_scores.items() if n in (2, 3, 4, 6))
    _axes(ax, f"5. Symmetry: strict {s.group_strict} / approx {s.group}\nrotation scores {rot}")


def plot_mirror_scan(ax, res):
    s = res.symmetry
    ax.plot(s.mirror_angles, s.mirror_scores, c="#17bebb")
    ax.axhline(s.baseline_mirror, c="grey", ls=":", label="chance level")
    ax.axhline(s.threshold, c="#e4572e", ls="--", label="approx threshold")
    ax.axhline(s.strict_threshold, c="#ffc914", ls="--", label="strict threshold")
    ax.set_xlabel("mirror axis angle (deg)"); ax.set_ylabel("overlap score"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Mirror-axis scan", fontsize=10)


def plot_reconstruction(ax, res):
    if res.model is None:
        ax.text(0.5, 0.5, "no structural model\n(dot lattice not recovered)", ha="center", va="center")
        _axes(ax, "6. Reconstructed kolam"); return
    ax.imshow(_rgb(render_model(res.model, u=24, color_by_loop=True)))
    _axes(ax, f"6. Reconstructed from rules ({res.model.n_loops()} loop(s))")


def plot_generated(ax, gk, colour=True):
    ax.imshow(_rgb(render_model(gk.model, u=24, color_by_loop=colour)))
    _axes(ax, f"7. New kolam: {gk.n_loops} loop(s), group {gk.group}")


def summary_figure(res, generated=None):
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    plot_original(axs[0, 0], res); plot_binary(axs[0, 1], res); plot_dots(axs[0, 2], res); plot_structure(axs[0, 3], res)
    plot_symmetry(axs[1, 0], res); plot_reconstruction(axs[1, 1], res)
    if generated is not None:
        plot_generated(axs[1, 2], generated)
    else:
        axs[1, 2].axis("off")
    plot_mirror_scan(axs[1, 3], res)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    return buf.getvalue()
