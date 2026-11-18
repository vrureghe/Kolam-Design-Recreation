"""Create sample kolam images (SYNTHETIC, produced by this project's generator).
Run from the project root:  python samples/make_samples.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kolam.generator import generate_kolam
from kolam.render import render_model
from kolam.synth import perturb
from kolam.io_utils import save_image

D, L = ((235, 235, 235), (34, 28, 24)), ((30, 30, 30), (245, 245, 245))
SPECS = [("sample_01_square5_rot4", "square", 5, "rot4", 3, D, {}),
         ("sample_02_diamond7_rot2", "diamond", 7, "rot2", 1, D, {}),
         ("sample_03_rect4x6_mirror_ink", "rect", 4, "mirror_vh", 2, L, dict(noise=5)),
         ("sample_04_cross7_D4_rotated", "cross", 7, "D4", 1, D, dict(angle=12, noise=8, blur=0.8)),
         ("sample_05_octagon7_rot2_warped", "octagon", 7, "rot2", 4, L, dict(warp=1.5, blur=0.8, shade=0.3, noise=6)),
         ("sample_06_square7_rot4_oneStroke", "square", 7, "rot4", 2, D, {})]
here = os.path.dirname(__file__)
for name, shape, n, grp, seed, (fg, bg), kw in SPECS:
    g = generate_kolam(shape, n, grp, seed=seed)
    save_image(os.path.join(here, name + ".png"), perturb(render_model(g.model, 28, fg, bg), bg, seed=seed, **kw))
    print(name, g.summary())
