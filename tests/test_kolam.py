import numpy as np, pytest
from kolam.generator import generate_kolam, generate_repeated, mutate, make_mask
from kolam.render import render_model
from kolam.model import KolamModel
from kolam.pipeline import analyze_image, generate_from_analysis
from kolam.align import compare_models
from kolam.evaluation import evaluate_model, _group_score
from kolam.synth import perturb

def test_single_dot_is_one_loop():
    assert KolamModel(np.ones((1, 1), bool)).n_loops() == 1

def test_loops_cover_all_segments():
    g = generate_kolam("square", 5, "rot4", seed=1)
    assert sum(len(l) for l in g.model.loops()) == g.model.n_segments

@pytest.mark.parametrize("shape,size,grp", [("square", 5, "rot4"), ("diamond", 7, "rot2"), ("rect", 4, "mirror_vh"), ("cross", 7, "D4")])
def test_generated_is_symmetric(shape, size, grp):
    g = generate_kolam(shape, size, grp, seed=2)
    assert _group_score(g.model, grp) == 1.0

def test_rot4_reaches_one_stroke():
    assert all(generate_kolam("square", 5, "rot4", seed=s).n_loops == 1 for s in range(5))

def test_incompatible_group_rejected():
    with pytest.raises(ValueError):
        generate_kolam("rect", 4, "D4")

def test_repeat_and_mutate():
    g = generate_repeated(3, (2, 2), "none", seed=1)
    m = mutate(g, 2, seed=3)
    assert m.n_loops <= g.n_loops

@pytest.mark.parametrize("kw", [dict(), dict(angle=20, noise=10), dict(warp=1.5, blur=0.8, shade=0.3, noise=6)])
def test_roundtrip(kw):
    g = generate_kolam("diamond", 7, "rot2", seed=1)
    fg, bg = (30, 30, 30), (245, 245, 245)
    res = analyze_image(perturb(render_model(g.model, 28, fg, bg), bg, **kw))
    cmp = compare_models(g.model, res.model)
    assert cmp["aligned"] and cmp["accuracy"] >= 0.97
    assert res.model.n_loops() == g.n_loops

def test_generate_from_analysis():
    g = generate_kolam("square", 5, "rot4", seed=3)
    res = analyze_image(render_model(g.model))
    new = generate_from_analysis(res, seed=9)
    assert (new.model.mask == g.model.mask).all() and new.n_loops == 1

def test_blank_image_errors():
    with pytest.raises(ValueError):
        analyze_image(np.full((300, 300, 3), 200, np.uint8))
