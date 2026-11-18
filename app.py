"""Streamlit interface.  Run:  streamlit run app.py"""
import glob, os
import matplotlib.pyplot as plt
import streamlit as st
from kolam.io_utils import load_image, encode_png
from kolam.pipeline import analyze_image, generate_from_analysis
from kolam.generator import generate_kolam, generate_repeated, mutate, MASK_SHAPES
from kolam.groups import GROUP_NAMES, GROUP_DESCRIPTIONS, compatible
from kolam.generator import make_mask
from kolam.render import render_model, THEMES
from kolam.evaluation import evaluate_model
from kolam.representation import save_json
import kolam.visualize as V
import json

st.set_page_config(page_title="Kolam Analyzer & Generator", layout="wide")
st.title("Kolam design analysis and recreation")
tab_an, tab_gen, tab_about = st.tabs(["Analyze a kolam", "Generate a kolam", "About / limits"])
theme_name = st.sidebar.selectbox("Colour theme", list(THEMES))
colour_loops = st.sidebar.checkbox("Colour each loop separately", True)
th = THEMES[theme_name]

def draw(gk, key):
    img = render_model(gk.model, u=30, color_by_loop=colour_loops, **th)
    st.image(img[:, :, ::-1], caption=gk.summary())
    st.download_button("Download PNG", encode_png(img), f"kolam_{key}.png", "image/png", key="dl_" + key)

with tab_an:
    samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*.png")))
    up = st.file_uploader("Upload a kolam image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    pick = st.selectbox("...or choose a bundled (synthetic) sample", ["-"] + [os.path.basename(s) for s in samples])
    img = None
    if up is not None:
        img = load_image(up, up.name)
    elif pick != "-":
        img = load_image(samples[[os.path.basename(s) for s in samples].index(pick)])
    if img is not None:
        st.image(img[:, :, ::-1], caption="Uploaded image", width=360)
        thr = st.slider("Approximate symmetry threshold", 0.80, 0.99, 0.93, 0.01)
        try:
            res = analyze_image(img, thr)
        except Exception as e:
            st.error(f"Analysis failed: {e}"); st.stop()
        for w in res.warnings:
            st.warning(w)
        fig = V.summary_figure(res)
        st.pyplot(fig)
        c1, c2 = st.columns(2)
        c1.subheader("Detected (measured)"); c1.json(res.detected, expanded=False)
        c2.subheader("Inferred (model-based)"); c2.json(res.inferred, expanded=False)
        st.download_button("Download analysis figure (PNG)", V.fig_to_png(fig), "analysis.png", "image/png")
        if res.descriptor:
            st.download_button("Download rule descriptor (JSON)", json.dumps(res.descriptor, indent=1), "descriptor.json")
            st.subheader("New kolam from the identified rules")
            seed = st.number_input("Seed", 0, 10**6, 1)
            if st.button("Generate"):
                gk = generate_from_analysis(res, int(seed))
                draw(gk, "from_analysis")
                st.json(evaluate_model(gk.model, gk.group), expanded=False)
        plt.close(fig)

with tab_gen:
    c = st.columns(4)
    mode = c[0].selectbox("Method", ["Symmetric rule search", "Motif repetition"])
    seed = c[3].number_input("Seed", 0, 10**6, 1, key="gseed")
    if mode == "Symmetric rule search":
        shape = c[1].selectbox("Dot arrangement", MASK_SHAPES, index=0)
        size = c[2].slider("Size (dots)", 3, 11, 5)
        group = st.selectbox("Symmetry", GROUP_NAMES, index=5, format_func=lambda g: f"{g} - {GROUP_DESCRIPTIONS[g]}")
        target = st.slider("Target number of loops (1 = one-stroke)", 1, 6, 1)
        mask = make_mask(shape, size, size + 2 if shape == "rect" else None)
        if not compatible(mask, group):
            st.info(f"'{group}' does not fit a {shape} {mask.shape} dot arrangement; choose another symmetry.")
        elif st.button("Generate", key="g1"):
            gk = generate_kolam(shape, size, group, target, int(seed), cols=size + 2 if shape == "rect" else None)
            st.session_state["gk"] = gk
    else:
        motif = c[1].slider("Motif size (dots)", 2, 4, 3); reps = c[2].slider("Tiles per side", 2, 4, 2)
        if st.button("Generate", key="g2"):
            st.session_state["gk"] = generate_repeated(motif, (reps, reps), "none", 1, int(seed))
    gk = st.session_state.get("gk")
    if gk is not None:
        draw(gk, "gen")
        if not gk.achieved:
            st.caption("Target loop count not reached for this symmetry (some symmetries cannot give one stroke).")
        if st.button("Make a variation (keeps symmetry, never adds loops)"):
            st.session_state["gk"] = mutate(gk, 2, int(seed) + 1); st.rerun()
        st.json(evaluate_model(gk.model, gk.group), expanded=False)

with tab_about:
    st.markdown(open(os.path.join(os.path.dirname(__file__), "docs", "LIMITS.md")).read())
