"""Command-line use.

  python run_pipeline.py --image samples/sample_01_square5_rot4.png --out outputs --generate
  python run_pipeline.py --generate-only --shape diamond --size 7 --group rot2 --seed 3 --out outputs
"""
import argparse, os
from kolam.io_utils import load_image, save_image
from kolam.pipeline import analyze_image, generate_from_analysis
from kolam.generator import generate_kolam, MASK_SHAPES
from kolam.groups import GROUP_NAMES
from kolam.render import render_model
from kolam.representation import save_json
from kolam import visualize as V

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--generate", action="store_true", help="also generate a new kolam from the identified rules")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--shape", default="square", choices=MASK_SHAPES); ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--group", default="rot4", choices=GROUP_NAMES); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.generate_only or not a.image:
        g = generate_kolam(a.shape, a.size, a.group, seed=a.seed)
        print(g.summary())
        print("saved", save_image(os.path.join(a.out, f"generated_{a.shape}{a.size}_{a.group}_s{a.seed}.png"), render_model(g.model)))
        return
    res = analyze_image(load_image(a.image))
    print(res.report_text())
    gk = generate_from_analysis(res, seed=a.seed) if a.generate and res.model is not None else None
    if gk:
        print("\nGENERATED:", gk.summary())
        save_image(os.path.join(a.out, "generated.png"), render_model(gk.model))
    stem = os.path.splitext(os.path.basename(a.image))[0]
    fig = V.summary_figure(res, gk)
    open(os.path.join(a.out, f"{stem}_analysis.png"), "wb").write(V.fig_to_png(fig))
    if res.descriptor: save_json(res.descriptor, os.path.join(a.out, f"{stem}_descriptor.json"))
    print("outputs written to", a.out)

if __name__ == "__main__":
    main()
