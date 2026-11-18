# Kolam Design Analysis & Recreation System

Analyses pulli (dot-grid) kolam images, recovers their structure (dot lattice, crossing rules, loops, symmetry)
and generates new symmetric closed-curve kolams from the identified rules.

## Install & run
```bash
pip install -r requirements.txt
streamlit run app.py                 # UI: upload/sample -> analysis -> generate -> download PNG
python run_pipeline.py --image samples/sample_01_square5_rot4.png --generate --out outputs
python run_pipeline.py --generate-only --shape diamond --size 7 --group rot2 --seed 3
python samples/make_samples.py       # regenerate synthetic samples
python evaluate.py 3                 # batch evaluation -> outputs/evaluation.csv
python -m pytest -q tests            # 14 tests
```
## Layout
`kolam/` modules (io_utils, preprocess, dots, topology, symmetry, rules, representation, generator, groups,
evaluation, render, visualize, pipeline), `app.py` UI, `docs/DESIGN.md` architecture/algorithms/roadmap.

## Measured results (synthetic images with known ground truth; `evaluate.py`, 24 images per condition)
| condition | dots exact | lattice+alignment | gate-state accuracy | loop count correct | strict pixel symmetry group |
|---|---|---|---|---|---|
| clean | 100% | 100% | 100% | 100% | 100% |
| rotation + noise | 100% | 100% | 100% | 100% | 100% |
| blur + shading | 100% | 100% | 100% | 100% | 92% |
| hand-drawn-style warp | 100% | 100% | 100% | 100% | 38% |

Lattice-level symmetry group (`discrete_symmetry_group`) was right on 96/96.

## Honest limits
- **No real kolam photographs were tested.** The images above come from this project's own generator, so
  detection of its geometry is partly self-consistent. Expect lower accuracy on chalk/rice-flour photos.
- Square dot lattices only for gate analysis and generation. Hex/triangular are detected, not modelled.
- Pixel-level symmetry is a coarse measure on dense patterns; trust the lattice-level group when the lattice is recovered.
- "Traditional-ness" score is a transparent heuristic checklist, not an art-historical judgement.
