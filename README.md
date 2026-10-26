# Shadow or Object?

CS 566 (Fall 2026) project: detect edges in a photo and label each one as a
**shadow edge** or an **object edge**, training-free.

## Setup (macOS)

```bash
cd snowy-road-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check (no dataset needed)

```bash
python make_synthetic.py
python eval.py --images data/synthetic/images --masks data/synthetic/masks
python run.py data/synthetic/images/syn_1.png
```

`run.py` writes overlays to `results/`: green = object edge, red = shadow edge.

## Real data

**ISTD** (images + shadow masks): download from the ST-CGAN repo
(https://github.com/DeepInsight-PCALab/ST-CGAN) and unzip into `data/ISTD/`, then:

```bash
python eval.py --images data/ISTD/test/test_A --masks data/ISTD/test/test_B --limit 30
```

**Our own photos**: put them in `data/own/`, then `python run.py data/own/`.
For hand labels, paint a shadow mask (white = shadow) with the same file name
in `data/own_masks/` and use `eval.py` the same way.

## Methods (`--method`)

| name        | what it does                                              |
|-------------|-----------------------------------------------------------|
| `canny`     | baseline: keep every edge                                 |
| `color`     | color-only: same chromaticity on both sides -> shadow     |
| `sharp`     | sharpness-only: wide (soft) transition -> shadow          |
| `full`      | both cues must agree, per pixel                           |
| `full_vote` | `full` + majority vote over each connected contour        |

## Code

- `shadow/edges.py` – Gaussian blur + Canny + gradient direction
- `shadow/cues.py` – cross-edge color check, transition width
- `shadow/classify.py` – thresholds, combining cues, contour voting
- `shadow/evaluate.py` – GT shadow-border band, P/R/F1, Hough demo

Thresholds are at the top of `shadow/classify.py` (`CHROMA_T`, `WIDTH_T`, `DARK_T`).

## Evaluation output

`eval.py` prints and saves `results/eval.csv`:
precision / recall / F1 for shadow edges (over Canny edge pixels), % shadow
edges removed, % object edges wrongly removed, and the number of Hough lines on
the cleaned edge map (total and how many lie on a shadow border).

## TODO (from the proposal timeline)

- Tune thresholds on ISTD; hand-label ~30 campus photos (incl. snow)
- Multi-scale sharpness check, Hough grouping of straight shadow edges
- Failure cases: gray/white surfaces, blue-tinted snow shadows
