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

**ISTD** (images + shadow masks). The official copy is on Google Drive (ST-CGAN repo,
https://github.com/DeepInsight-PCALab/ST-CGAN). We used an unofficial mirror of the
same files on Hugging Face (`CK234/tmp_ISTD`): the full test set (540 images,
`test_A` + `test_B`) and every 10th training image (133) for tuning. Put them in
`data/ISTD/test/{test_A,test_B}` and `data/ISTD/train/{train_A,train_B}` (gitignored).

```bash
python tune.py --images data/ISTD/train/train_A --masks data/ISTD/train/train_B   # -> results/params.json
python eval.py --images data/ISTD/test/test_A --masks data/ISTD/test/test_B --out results/eval_istd_test.csv
```

**Snow photos**: 30 CC-licensed snow scenes from Wikimedia Commons in `data/snow_web/`
(credits and licenses in `data/CREDITS.md`) and 3 dry-road scenes in `data/dry_web/`.
`tools/draft_snow_masks.py` makes **draft** shadow masks (`data/snow_masks_draft/`);
they still need hand correction with `tools/label_masks.py` (saves to `data/snow_masks/`).

```bash
python eval.py --images data/snow_web --masks data/snow_masks_draft --out results/eval_snow_draft.csv
python run.py data/snow_web --method full_vote --out results/snow
python make_figures.py            # results/figures/*  (needs matplotlib)
```

**Our own photos**: put them in `data/own/`, then `python run.py data/own/`.

All images are processed at 640 px on the long side (`WORK_SIDE`), ISTD's resolution,
so pixel-width thresholds mean the same thing on every image.

## Methods (`--method`)

| name        | what it does                                                         |
|-------------|----------------------------------------------------------------------|
| `canny`     | baseline: keep every edge                                            |
| `color`     | color-only baseline: darker + same chromaticity on both sides        |
| `width`     | proposal sharpness cue: 10–90% transition wider than `WIDTH_T`       |
| `blue`      | darker + chromaticity shifts along the sun→skylight direction        |
| `msblur`    | darker + multi-scale blur estimate above `BLUR_T` (soft penumbra)    |
| `full`      | `blue` AND `msblur`, per pixel                                       |
| `full_vote` | `full` + majority vote per contour (contours cut at junctions)       |

Each method uses its own thresholds tuned on the training split (`results/params.json`).

## Code

- `shadow/edges.py` – Gaussian blur + Canny + gradient direction
- `shadow/cues.py` – cross-edge color (chromaticity, log ratios), transition width,
  multi-scale blur estimate, texture contrast
- `shadow/classify.py` – thresholds, combining cues, junction-split contour voting
- `shadow/evaluate.py` – GT shadow-border band, P/R/F1, Hough demo
- `tune.py` – shadow direction + threshold grid search on the training split
- `make_figures.py` – qualitative grids, Hough demo, cue histograms, bar chart
- `tools/` – draft snow masks, hand-labeling tool

## Results so far (mid-term)

ISTD test, 540 images, pixel-level over Canny edge pixels:

| method | P | R | F1 | object edges wrongly removed |
|---|---|---|---|---|
| color (baseline) | 0.28 | 0.58 | 0.38 | 18.7% |
| width (proposal) | 0.32 | 0.59 | 0.42 | 15.7% |
| blue | 0.62 | 0.65 | 0.64 | 5.0% |
| msblur | 0.52 | 0.60 | 0.56 | 7.1% |
| full | 0.62 | 0.69 | 0.65 | 5.4% |
| full_vote | 0.63 | 0.69 | **0.66** | 5.0% |

Snow photos (draft labels, not final): color 0.24 → full 0.45 F1.

## TODO (from the proposal timeline)

- Hand-correct the 30 snow masks; take our own ~30 campus photos once it snows
- Snow: Canny fires on snow grain texture; try scale-adaptive Canny thresholds
- Hough grouping of straight shadow edges; better use of contour voting
- Failure cases: gray/overcast shadows (no blue shift), dark objects on snow
