# Shadow or Object?

CS 566 (Fall 2026) project: detect the edges in a photo and label each one as a
**shadow edge** or an **object edge**, without any training, so shadow edges can be
removed before later steps such as line fitting. Snow scenes (low sun, long blue
shadows on white ground) are our stress test.

**Mid-term status (Oct 29):** the full pipeline runs on real data. On the ISTD test
set (540 photos with ground-truth shadow masks), shadow-edge **F1 is 0.66**, against
**0.38** for the color-only baseline. Only **5%** of real object edges are wrongly
removed (baseline: 19%). Hough lines that lie on shadow borders drop from 1,787 to 158.

![F1 by method](results/figures/f1_bars.png)

---

## 1. What is done

| Proposal item | Status | Where |
|---|---|---|
| Collect public shadow data | Done: ISTD test (540) + every 10th training image (133) | `data/ISTD/` (not in git, see §6) |
| Collect snow photos | 30 CC-licensed snow scenes from Wikimedia Commons (own campus photos once it snows) | `data/snow_web/`, `data/CREDITS.md` |
| Label snow photos | Draft masks generated automatically. **Hand correction still to do** | `tools/draft_snow_masks.py`, `tools/label_masks.py` |
| Baseline: Gaussian + Canny, color-only test | Done | `shadow/edges.py`, method `color` |
| Cue 1: cross-edge color check | Done, **redesigned** after measuring it (see §3) | `color_cue`, method `blue` |
| Cue 2: edge sharpness | Done, **redesigned**: multi-scale blur estimate | `multiscale_blur_cue`, method `msblur` |
| Contour-level voting | Done (contours cut at junctions); small gain so far | `contour_vote` |
| Threshold tuning on a training split | Done, grid search | `tune.py` -> `results/params.json` |
| Quantitative evaluation + ablations | Done on ISTD test; snow on draft labels | `eval.py` -> `results/eval_*.csv` |
| Hough line demo | Done | `make_figures.py` |
| Qualitative figures incl. failure cases | Done | `results/figures/` |

**Still to do:** hand-correct the snow masks and re-report snow numbers; take our own
campus photos; scale-adaptive Canny thresholds (snow grain); better voting (vote on
cue values, group straight shadow edges with Hough); an overcast-shadow fallback
(gray shadows have no blue shift); precision-recall curves.

---

## 2. Pipeline and how it maps to the course

Every step uses material from the lectures so far. No learned model: four thresholds
plus one direction vector, all picked on the training split only.

| Step | Course topic | What we do | Code |
|---|---|---|---|
| 0. Working scale | Image formation & sensing | Resize every photo to 640 px on the long side so pixel widths (penumbra size) mean the same thing on every image | `classify.resize_to_work` |
| 1. Smoothing | Image filtering (convolution, Gaussian) | Gaussian blur, sigma 1.4, to suppress noise | `edges.detect_edges` |
| 2. Edges | Edge detection (gradients, Sobel, Canny) | Canny (40/100) for thin, linked edges; Sobel gives the gradient direction at each edge pixel, which tells us which side is dark and which is bright | `edges.detect_edges` |
| 3. Brightness test | Image formation: intensity = reflectance x illumination | A shadow only lowers illumination, so the dark side must be clearly darker (ratio < 0.8) | `cues.color_cue` |
| 4. Color test | Image formation, color / chromaticity | Shadowed ground is lit only by blue skylight, so across a shadow edge the log-chromaticity moves along one fixed sun->sky direction (Finlayson et al.). We test for that shift | `cues.color_cue`, `classify.SHADOW_DIR` |
| 5. Sharpness test | Filtering at several scales, Fourier view (blur = low-pass) | A blurred step loses gradient strength more slowly when smoothed again. From the gradient at sigma 1 and sigma 3 we solve for the edge's own blur width; soft penumbras give a large width | `cues.multiscale_blur_cue` |
| 6. Contour voting | Boundary detection, linking edges into contours | Link edge pixels into 8-connected curves, cut them at junctions, and let each curve take a majority vote | `classify.contour_vote` |
| 7. Downstream demo | Hough transform (HW2) | Line detection on the raw vs. cleaned edge map; count lines on shadow borders | `evaluate.hough_lines` |

The blur formula: a step blurred by width s, smoothed by a Gaussian of std sigma, has
peak gradient proportional to 1/sqrt(s^2 + sigma^2). With r = G(sigma2)/G(sigma1):

    s = sqrt((r^2 * sigma2^2 - sigma1^2) / (1 - r^2))

---

## 3. Results

### Why we changed the proposal's cues

Each cue measured on 2.75 M training edge pixels (red = shadow border, blue = other edges):

![Cue distributions](results/figures/cue_histograms.png)

- **"Same chromaticity = shadow" is backwards outdoors.** Shadow borders show *larger*
  chromaticity changes than other edges, because the shadow side turns bluer. Testing
  for the blue shift instead gives AUC 0.81.
- **The 10-90% transition width barely separates the classes (AUC 0.55).** The
  multi-scale blur estimate is the best single cue (AUC 0.85).

### ISTD test set (540 images, real ground truth)

| Method | Precision | Recall | F1 | Object edges wrongly removed | Hough lines | Lines on shadow borders |
|---|---|---|---|---|---|---|
| `canny` (keep all) | - | - | - | 0% | 22,611 | 1,787 |
| `color` (baseline) | 0.28 | 0.58 | 0.38 | 18.7% | 12,597 | 513 |
| `width` (proposal cue) | 0.32 | 0.59 | 0.42 | 15.7% | 13,247 | 417 |
| `blue` | 0.62 | 0.65 | 0.64 | 5.0% | 19,548 | 267 |
| `msblur` | 0.52 | 0.60 | 0.56 | 7.1% | 18,668 | 219 |
| `full` | 0.62 | 0.69 | 0.65 | 5.4% | 19,260 | 158 |
| `full_vote` | 0.63 | 0.69 | **0.66** | 5.0% | 19,324 | 181 |

Scored per Canny edge pixel; a shadow edge = within 3 px of the shadow-mask border.
Each method uses its own best thresholds from the training split.

Green = kept as object edge, red = labeled shadow. Rows: test images at the 90th, 75th
and 50th percentile of per-image F1, then a failure case (a painted mural whose dark
blue regions look like bluish shadows):

![ISTD examples](results/figures/istd_qualitative.jpg)

### Hough demo

Left: lines found on raw Canny edges (red = lines on the shadow border). Right: after
removing our shadow edges.

![Hough demo](results/figures/hough_demo.jpg)

### Snow photos (30 images, DRAFT labels)

Same thresholds as ISTD, no re-tuning. F1: color baseline 0.24, blue 0.44, full 0.45,
full + vote 0.45. These labels are auto-generated drafts, so treat the numbers as
preliminary.

![Snow examples](results/figures/snow_qualitative.jpg)

### Problems we hit

- **OpenCV crash on real photos.** `cv2.remap` refuses maps longer than 32,767 px, and a
  real photo has more edge pixels than that. Fixed by laying the samples out as a 2-D grid.
- **Snow texture.** Canny fires on snow grain and branches, which dilutes precision.
- **Dark clothing on snow.** Where a shadow touches a person, the vote can spread red
  onto the body outline.
- **Faint, wide shadows** often produce no Canny edge at all.
- **Contour voting adds little** (+0.004 F1). Cutting at junctions matters, though:
  without the cut, F1 drops to 0.56 on the training split.

---

## 4. Methods (`--method`)

| name | what it does |
|---|---|
| `canny` | baseline: keep every edge |
| `color` | color-only baseline: darker + same chromaticity on both sides |
| `width` | proposal sharpness cue: 10-90% transition wider than `WIDTH_T` |
| `blue` | darker + chromaticity shifts along the sun->skylight direction |
| `msblur` | darker + multi-scale blur estimate above `BLUR_T` (soft penumbra) |
| `full` | `blue` AND `msblur`, per pixel |
| `full_vote` | `full` + majority vote per contour (contours cut at junctions) |

## 5. Code

- `shadow/edges.py` - Gaussian blur + Canny + gradient direction
- `shadow/cues.py` - cross-edge color, transition width, multi-scale blur, texture contrast
- `shadow/classify.py` - working scale, thresholds, combining cues, contour voting
- `shadow/evaluate.py` - ground-truth border band, P/R/F1, Hough lines
- `tune.py` - shadow direction + threshold grid search on the training split
- `eval.py` - all methods on a folder of images + masks -> CSV
- `run.py` - overlays for any image or folder
- `make_figures.py` - every figure in this README (needs matplotlib)
- `tools/draft_snow_masks.py`, `tools/label_masks.py` - draft snow labels + hand-labeling tool

## 6. Setup and reproduction (macOS)

```bash
cd snowy-road-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib          # only for make_figures.py
```

Quick check, no dataset needed:

```bash
python make_synthetic.py
python eval.py --images data/synthetic/images --masks data/synthetic/masks
python run.py data/synthetic/images/syn_1.png    # overlays in results/
```

**ISTD.** The official copy is on Google Drive (ST-CGAN repo,
https://github.com/DeepInsight-PCALab/ST-CGAN). We used an unofficial mirror with the same
files on Hugging Face (`CK234/tmp_ISTD`). Put the test set in `data/ISTD/test/{test_A,test_B}`
and every 10th training image in `data/ISTD/train/{train_A,train_B}` (gitignored, 370 MB).

Full reproduction (about 1.5 minutes):

```bash
python tune.py --images data/ISTD/train/train_A --masks data/ISTD/train/train_B
python eval.py --images data/ISTD/test/test_A --masks data/ISTD/test/test_B --out results/eval_istd_test.csv
python eval.py --images data/snow_web --masks data/snow_masks_draft --out results/eval_snow_draft.csv
python make_figures.py
```

Hand-correcting the snow labels (saves to `data/snow_masks/`):

```bash
python tools/label_masks.py data/snow_web data/snow_masks_draft data/snow_masks
```

Our own photos: put them in `data/own/`, then `python run.py data/own/ --method full_vote`.
