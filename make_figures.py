"""Figures for the report / webpage -> results/figures/

    python make_figures.py

Needs ISTD (data/ISTD), the snow photos (data/snow_web + data/snow_masks_draft),
results/params.json and the two eval CSVs. matplotlib is only needed here.
"""
import csv
import glob
import os

import cv2
import numpy as np

from shadow import classify as C
from shadow.evaluate import counts, gt_shadow_band, hough_lines, metrics

OUT = "results/figures"
ISTD_A, ISTD_B = "data/ISTD/test/test_A", "data/ISTD/test/test_B"
LABELS = {"canny": "Canny (keep all)", "color": "Color-only (baseline)", "width": "Width (proposal)",
          "blue": "Blue-shift color", "msblur": "Multi-scale blur", "full": "Full (pixel)",
          "full_vote": "Full + contour vote"}


def tile(img, text, w=320):
    h = round(img.shape[0] * w / img.shape[1])
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if t.ndim == 2:
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(t, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(t, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def gt_overlay(img, mask):
    band = gt_shadow_band(mask, 1)
    out = (img * 0.4).astype(np.uint8)
    out[band] = (0, 0, 255)
    return out


def per_image_f1(files, img_dir, mask_dir, method="full_vote"):
    res = []
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        m = cv2.imread(os.path.join(mask_dir, os.path.splitext(f)[0] + ".png"), 0)
        d = C.compute(img)
        m = cv2.resize(m, d["edges"].shape[::-1], interpolation=cv2.INTER_NEAREST)
        band = gt_shadow_band(m)
        res.append((metrics(counts(C.label_map(d, method), band))["f1"], f, int(band.sum())))
    return res


def qual_grid(names, img_dir, mask_dir, methods, path, w=300):
    rows = []
    for f in names:
        img = cv2.imread(os.path.join(img_dir, f))
        d = C.compute(img)
        cols = [tile(d["img"], f"input {os.path.splitext(f)[0]}", w)]
        mp = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
        if os.path.exists(mp):
            m = cv2.resize(cv2.imread(mp, 0), d["edges"].shape[::-1], interpolation=cv2.INTER_NEAREST)
            cols.append(tile(gt_overlay(d["img"], m), "GT shadow border" if "ISTD" in mask_dir else "draft label border", w))
        for m_ in methods:
            cols.append(tile(C.overlay(d["img"], C.label_map(d, m_), thick=True), LABELS[m_], w))
        hmax = max(c.shape[0] for c in cols)
        cols = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cols]
        rows.append(cv2.copyMakeBorder(np.hstack(cols), 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(path, np.vstack(rows))
    print("saved", path)


def hough_fig(f, path):
    img = cv2.imread(os.path.join(ISTD_A, f))
    d = C.compute(img)
    band = gt_shadow_band(cv2.imread(os.path.join(ISTD_B, f), 0))
    cols = []
    for m_, title in (("canny", "Hough on raw Canny"), ("full_vote", "Hough after removing shadow edges")):
        lab = C.label_map(d, m_)
        lines = hough_lines(lab == C.OBJECT)
        vis = (d["img"] * 0.5).astype(np.uint8)
        vis[lab == C.OBJECT] = (90, 90, 90)
        n_sh = 0
        for x1, y1, x2, y2 in lines:
            canvas = np.zeros(band.shape, np.uint8)
            cv2.line(canvas, (x1, y1), (x2, y2), 1, 1)
            on_shadow = (band & (canvas > 0)).sum() > 0.5 * (canvas > 0).sum()
            n_sh += on_shadow
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255) if on_shadow else (0, 255, 0), 2)
        cols.append(tile(vis, f"{title}: {len(lines)} lines, {n_sh} on shadows (red)", 480))
    cv2.imwrite(path, np.hstack(cols))
    print("saved", path)


def cue_hist(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = {k: [] for k in ("chroma", "blue", "width", "blur", "dark")}
    ys = []
    for f in sorted(glob.glob("data/ISTD/train/train_A/*.png")):
        img = cv2.imread(f)
        d = C.compute(img)
        band = gt_shadow_band(cv2.imread(f.replace("train_A", "train_B"), 0))
        ys.append(band[d["ys"], d["xs"]])
        for k in feats:
            feats[k].append(d[k])
    y = np.concatenate(ys)
    titles = dict(dark="brightness ratio dark/bright side", chroma="chromaticity difference\n(baseline: small = shadow)",
                  blue="shift along sun→sky direction\n(ours: large = shadow)", width="10–90% transition width (px)\n(proposal sharpness cue)",
                  blur="multi-scale blur estimate (px)\n(ours: large = shadow)")
    ranges = dict(dark=(0, 1.2), chroma=(0, 0.4), blue=(-0.3, 0.6), width=(0, 16), blur=(0, 3))
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.2))
    for ax, k in zip(axs, ("dark", "chroma", "blue", "width", "blur")):
        v = np.concatenate(feats[k])
        bins = np.linspace(*ranges[k], 41)
        ax.hist(np.clip(v[~y], *ranges[k]), bins, density=True, alpha=0.55, color="#2a7ab0", label="non-shadow edge px")
        ax.hist(np.clip(v[y], *ranges[k]), bins, density=True, alpha=0.55, color="#d1495b", label="shadow-border edge px")
        x = np.concatenate([v[y], v[~y]])
        r = x.argsort().argsort() + 1.0
        auc = (r[: y.sum()].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
        ax.set_title(f"{titles[k]}\nAUC = {max(auc, 1 - auc):.2f}", fontsize=9)
        ax.set_yticks([])
    axs[0].legend(fontsize=8)
    fig.suptitle("Per-pixel cue distributions on the ISTD training subset (133 images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("saved", path)


def bar_fig(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [("results/eval_istd_test.csv", "ISTD test (540 img, real GT)"),
            ("results/eval_snow_draft.csv", "Snow photos (30 img, DRAFT labels)")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, (p, title) in zip(axs, sets):
        rows = [r for r in csv.DictReader(open(p)) if r["method"] != "canny"]
        names = [LABELS[r["method"]] for r in rows]
        f1 = [float(r["f1"]) for r in rows]
        orm = [float(r["object_removed"]) for r in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, f1, 0.4, label="shadow-edge F1 (higher better)", color="#d1495b")
        ax.bar(x + 0.2, orm, 0.4, label="object edges wrongly removed (lower better)", color="#7a7a7a")
        for i, v in enumerate(f1):
            ax.text(i - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 0.95)
        ax.set_title(title, fontsize=10)
    axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("saved", path)


def main():
    C.load_params()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(ISTD_A))
    scores = sorted(per_image_f1(files, ISTD_A, ISTD_B), reverse=True)
    scores = [s for s in scores if s[2] > 1500]          # images with a real shadow border
    pick = [scores[len(scores) // 10][1], scores[len(scores) // 4][1], scores[len(scores) // 2][1],
            scores[-len(scores) // 10][1]]               # good, good, median, failure
    with open(os.path.join(OUT, "istd_pick.txt"), "w") as fh:
        for s in scores:
            if s[1] in pick:
                fh.write(f"{s[1]} F1={s[0]:.3f}\n")
    qual_grid(pick, ISTD_A, ISTD_B, ["canny", "color", "full_vote"], os.path.join(OUT, "istd_qualitative.jpg"))
    qual_grid(["snow_07.jpg", "snow_10.jpg", "snow_18.jpg", "snow_23.jpg", "snow_24.jpg", "snow_03.jpg"],
              "data/snow_web", "data/snow_masks_draft", ["color", "full_vote"], os.path.join(OUT, "snow_qualitative.jpg"))
    hough_fig(pick[0], os.path.join(OUT, "hough_demo.jpg"))
    cue_hist(os.path.join(OUT, "cue_histograms.png"))
    bar_fig(os.path.join(OUT, "f1_bars.png"))


if __name__ == "__main__":
    main()
