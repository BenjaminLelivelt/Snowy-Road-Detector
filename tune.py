"""Pick thresholds on a TRAINING split (never on the test images).

    python tune.py --images data/ISTD/train/train_A --masks data/ISTD/train/train_B

1. Estimates SHADOW_DIR: the median direction in which log-chromaticity moves
   across ground-truth shadow borders (sunlit -> shadowed side).
2. Grid-searches each method's thresholds to maximise shadow-edge F1.
Writes results/params.json, which run.py / eval.py load automatically.
This is a handful of hand-set thresholds, not a learned model.
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np

from eval import EXTS, find_mask
from shadow import classify as C
from shadow.cues import color_cue
from shadow.edges import detect_edges
from shadow.evaluate import counts, gt_shadow_band, metrics


def estimate_shadow_dir(samples):
    vecs = []
    for img, band in samples:
        e, _, nx, ny = detect_edges(img)
        ys, xs = np.nonzero(e)
        lr = color_cue(img, ys, xs, nx, ny)["lr"]
        d = np.stack([lr[:, 2] - lr[:, 1], lr[:, 0] - lr[:, 1]], 1)[band[ys, xs]]
        n = np.linalg.norm(d, axis=1)
        vecs.append(d[n > 0.05] / n[n > 0.05, None])
    v = np.median(np.concatenate(vecs), axis=0)
    return v / np.linalg.norm(v)


def score(cache, method, p):
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for d, band in cache:
        for k, v in counts(C.label_map(d, method, p), band).items():
            tot[k] += v
    return metrics(tot)


GRID = dict(
    color=dict(DARK_T=[0.5, 0.6, 0.7, 0.8, 0.9], CHROMA_T=[0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 1.0]),
    width=dict(DARK_T=[0.5, 0.6, 0.7, 0.8, 0.9], WIDTH_T=[0, 1, 2, 3, 4, 5, 6, 8]),
    blue=dict(DARK_T=[0.5, 0.6, 0.7, 0.8, 0.9], BLUE_T=[0.0, 0.02, 0.05, 0.08, 0.12, 0.16]),
    msblur=dict(DARK_T=[0.5, 0.6, 0.7, 0.8, 0.9], BLUR_T=[0.2, 0.4, 0.6, 0.8, 1.0, 1.3]),
    full=dict(DARK_T=[0.6, 0.7, 0.8, 0.85, 0.9], BLUE_T=[0.0, 0.02, 0.05, 0.08, 0.12],
              BLUR_T=[0.0, 0.2, 0.4, 0.6, 0.8]),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="results/params.json")
    args = ap.parse_args()

    samples = []
    for f in sorted(os.listdir(args.images)):
        if not f.lower().endswith(EXTS):
            continue
        mp = find_mask(args.masks, os.path.splitext(f)[0])
        if mp:
            img = C.resize_to_work(cv2.imread(os.path.join(args.images, f)))
            mask = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), img.shape[1::-1],
                              interpolation=cv2.INTER_NEAREST)
            samples.append((img, gt_shadow_band(mask)))
    print(len(samples), "training images")

    C.SHADOW_DIR[:] = estimate_shadow_dir(samples)
    print("SHADOW_DIR", C.SHADOW_DIR.round(3))
    cache = [(C.compute(img), band) for img, band in samples]

    best = {"SHADOW_DIR": C.SHADOW_DIR.tolist()}
    per_method = {}
    for method, grid in GRID.items():
        keys = list(grid)
        top = None
        for vals in itertools.product(*grid.values()):
            p = dict(C.PARAMS, **dict(zip(keys, vals)))
            r = score(cache, method, p)
            if top is None or r["f1"] > top[0]["f1"]:
                top = (r, dict(zip(keys, vals)))
        per_method[method] = dict(top[1], f1=round(top[0]["f1"], 3))
        print(f"{method:8s} best F1={top[0]['f1']:.3f}  {top[1]}")

    # a "darker only" reference: no shape/colour cue at all
    top = max(((score(cache, "color", dict(C.PARAMS, DARK_T=t, CHROMA_T=9.0)), t)
               for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), key=lambda z: z[0]["f1"])
    print(f"dark-only reference F1={top[0]['f1']:.3f}  DARK_T={top[1]}")

    # the shipped thresholds are those of the full method
    best.update({k: v for k, v in per_method["full"].items() if k != "f1"})
    best["CHROMA_T"] = per_method["color"]["CHROMA_T"]
    best["WIDTH_T"] = per_method["width"]["WIDTH_T"]
    p = dict(C.PARAMS, **{k: v for k, v in best.items() if k in C.PARAMS})
    top = None
    for vt in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = score(cache, "full_vote", dict(p, VOTE_T=vt))
        if top is None or r["f1"] > top[0]["f1"]:
            top = (r, vt)
    best["VOTE_T"] = top[1]
    per_method["full_vote"] = dict(VOTE_T=top[1], f1=round(top[0]["f1"], 3))
    print(f"full_vote best F1={top[0]['f1']:.3f}  VOTE_T={top[1]}")

    best["per_method"] = per_method
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(best, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
