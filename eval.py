"""Evaluate every method on images + shadow masks (e.g. ISTD test_A / test_B).

    python eval.py --images data/ISTD/test/test_A --masks data/ISTD/test/test_B

Prints a table (precision / recall / F1 for shadow edges, % shadow edges removed,
% object edges wrongly removed, false Hough lines caused by shadows) and
saves it to results/eval.csv. Masks must have the same file name as the image
(extension may differ).
"""
import argparse
import csv
import os

import cv2

from shadow.classify import METHODS, OBJECT, compute, label_map
from shadow.evaluate import count_shadow_lines, counts, gt_shadow_band, hough_lines, metrics

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")


def find_mask(mask_dir, name):
    for e in EXTS:
        p = os.path.join(mask_dir, name + e)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N images")
    ap.add_argument("--tol", type=int, default=3, help="GT border tolerance in px")
    ap.add_argument("--out", default="results/eval.csv")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(EXTS))
    if args.limit:
        files = files[:args.limit]

    total = {m: dict(tp=0, fp=0, fn=0, tn=0) for m in METHODS}
    lines = {m: [0, 0] for m in METHODS}  # [all lines, shadow lines]
    used = 0
    for f in files:
        name = os.path.splitext(f)[0]
        mp = find_mask(args.masks, name)
        img = cv2.imread(os.path.join(args.images, f))
        if mp is None or img is None:
            continue
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        band = gt_shadow_band(mask, args.tol)
        d = compute(img)
        for m in METHODS:
            lab = label_map(d, m)
            for k, v in counts(lab, band).items():
                total[m][k] += v
            ls = hough_lines(lab == OBJECT)
            lines[m][0] += len(ls)
            lines[m][1] += count_shadow_lines(ls, band)
        used += 1
        print(f"[{used}] {f}")

    header = ["method", "precision", "recall", "f1", "shadow_removed", "object_removed",
              "hough_lines", "shadow_lines"]
    rows = []
    for m in METHODS:
        r = metrics(total[m])
        rows.append([m] + [round(r[k], 3) for k in header[1:6]] + lines[m])

    print(f"\n{used} images")
    print("".join(h.ljust(15) for h in header))
    for row in rows:
        print("".join(str(x).ljust(15) for x in row))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
