"""Label edges in one image (or every image in a folder) and save overlays.

    python run.py photo.jpg
    python run.py data/own/ --method full_vote --out results/own

Output: <name>_<method>.png  (green = object edge, red = shadow edge)
        <name>_<method>_cleaned.png  (edge map with shadow edges removed)
"""
import argparse
import os

import cv2

from shadow.classify import METHODS, OBJECT, compute, label_map, load_params, overlay

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--method", default="all", choices=METHODS + ["all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--params", default="results/params.json")
    args = ap.parse_args()
    load_params(args.params)

    if os.path.isdir(args.input):
        files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                       if f.lower().endswith(EXTS))
    else:
        files = [args.input]
    methods = METHODS if args.method == "all" else [args.method]
    os.makedirs(args.out, exist_ok=True)

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print("skip (can't read):", path)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        d = compute(img)
        for m in methods:
            lab = label_map(d, m)
            cv2.imwrite(os.path.join(args.out, f"{name}_{m}.png"), overlay(d["img"], lab))
            cv2.imwrite(os.path.join(args.out, f"{name}_{m}_cleaned.png"), (lab == OBJECT).astype("uint8") * 255)
        print("done:", path)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
