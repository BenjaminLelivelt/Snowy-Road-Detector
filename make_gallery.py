"""Side-by-side result image for EVERY photo -> results/gallery/  (used by the README)

    python make_gallery.py

Each image: input | color-only baseline | ours (full + contour vote).
Green = kept as object edge, red = labeled shadow edge.
"""
import os

import cv2
import numpy as np

from shadow import classify as C

SETS = [("data/snow_web", "snow"), ("data/dry_web", "dry"), ("data/ISTD/test/test_A", "istd")]
ISTD_PICK = ["100-1.png", "103-2.png", "106-3.png", "110-3.png", "111-12.png", "114-1.png",
             "117-4.png", "120-10.png", "121-13.png", "124-2.png", "127-1.png", "130-3.png"]


def panel(img, text, w=420):
    h = round(img.shape[0] * w / img.shape[1])
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(t, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main():
    C.load_params()
    os.makedirs("results/gallery", exist_ok=True)
    for folder, tag in SETS:
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
        if tag == "istd":
            files = [f for f in ISTD_PICK if f in files]
        for f in files:
            d = C.compute(cv2.imread(os.path.join(folder, f)))
            name = os.path.splitext(f)[0]
            row = np.hstack([panel(d["img"], f"input {name}"),
                             panel(C.overlay(d["img"], C.label_map(d, "color"), thick=True), "color-only baseline"),
                             panel(C.overlay(d["img"], C.label_map(d, "full_vote"), thick=True), "ours: full + contour vote")])
            out = f"results/gallery/{tag}_{name}.jpg" if tag == "istd" else f"results/gallery/{name}.jpg"
            cv2.imwrite(out, row, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(tag, len(files))


if __name__ == "__main__":
    main()
