"""Step 3 preview: try to find lane / road lines on the snowy road photos.

    python lane_demo.py      -> results/lanes/*.jpg, results/lanes/lanes.csv

Standard lane pipeline: edges -> keep the lower part of the image (road region)
-> probabilistic Hough -> keep lines that are not near-horizontal.
Run twice: on raw Canny edges, and on edges after removing our shadow edges.
A line counts as "on a shadow" if most of it lies on a (draft) shadow-mask border.
"""
import csv
import os

import cv2
import numpy as np

from shadow import classify as C
from shadow.evaluate import gt_shadow_band

ROADS = ["snow_00", "snow_01", "snow_02", "snow_03", "snow_04", "snow_05", "snow_06",
         "snow_07", "snow_08", "snow_09", "snow_11", "snow_12", "snow_13"]
ROI_TOP = 0.45          # ignore the top 45% (sky, trees)
MIN_ANGLE = 20          # degrees from horizontal; lanes are not horizontal


def lanes(edge_mask):
    h, w = edge_mask.shape
    m = edge_mask.astype(np.uint8) * 255
    m[: int(ROI_TOP * h)] = 0
    ls = cv2.HoughLinesP(m, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=8)
    out = []
    for x1, y1, x2, y2 in ([] if ls is None else ls.reshape(-1, 4)):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        ang = min(ang, 180 - ang)
        if ang >= MIN_ANGLE:
            out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def on_shadow(line, band):
    canvas = np.zeros(band.shape, np.uint8)
    cv2.line(canvas, line[:2], line[2:], 1, 1)
    on = canvas > 0
    return (band & on).sum() > 0.5 * on.sum()


def draw(img, edge_mask, ls, band, title):
    vis = (img * 0.45).astype(np.uint8)
    vis[edge_mask > 0] = (110, 110, 110)
    h = vis.shape[0]
    cv2.line(vis, (0, int(ROI_TOP * h)), (vis.shape[1], int(ROI_TOP * h)), (255, 255, 0), 1)
    n_sh = 0
    for l in ls:
        s = on_shadow(l, band)
        n_sh += s
        cv2.line(vis, l[:2], l[2:], (0, 0, 255) if s else (0, 255, 0), 2)
    t = f"{title}: {len(ls)} lines, {n_sh} on shadows"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis, n_sh


def main():
    C.load_params()
    os.makedirs("results/lanes", exist_ok=True)
    rows = []
    for name in ROADS:
        d = C.compute(cv2.imread(f"data/snow_web/{name}.jpg"))
        img = d["img"]
        m = cv2.imread(f"data/snow_masks_draft/{name}.png", 0)
        band = gt_shadow_band(cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST), 3)
        raw = d["edges"] > 0
        clean = C.label_map(d, "full_vote") == C.OBJECT
        l_raw, l_clean = lanes(raw), lanes(clean)
        a, s_raw = draw(img, raw, l_raw, band, "raw Canny + Hough")
        b, s_clean = draw(img, clean, l_clean, band, "shadow edges removed + Hough")
        inp = img.copy()
        cv2.rectangle(inp, (0, 0), (inp.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(inp, f"input {name}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sep = np.full((img.shape[0], 4, 3), 255, np.uint8)
        cv2.imwrite(f"results/lanes/{name}.jpg", np.hstack([inp, sep, a, sep, b]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append([name, len(l_raw), s_raw, len(l_clean), s_clean])
        print(rows[-1])
    with open("results/lanes/lanes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "lines_raw", "shadow_lines_raw", "lines_clean", "shadow_lines_clean"])
        w.writerows(rows)
    t = np.array([r[1:] for r in rows]).sum(0)
    print("TOTAL raw lines", t[0], "on shadows", t[1], "| cleaned lines", t[2], "on shadows", t[3])


if __name__ == "__main__":
    main()
