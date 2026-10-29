"""Turn cues into shadow/object labels, plus step 4: contour-level voting.

Label maps: 0 = no edge, 1 = object edge, 2 = shadow edge.

Methods (ablation ladder):
  canny        baseline: keep every edge
  color        literature-style single color cue: darker + same chromaticity
  width        proposal sharpness cue: 10%-90% transition wider than WIDTH_T
  blue         illumination-direction color cue: darker AND the dark side is
               shifted toward blue along the sun->sky direction in
               log-chromaticity (what an outdoor shadow does)
  msblur       multi-scale blur estimate larger than BLUR_T (soft penumbra)
  full         blue AND msblur (both independent cues must agree), per pixel
  full_vote    full + majority vote over each contour (contours split at junctions)
"""
import json
import os

import cv2
import numpy as np

from .edges import detect_edges
from .cues import color_cue, multiscale_blur_cue, sharpness_cue, texture_cue

OBJECT, SHADOW = 1, 2

# Default thresholds. tune.py overwrites these via results/params.json.
PARAMS = dict(
    CHROMA_T=0.06,   # color baseline: chroma diff below this -> same color
    DARK_T=0.9,      # dark side must be at most this fraction of the bright side
    WIDTH_T=4.0,     # width baseline: 10-90% transition wider than this (px)
    BLUE_T=0.05,     # projection of the log-chroma change on the shadow direction
    BLUR_T=0.5,      # multi-scale blur estimate (px) above this -> soft
    VOTE_T=0.5,      # contour is shadow if more than this fraction of pixels vote shadow
)

# Direction (in (log R/G, log B/G) space) in which chromaticity moves from the
# sunlit to the shadowed side: sunlight -> bluish skylight. Unit vector,
# estimated once from the ISTD training subset (see tune.py); it is a property
# of daylight + camera, not of any test image.
SHADOW_DIR = np.array([-0.526, 0.850], np.float32)

PER_METHOD = {}   # method -> its own tuned thresholds (fair ablation)

METHODS = ["canny", "color", "width", "blue", "msblur", "full", "full_vote"]


def load_params(path="results/params.json"):
    if os.path.exists(path):
        with open(path) as fh:
            p = json.load(fh)
        PARAMS.update({k: v for k, v in p.items() if k in PARAMS})
        if "SHADOW_DIR" in p:
            SHADOW_DIR[:] = p["SHADOW_DIR"]
        PER_METHOD.update(p.get("per_method", {}))
    return PARAMS


def params_for(method):
    """Global thresholds, overridden by the ones tuned for this method alone."""
    p = dict(PARAMS)
    p.update({k: v for k, v in PER_METHOD.get(method, {}).items() if k in PARAMS})
    if method == "full_vote":   # full_vote = full's pixel thresholds + its vote threshold
        p.update({k: v for k, v in PER_METHOD.get("full", {}).items() if k in PARAMS})
    return p


WORK_SIDE = 640   # images are processed at this longest side (ISTD's resolution)


def resize_to_work(img, interp=cv2.INTER_AREA, side=None):
    """Bring an image to the working scale so penumbra widths (pixels) mean the same thing."""
    side = side or WORK_SIDE
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def compute(img_bgr, sigma=1.4, low=40, high=100):
    """Run edge detection and every cue once. Returns a dict reused by every method.

    The image is first resized to WORK_SIDE; d["img"] is the image actually used,
    and all label maps have its size.
    """
    img_bgr = resize_to_work(img_bgr)
    edges, gray, nx, ny = detect_edges(img_bgr, sigma, low, high)
    raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(edges)
    c = color_cue(img_bgr, ys, xs, nx, ny)
    lr = c["lr"].reshape(-1, 3)
    dchrom = np.stack([lr[:, 2] - lr[:, 1], lr[:, 0] - lr[:, 1]], 1)  # d log(R/G), d log(B/G)
    return dict(
        img=img_bgr, edges=edges, ys=ys, xs=xs,
        chroma=c["chroma"], dark=c["dark"],
        blue=dchrom @ SHADOW_DIR,
        width=sharpness_cue(gray, ys, xs, nx, ny),
        blur=multiscale_blur_cue(raw, ys, xs, nx, ny),
        texture=texture_cue(raw, ys, xs, nx, ny),
    )


def pixel_votes(d, method, p=None):
    """Boolean array: True if that edge pixel is called a shadow."""
    p = p or params_for(method)
    darker = d["dark"] < p["DARK_T"]
    if method == "canny":
        return np.zeros(len(d["xs"]), bool)
    if method == "color":
        return darker & (d["chroma"] < p["CHROMA_T"])
    if method == "width":
        return darker & (d["width"] > p["WIDTH_T"])
    blue = darker & (d["blue"] > p["BLUE_T"])
    soft = darker & (d["blur"] > p["BLUR_T"])
    if method == "blue":
        return blue
    if method == "msblur":
        return soft
    if method in ("full", "full_vote"):
        return blue & soft
    raise ValueError(method)


def split_contours(edges):
    """Label 8-connected edge curves, cut at junctions (pixels with 3+ edge neighbours).

    Without the cut, a shadow border that touches an object outline would be
    merged with it into one component and forced to share one label.
    """
    e = edges.astype(np.uint8)
    nb = cv2.filter2D(e, cv2.CV_16S, np.ones((3, 3), np.float32)) - e
    junction = (e > 0) & (nb >= 3)
    pieces = e.copy()
    pieces[junction] = 0
    n, comp = cv2.connectedComponents(pieces, connectivity=8)
    return n, comp, junction


def contour_vote(edges, is_shadow, ys, xs, vote_t=0.5, min_len=8):
    """Each contour piece takes the majority label of its pixels.

    Junction pixels and pieces shorter than min_len keep their own pixel label.
    """
    n, comp, junction = split_contours(edges)
    c = comp[ys, xs]
    shadow_count = np.bincount(c, weights=is_shadow.astype(float), minlength=n)
    total = np.bincount(c, minlength=n)
    majority = shadow_count > vote_t * np.maximum(total, 1)
    out = majority[c]
    keep = (c == 0) | (total[c] < min_len) | junction[ys, xs]
    out[keep] = is_shadow[keep]
    return out


def label_map(d, method, p=None):
    p = p or params_for(method)
    is_shadow = pixel_votes(d, method, p)
    if method == "full_vote":
        is_shadow = contour_vote(d["edges"], is_shadow, d["ys"], d["xs"], p["VOTE_T"])
    lab = np.zeros(d["edges"].shape, np.uint8)
    lab[d["ys"], d["xs"]] = np.where(is_shadow, SHADOW, OBJECT)
    return lab


def overlay(img_bgr, lab, thick=False):
    """Object edges green, shadow edges red, on a dimmed copy of the image."""
    out = (img_bgr * 0.4).astype(np.uint8)
    if thick:
        k = np.ones((2, 2), np.uint8)
        obj = cv2.dilate((lab == OBJECT).astype(np.uint8), k) > 0
        sha = cv2.dilate((lab == SHADOW).astype(np.uint8), k) > 0
    else:
        obj, sha = lab == OBJECT, lab == SHADOW
    out[obj] = (0, 255, 0)
    out[sha] = (0, 0, 255)
    return out
