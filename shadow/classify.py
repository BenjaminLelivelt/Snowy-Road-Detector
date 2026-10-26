"""Turn cues into shadow/object labels, plus step 4: contour-level voting.

Label maps: 0 = no edge, 1 = object edge, 2 = shadow edge.
"""
import cv2
import numpy as np

from .edges import detect_edges
from .cues import color_cue, sharpness_cue

OBJECT, SHADOW = 1, 2

# Thresholds (tune these on your data)
CHROMA_T = 0.06   # chroma diff below this -> same color on both sides
WIDTH_T = 4.0     # transition wider than this (px) -> soft edge
DARK_T = 0.9      # dark side must be at most this fraction of bright side

METHODS = ["canny", "color", "sharp", "full", "full_vote"]


def compute(img_bgr, sigma=1.4, low=40, high=100):
    """Run edge detection and both cues once. Returns a dict reused by every method."""
    edges, gray, nx, ny = detect_edges(img_bgr, sigma, low, high)
    ys, xs = np.nonzero(edges)
    chroma, dark = color_cue(img_bgr, ys, xs, nx, ny)
    width = sharpness_cue(gray, ys, xs, nx, ny)
    return dict(edges=edges, ys=ys, xs=xs, chroma=chroma, dark=dark, width=width)


def pixel_votes(d, method):
    """Boolean array: True if that edge pixel is called a shadow."""
    darker = d["dark"] < DARK_T
    color_ok = (d["chroma"] < CHROMA_T) & darker
    soft = (d["width"] > WIDTH_T) & darker
    if method == "canny":
        return np.zeros(len(d["xs"]), bool)      # keep every edge
    if method == "color":
        return color_ok
    if method == "sharp":
        return soft
    if method in ("full", "full_vote"):
        return color_ok & soft                   # both cues must agree
    raise ValueError(method)


def contour_vote(edges, is_shadow, ys, xs):
    """Each 8-connected edge contour takes the majority label of its pixels."""
    n, comp = cv2.connectedComponents(edges, connectivity=8)
    c = comp[ys, xs]
    shadow_count = np.bincount(c, weights=is_shadow.astype(float), minlength=n)
    total = np.bincount(c, minlength=n)
    majority = shadow_count > total / 2
    return majority[c]


def label_map(d, method):
    is_shadow = pixel_votes(d, method)
    if method == "full_vote":
        is_shadow = contour_vote(d["edges"], is_shadow, d["ys"], d["xs"])
    lab = np.zeros(d["edges"].shape, np.uint8)
    lab[d["ys"], d["xs"]] = np.where(is_shadow, SHADOW, OBJECT)
    return lab


def overlay(img_bgr, lab):
    """Object edges green, shadow edges red, on a dimmed copy of the image."""
    out = (img_bgr * 0.4).astype(np.uint8)
    out[lab == OBJECT] = (0, 255, 0)
    out[lab == SHADOW] = (0, 0, 255)
    return out
