"""Evaluation against shadow masks (ISTD/SBU style) and the Hough demo."""
import cv2
import numpy as np

from .classify import OBJECT, SHADOW


def gt_shadow_band(mask, tol=3):
    """Shadow-edge ground truth = border of the shadow mask, widened by tol px."""
    m = (mask > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    border = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    if tol > 0:
        border = cv2.dilate(border, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1,) * 2))
    return border > 0


def counts(lab, band):
    """Pixel counts over Canny edge pixels. Positive class = shadow edge."""
    edge = lab > 0
    pred_s = lab == SHADOW
    true_s = band & edge
    return dict(
        tp=int((pred_s & true_s).sum()),
        fp=int((pred_s & ~true_s).sum()),
        fn=int((~pred_s & true_s).sum()),
        tn=int((edge & ~pred_s & ~true_s).sum()),
    )


def metrics(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(
        precision=p, recall=r, f1=f1,
        shadow_removed=r,                                        # % shadow edges removed
        object_removed=fp / (fp + tn) if fp + tn else 0.0,      # % object edges wrongly removed
    )


def hough_lines(edge_mask, thresh=50, min_len=40, max_gap=5):
    lines = cv2.HoughLinesP(edge_mask.astype(np.uint8) * 255, 1, np.pi / 180,
                            thresh, minLineLength=min_len, maxLineGap=max_gap)
    return [] if lines is None else lines.reshape(-1, 4).tolist()


def count_shadow_lines(lines, band, frac=0.5):
    """A line is 'caused by a shadow' if most of its pixels lie on the GT shadow border."""
    n = 0
    for x1, y1, x2, y2 in lines:
        canvas = np.zeros(band.shape, np.uint8)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 1, 1)
        on = canvas > 0
        if (band & on).sum() > frac * on.sum():
            n += 1
    return n
