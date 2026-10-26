"""Step 1: Gaussian smoothing + Canny edges, plus gradient direction at each edge pixel."""
import cv2
import numpy as np


def detect_edges(img_bgr, sigma=1.4, low=40, high=100):
    """Return (edge_mask uint8 0/1, gray_blur float32, nx, ny).

    nx, ny = unit gradient direction (points from dark to bright).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    edges = cv2.Canny(blur, low, high)

    g = blur.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    return (edges > 0).astype(np.uint8), g, gx / mag, gy / mag
