"""Steps 2 and 3: per-edge-pixel cues.

Color cue:     across a shadow edge, brightness drops but chromaticity
               (R:G:B proportions) stays about the same.
Sharpness cue: shadow borders (penumbra) are soft/wide, object edges are sharp.
"""
import cv2
import numpy as np


def sample(img, xs, ys):
    """Bilinear sample img at float coords. img: HxW or HxWxC. Returns N or NxC."""
    mx = xs.reshape(1, -1).astype(np.float32)
    my = ys.reshape(1, -1).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(xs.size, -1).squeeze()


def color_cue(img_bgr, ys, xs, nx, ny, offset=4, patch=3):
    """Chromaticity difference between the two sides of each edge pixel.

    Returns (chroma_diff, dark_ratio). Small chroma_diff -> looks like a shadow.
    dark_ratio = mean brightness of dark side / bright side (per pixel).
    """
    img = cv2.blur(img_bgr.astype(np.float32), (patch, patch))  # small neighborhood average
    dx, dy = nx[ys, xs], ny[ys, xs]
    bright = sample(img, xs + offset * dx, ys + offset * dy)  # gradient points to bright side
    dark = sample(img, xs - offset * dx, ys - offset * dy)

    eps = 1.0
    cb = bright / (bright.sum(axis=1, keepdims=True) + eps)  # chromaticity
    cd = dark / (dark.sum(axis=1, keepdims=True) + eps)
    chroma_diff = np.abs(cb - cd).sum(axis=1)
    dark_ratio = (dark.sum(axis=1) + eps) / (bright.sum(axis=1) + eps)
    return chroma_diff, dark_ratio


def sharpness_cue(gray, ys, xs, nx, ny, half=8):
    """Width (pixels) of the brightness transition along the gradient (10%-90% rise).

    Large width -> soft edge -> looks like a shadow.
    """
    dx, dy = nx[ys, xs], ny[ys, xs]
    ts = np.arange(-half, half + 1, dtype=np.float32)
    # profile[i, k] = brightness at pixel i, step ts[k] along the gradient
    px = xs[:, None] + ts[None, :] * dx[:, None]
    py = ys[:, None] + ts[None, :] * dy[:, None]
    prof = sample(gray, px.ravel(), py.ravel()).reshape(len(xs), len(ts))

    lo = prof[:, :3].mean(axis=1, keepdims=True)   # dark plateau
    hi = prof[:, -3:].mean(axis=1, keepdims=True)  # bright plateau
    norm = (prof - lo) / (hi - lo + 1e-6)          # 0 .. 1 across the edge
    # first step where the profile passes 10% and 90%
    i10 = np.argmax(norm >= 0.1, axis=1)
    i90 = np.argmax(norm >= 0.9, axis=1)
    width = (i90 - i10).astype(np.float32)
    width[width < 0] = 2 * half  # messy profile, treat as soft
    return width
