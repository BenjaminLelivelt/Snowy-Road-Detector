"""Steps 2 and 3: per-edge-pixel cues.

Every cue is measured by sampling the image on the two sides of an edge pixel,
along the gradient direction (n points from the dark side to the bright side).

Color cues      across a shadow edge the surface is the same, so brightness
                drops but chromaticity (R:G:B proportions) stays about the same,
                and every channel is darkened by a similar factor.
Texture cue     the same texture lit less brightly keeps the same *relative*
                contrast (std / mean), so this ratio should match on both sides.
Sharpness cues  shadow borders (penumbra) are soft/wide, object edges are sharp:
                (a) 10%-90% transition width of the brightness profile,
                (b) multi-scale blur estimate from gradient magnitudes at two
                    Gaussian scales (a Gaussian-blurred step of width s has peak
                    gradient ~ 1/sqrt(s^2 + sigma^2)).
"""
import cv2
import numpy as np


def sample(img, xs, ys):
    """Bilinear sample img at float coords. img: HxW or HxWxC. Returns N or NxC."""
    # cv2.remap needs maps smaller than 32767 px per side, so lay the N points
    # out as a (rows x 4096) grid instead of one long row.
    n = xs.size
    cols = 4096
    rows = max(1, -(-n // cols))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n] = xs.ravel()
    my[:n] = ys.ravel()
    out = cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n].squeeze()


def _sides(img, ys, xs, nx, ny, offset):
    dx, dy = nx[ys, xs], ny[ys, xs]
    bright = sample(img, xs + offset * dx, ys + offset * dy)  # gradient points to bright side
    dark = sample(img, xs - offset * dx, ys - offset * dy)
    return bright, dark


def color_cue(img_bgr, ys, xs, nx, ny, offset=4, patch=3):
    """Cross-edge color comparison.

    Returns a dict of per-pixel arrays:
      chroma    L1 difference of chromaticity (b,g,r)/(b+g+r) between the sides
      dark      mean brightness of dark side / bright side
      lr        Nx3 per-channel log ratios log(dark_c / bright_c)   (B, G, R)
      spread    std of the three log ratios (0 = every channel darkened equally)
      blue      lr_B - lr_R  (> 0: dark side is relatively bluer, as in a
                sky-lit outdoor shadow)
    """
    img = cv2.blur(img_bgr.astype(np.float32), (patch, patch))
    bright, dark = _sides(img, ys, xs, nx, ny, offset)
    eps = 1.0
    cb = bright / (bright.sum(axis=1, keepdims=True) + eps)
    cd = dark / (dark.sum(axis=1, keepdims=True) + eps)
    lr = np.log((dark + eps) / (bright + eps))
    return dict(
        chroma=np.abs(cb - cd).sum(axis=1),
        dark=(dark.sum(axis=1) + eps) / (bright.sum(axis=1) + eps),
        lr=lr,
        spread=lr.std(axis=1),
        blue=lr[:, 0] - lr[:, 2],
    )


def texture_cue(gray, ys, xs, nx, ny, offset=6, win=7):
    """|log(cv_bright / cv_dark)|, cv = local std / local mean (coefficient of variation).

    Illumination scales mean and std together, so cv is unchanged across a
    shadow edge but usually changes across a material boundary.
    """
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (win, win))
    sq = cv2.blur(g * g, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    cv = (std + 1.0) / (mean + 1.0)
    b, d = _sides(cv, ys, xs, nx, ny, offset)
    return np.abs(np.log(b / d))


def sharpness_cue(gray, ys, xs, nx, ny, half=8):
    """Width (pixels) of the brightness transition along the gradient (10%-90% rise).

    Large width -> soft edge -> looks like a shadow.
    """
    dx, dy = nx[ys, xs], ny[ys, xs]
    ts = np.arange(-half, half + 1, dtype=np.float32)
    px = xs[:, None] + ts[None, :] * dx[:, None]
    py = ys[:, None] + ts[None, :] * dy[:, None]
    prof = sample(gray, px.ravel(), py.ravel()).reshape(len(xs), len(ts))

    lo = prof[:, :3].mean(axis=1, keepdims=True)   # dark plateau
    hi = prof[:, -3:].mean(axis=1, keepdims=True)  # bright plateau
    norm = (prof - lo) / (hi - lo + 1e-6)          # 0 .. 1 across the edge
    i10 = np.argmax(norm >= 0.1, axis=1)
    i90 = np.argmax(norm >= 0.9, axis=1)
    width = (i90 - i10).astype(np.float32)
    width[width < 0] = 2 * half  # messy profile, treat as soft
    return width


def _grad_mag(gray, sigma):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def multiscale_blur_cue(gray_raw, ys, xs, nx, ny, s1=1.0, s2=3.0, search=2):
    """Estimated edge blur (pixels) from the drop of gradient strength between two scales.

    For an ideal step of height A blurred by a Gaussian of std s, smoothing with
    another Gaussian of std sigma gives peak gradient A / sqrt(2*pi*(s^2+sigma^2)).
    So r = G(s2) / G(s1) = sqrt(s^2+s1^2) / sqrt(s^2+s2^2), and
        s^2 = (r^2 s2^2 - s1^2) / (1 - r^2).
    Sharp object edges -> small s; soft shadow borders (penumbra) -> large s.
    Peaks are searched +-search px along the normal because blur can shift them.
    """
    g1, g2 = _grad_mag(gray_raw, s1), _grad_mag(gray_raw, s2)
    dx, dy = nx[ys, xs], ny[ys, xs]
    best1 = np.zeros(len(xs), np.float32)
    best2 = np.zeros(len(xs), np.float32)
    for t in range(-search, search + 1):
        best1 = np.maximum(best1, sample(g1, xs + t * dx, ys + t * dy))
        best2 = np.maximum(best2, sample(g2, xs + t * dx, ys + t * dy))
    r = np.clip(best2 / (best1 + 1e-6), 0.0, 0.999)
    s_sq = (r * r * s2 * s2 - s1 * s1) / (1 - r * r)
    return np.sqrt(np.clip(s_sq, 0, 15 ** 2))
