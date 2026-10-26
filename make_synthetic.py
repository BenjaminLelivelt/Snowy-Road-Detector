"""Make a few fake test images (objects with sharp edges + soft cast shadows) with
shadow masks, so the whole pipeline can be checked before real data is ready.

    python make_synthetic.py        -> data/synthetic/images, data/synthetic/masks
"""
import os

import cv2
import numpy as np

OUT = "data/synthetic"


def make(seed, snow=False):
    rng = np.random.default_rng(seed)
    h, w = 360, 480
    ground = (235, 238, 240) if snow else (120, 140, 150)        # BGR
    img = np.full((h, w, 3), ground, np.float32)
    img += rng.normal(0, 4, img.shape)                            # sensor noise

    # cast shadow: soft-edged polygon, darker + slightly bluer (skylight)
    mask = np.zeros((h, w), np.uint8)
    x0 = rng.integers(40, 200)
    pts = np.array([[x0, 120], [x0 + 140, 120], [x0 + 260, 340], [x0 + 60, 340]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 3.0)[..., None]
    shade = np.array([0.55, 0.45, 0.42], np.float32) if snow else np.array([0.5, 0.45, 0.45], np.float32)
    img = img * (1 - soft) + img * shade * soft

    # objects: sharp-edged colored boxes
    for _ in range(2):
        x, y = rng.integers(20, w - 120), rng.integers(20, h - 120)
        color = rng.integers(20, 230, 3).tolist()
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 80, int(y) + 60), color, -1)

    return np.clip(img, 0, 255).astype(np.uint8), mask


def main():
    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)
    for i in range(6):
        img, mask = make(i, snow=(i % 2 == 1))
        cv2.imwrite(f"{OUT}/images/syn_{i}.png", img)
        cv2.imwrite(f"{OUT}/masks/syn_{i}.png", mask)
    print("wrote 6 images to", OUT)


if __name__ == "__main__":
    main()
