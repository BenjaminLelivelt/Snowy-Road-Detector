"""Hand-correct shadow masks (white = shadow) with the mouse.

    python tools/label_masks.py data/snow_web data/snow_masks_draft data/snow_masks

Starts from the draft mask if one exists (second folder), saves to the third.
  left drag   paint shadow          right drag   erase
  [ / ]       brush smaller/larger
  s           save + next image     n / p        next / previous without saving
  r           reset to draft        q            quit
"""
import os
import sys

import cv2
import numpy as np

EXTS = (".png", ".jpg", ".jpeg")


def main():
    src, draft, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXTS))
    i, brush = 0, 8
    state = {}

    def load(k):
        name = os.path.splitext(files[k])[0] + ".png"
        img = cv2.imread(os.path.join(src, files[k]))
        for d in (out, draft):
            p = os.path.join(d, name)
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                return img, cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST), name
        return img, np.zeros(img.shape[:2], np.uint8), name

    def on_mouse(ev, x, y, flags, _):
        state["xy"] = (x, y)
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(state["mask"], (x, y), brush, 255, -1)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(state["mask"], (x, y), brush, 0, -1)

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)
    img, state["mask"], name = load(i)
    while True:
        vis = img.copy()
        m = state["mask"] > 0
        vis[m] = (0.5 * vis[m] + (0, 0, 127)).astype(np.uint8)
        cnts, _ = cv2.findContours(state["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
        cv2.putText(vis, f"{i + 1}/{len(files)} {name} brush={brush}", (8, 24), 0, 0.7, (0, 255, 0), 2)
        cv2.imshow("label", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        if k == ord("["):
            brush = max(1, brush - 2)
        if k == ord("]"):
            brush += 2
        if k == ord("r"):
            img, state["mask"], name = load(i)
        if k in (ord("s"), ord("n"), ord("p")):
            if k == ord("s"):
                cv2.imwrite(os.path.join(out, name), state["mask"])
                print("saved", name)
            i = (i - 1) % len(files) if k == ord("p") else (i + 1) % len(files)
            img, state["mask"], name = load(i)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
