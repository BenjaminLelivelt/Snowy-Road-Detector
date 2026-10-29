"""Make DRAFT shadow masks for the snow photos (data/snow_web -> data/snow_masks_draft).

These are NOT ground truth. On snow, a shadow is lit only by blue skylight, so it
is darker than sunlit snow and shifted toward blue. We threshold exactly that
(brightness 20-85% of sunlit snow, log(B/R) at least 0.12 above sunlit snow),
clean up with morphology and drop sky blobs touching the top border.
Every draft must be checked / corrected by hand (tools/label_masks.py) before
the numbers are used as final results. Note the bias: the drafts use a
blue-shift rule, which favours our blue cue.

    python tools/draft_snow_masks.py
"""
import cv2,numpy as np,glob,os
os.makedirs('data/snow_masks_draft',exist_ok=True)
for f in sorted(glob.glob('data/snow_web/*.jpg')):
    im=cv2.imread(f).astype(np.float32)
    b,g,r=im[...,0],im[...,1],im[...,2]; s=b+g+r+1
    I=s/3; bl=np.log((b+1)/(r+1))
    # sunlit snow reference: bright and nearly neutral
    snowlit=(I>np.percentile(I,90)*0.85)&(np.abs(bl)<0.25)
    ref=np.median(I[snowlit]) if snowlit.sum()>500 else np.percentile(I,95)
    blref=np.median(bl[snowlit]) if snowlit.sum()>500 else 0
    sh=(I<0.85*ref)&(I>0.2*ref)&(bl-blref>0.12)
    sh=cv2.morphologyEx(sh.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    sh=cv2.morphologyEx(sh,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,cc,st,_=cv2.connectedComponentsWithStats(sh)
    keep=np.zeros_like(sh)
    for i in range(1,n):
        x,y,w,h,a=st[i]
        sky = y==0 and (y+h) < 0.6*sh.shape[0]          # blue sky touching the top
        if a>0.002*sh.size and not sky: keep[cc==i]=255
    cv2.imwrite('data/snow_masks_draft/'+os.path.basename(f).replace('.jpg','.png'),keep)
