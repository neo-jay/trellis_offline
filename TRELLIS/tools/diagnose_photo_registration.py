import sys,cv2,numpy as np
from pathlib import Path
root=Path(sys.argv[1]);s=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.004)
photo=cv2.imread(str(root/'semantics/aligned_photo.png'));kp2,d2=s.detectAndCompute(photo,None)
for path in sorted(root.glob('view_*.png')):
 im=cv2.imread(str(path));kp,d=s.detectAndCompute(im,None)
 pairs=cv2.BFMatcher().knnMatch(d,d2,k=2)
 results=[]
 for ratio in (.75,.85,.9):
  good=[a for a,b in pairs if a.distance<ratio*b.distance]
  count=0
  if len(good)>=4:
   a=np.float32([kp[m.queryIdx].pt for m in good]);b=np.float32([kp2[m.trainIdx].pt for m in good])
   H,inliers=cv2.findHomography(a,b,cv2.RANSAC,4.)
   count=0 if H is None else int(inliers.sum())
  results.append((ratio,len(good),count))
 print(path.name,len(kp),results,flush=True)
