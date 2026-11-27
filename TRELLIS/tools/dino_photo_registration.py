"""Offline DINOv2 dense correspondences, geometrically verified across views."""
import sys,json,os
from pathlib import Path
os.environ['TORCH_HOME']='/root/.cache/torch'
import numpy as np
import cv2,torch
from PIL import Image
from build_photo_ship_direction import read_obj,load_texture,render_map

root=Path(sys.argv[1]);result=json.loads((root/'direction_evidence.json').read_text())
(root/'sift_direction_evidence.json').write_text(json.dumps(result,indent=2))
model=torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main','dinov2_vitl14_reg',source='local',pretrained=False)
model.load_state_dict(torch.load('/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth',map_location='cpu'))
model=model.eval().cuda()


def features(im):
 h,w=im.shape[:2];s=490/max(h,w);nw,nh=round(w*s),round(h*s);ox=(518-nw)//2;oy=(518-nh)//2
 padded=np.full((518,518,3),230,np.uint8);padded[oy:oy+nh,ox:ox+nw]=cv2.resize(im,(nw,nh))
 tensor=torch.tensor(padded,device='cuda').permute(2,0,1).float()[None]/255
 mean=tensor.new_tensor([.485,.456,.406])[None,:,None,None];std=tensor.new_tensor([.229,.224,.225])[None,:,None,None]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):desc=model.forward_features((tensor-mean)/std)['x_norm_patchtokens'][0].float()
 yy,xx=np.meshgrid(np.arange(37)*14+7,np.arange(37)*14+7,indexing='ij');xy=np.c_[(xx.ravel()-ox)/s,(yy.ravel()-oy)/s]
 valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
 ids=np.flatnonzero(valid);pixel=np.rint(xy[ids]).astype(int);pixel[:,0]=np.clip(pixel[:,0],0,w-1);pixel[:,1]=np.clip(pixel[:,1],0,h-1)
 valid[ids]&=np.any(im[pixel[:,1],pixel[:,0]]!=230,axis=1)
 desc=desc[torch.tensor(valid,device='cuda')];desc=desc/desc.norm(dim=1,keepdim=True)
 return desc,xy[valid]


photo=np.asarray(Image.open(root/'semantics/aligned_photo.png').convert('RGB'));dp,xyp=features(photo)
v,vt,f,fvt=read_obj(Path(result['mesh']))
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
from auto_vehicle_parts import infer_frame
class Error(Exception):
 def __init__(self,m,code=5):super().__init__(m)
frame,_,_=infer_frame(v,f,'ship',{},Error);axes=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
local=v[:,axes];span=np.ptp(local[:,0]);material=next(x.split(None,1)[1].strip() for x in Path(result['mesh']).read_text().splitlines() if x.startswith('mtllib '))
texture=load_texture(str(Path(result['mesh']).parent/material),str(Path(result['mesh']).parent))
records=[];best=None
for index,(elev,azim) in enumerate([(e,a) for e in (25,50) for a in range(0,360,45)]):
 rendered=np.asarray(Image.open(root/f'view_{index:02d}.png').convert('RGB'));dr,xyr=features(rendered)
 similarity=dr@dp.T
 values,nn=similarity.topk(2,dim=1);reverse=similarity.argmax(dim=0)
 ids=torch.arange(len(dr),device='cuda');valid=(reverse[nn[:,0]]==ids)&(values[:,0]>.55)&((values[:,0]-values[:,1])>.01)
 ri=ids[valid].cpu().numpy();pi=nn[valid,0].cpu().numpy()
 if len(ri)<12:continue
 a=np.float32(xyr[ri]);b=np.float32(xyp[pi])
 M,inliers=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=8.,maxIters=10000,confidence=.999)
 if M is None or inliers.sum()<12:continue
 keep=np.flatnonzero(inliers.ravel());map_path=root/f'world_{index:02d}.npy'
 if map_path.exists():world=np.load(map_path)
 else:
  _,world=render_map(local,vt,f,fvt,texture,elev,azim);np.save(map_path,world)
 q=[];s=[];used=[]
 for j in keep:
  x,y=np.rint(a[j]).astype(int)
  if 0<=x<world.shape[1] and 0<=y<world.shape[0] and np.isfinite(world[y,x]):q.append(world[y,x]);s.append(b[j,0]);used.append(j)
 if len(q)<10:continue
 corr=float(np.corrcoef(q,s)[0,1]);coverage=float(np.ptp(q)/span)
 record={'view':index,'elevation':elev,'azimuth':azim,'inliers':len(q),'tentative_matches':len(ri),
   'mesh_long_axis_photo_u_correlation':corr,'longitudinal_coverage':coverage,'mapping_sign':int(np.sign(corr)),
   'accepted':len(q)>=12 and coverage>=.4 and abs(corr)>=.8,'method':'DINOv2 mutual NN + affine RANSAC + visible surface coordinate map'}
 records.append(record);print('DINO_REGISTRATION',record,flush=True)
 if record['accepted'] and (best is None or len(q)*coverage>best[0]):
  best=(len(q)*coverage,record)
  h=max(len(rendered),len(photo));canvas=np.full((h,rendered.shape[1]+photo.shape[1],3),230,np.uint8)
  canvas[:len(rendered),:rendered.shape[1]]=rendered;canvas[:len(photo),rendered.shape[1]:]=photo
  for j in used:
   start=tuple(np.rint(a[j]).astype(int));end=tuple(np.rint(b[j]+[rendered.shape[1],0]).astype(int))
   cv2.line(canvas,start,end,(40,180,60),1)
  Image.fromarray(canvas).save(root/'best_photo_model_matches.jpg')
accepted=[r for r in records if r['accepted']];result['dino_registration_candidates']=records
if best:
 sign=best[1]['mapping_sign'];agreement=sum(r['mapping_sign']==sign for r in accepted)/len(accepted)
 result['registration']={'best':best[1],'accepted_views':len(accepted),'sign_agreement':agreement,'feature_backend':'DINOv2 ViT-L/14 reg4'}
 if len(accepted)>=2 and agreement>=.8:
  result.update(status='ok',bow_sign=(1 if result['photo_semantics']['bow_end']==1 else -1)*sign,long_axis=frame['long_axis'],
     note='offline CLIP photo semantics + DINOv2 dense correspondences; uncalibrated evidence')
  result.pop('reason',None)
 else:result.update(status='rejected',reason='DINO image-model direction inconsistent across views')
else:result.update(status='rejected',reason='DINO image-model correspondences do not meet geometric checks')
(root/'direction_evidence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
