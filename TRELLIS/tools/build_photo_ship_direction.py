"""Photo end semantics + verified textured-view correspondence; no manual bow label."""
import argparse,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/PartField')
from render_textured import read_obj,load_texture,view_basis


def geometry_hash(v,f):
 return hashlib.sha256(np.asarray(v,'<f8').tobytes()+np.asarray(f,'<i8').tobytes()).hexdigest()


def render_map(v,vt,faces,fvt,texture,elev,azim,px=420,frame_bounds=None):
 """Reuse the existing orthographic texture rasterizer, adding a long-axis map."""
 r,u,forward=view_basis(elev,azim)
 center=(v.min(0)+v.max(0))/2 if frame_bounds is None else np.mean(frame_bounds,axis=0)
 p=v-center
 x,y,d=p@r,p@u,p@forward
 if frame_bounds is None:
  lo=np.array([x.min(),y.min()]);hi=np.array([x.max(),y.max()])
 else:
  import itertools
  corners=np.array(list(itertools.product(*zip(*np.asarray(frame_bounds)))))-center
  cx,cy=corners@r,corners@u
  lo=np.array([cx.min(),cy.min()]);hi=np.array([cx.max(),cy.max()])
 pad=max(hi-lo)*.06
 lo-=pad;hi+=pad;scale=px/max(hi-lo);W,H=(np.ceil((hi-lo)*scale)+1).astype(int)
 sx=(x-lo[0])*scale;sy=(y-lo[1])*scale
 image=np.full((H,W,3),230,np.uint8);zbuf=np.full((H,W),-np.inf);world=np.full((H,W),np.nan)
 th,tw=texture.shape[:2]
 for ti in np.argsort(d[faces].mean(1)):
  face=faces[ti];uv=fvt[ti];X=sx[face];Y=sy[face];Z=d[face]
  x0=max(0,int(np.floor(X.min())));x1=min(W-1,int(np.ceil(X.max())))
  y0=max(0,int(np.floor(Y.min())));y1=min(H-1,int(np.ceil(Y.max())))
  if x1<x0 or y1<y0:continue
  den=(Y[1]-Y[2])*(X[0]-X[2])+(X[2]-X[1])*(Y[0]-Y[2])
  if abs(den)<1e-12:continue
  gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
  a=((Y[1]-Y[2])*(gx-X[2])+(X[2]-X[1])*(gy-Y[2]))/den
  b=((Y[2]-Y[0])*(gx-X[2])+(X[0]-X[2])*(gy-Y[2]))/den;c=1-a-b
  depth=a*Z[0]+b*Z[1]+c*Z[2];sub=zbuf[y0:y1+1,x0:x1+1]
  mask=(a>=-1e-6)&(b>=-1e-6)&(c>=-1e-6)&(depth>sub)
  if not mask.any() or uv.min()<0:continue
  uu=a*vt[uv[0],0]+b*vt[uv[1],0]+c*vt[uv[2],0]
  vv=a*vt[uv[0],1]+b*vt[uv[1],1]+c*vt[uv[2],1]
  tx=np.clip((uu%1)*(tw-1),0,tw-1).astype(int);ty=np.clip((1-vv%1)*(th-1),0,th-1).astype(int)
  image[y0:y1+1,x0:x1+1][mask]=texture[ty,tx][mask];sub[mask]=depth[mask]
  world[y0:y1+1,x0:x1+1][mask]=(a*v[face[0],0]+b*v[face[1],0]+c*v[face[2],0])[mask]
 return np.flipud(image).copy(),np.flipud(world).copy()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mesh',required=True);parser.add_argument('--photo',required=True);parser.add_argument('--output-dir',required=True)
 args=parser.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
 source=Path(args.photo);mesh=Path(args.mesh)
 v,vt,f,fvt=read_obj(mesh)
 sys.path.insert(0,'/root/TRELLIS/trellis/utils')
 from photo_direction_bridge import asset_hash
 result={'status':'rejected','mesh_geometry_sha256':geometry_hash(v,f),
  'photo_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'photo':str(source),'mesh':str(mesh),
  'mesh_appearance_sha256':asset_hash(mesh)}
 def finish(reason=None):
  if reason:result['reason']=reason
  (root/'direction_evidence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
 def dense_fallback(reason):
  finish(reason)
  subprocess.run([sys.executable,str(Path(__file__).with_name('dino_photo_registration.py')),str(root)],check=True)
 image=Image.open(source).convert('RGBA')
 if np.all(np.asarray(image)[:,:,3]==255):
  import os,rembg
  os.environ['U2NET_HOME']='/root/.u2net'
  image=rembg.remove(image.convert('RGB'),session=rembg.new_session('isnet-general-use',providers=['CPUExecutionProvider']))
 image.save(root/'foreground_photo.png')
 semantic=root/'semantics'
 if not (semantic/'clip_probe.json').exists():
  subprocess.run([sys.executable,str(Path(__file__).with_name('probe_photo_bow.py')),str(semantic),str(root/'foreground_photo.png')],check=True)
 probe=json.loads((semantic/'clip_probe.json').read_text())
 # Every prompt votes between the two ENDS, not merely bow-vs-stern for one crop.
 margins=-np.asarray(probe['end0_minus_end1_per_prompt'])
 votes=np.sign(margins);direction=1 if margins.mean()>0 else -1
 agreement=float((votes==direction).mean());strength=float(abs(np.median(margins)))
 result['photo_semantics']={'bow_end':1 if direction>0 else 0,'vote_agreement':agreement,'median_cosine_margin':strength,
  'prompt_and_mirror_margins':margins.tolist(),'method':'offline CLIP paired end crops + mirror consistency; uncalibrated'}
 if agreement<.875 or strength<.008:return finish('photo semantic evidence weak or inconsistent')
 # Derive axes from mesh, with no front sign supplied.
 sys.path.insert(0,'/root/TRELLIS/trellis/utils')
 from auto_vehicle_parts import infer_frame
 class Failure(Exception):
  def __init__(self,message,code=5):super().__init__(message)
 frame,geom,_=infer_frame(v,f,'ship',{},Failure)
 axes=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
 local=v[:,axes];span=np.ptp(local[:,0])
 material=next(x.split(None,1)[1].strip() for x in mesh.read_text().splitlines() if x.startswith('mtllib '))
 texture=load_texture(str(mesh.parent/material),str(mesh.parent))
 if texture is None:return finish('mesh texture missing')
 photo=np.array(Image.open(semantic/'aligned_photo.png').convert('RGB'))
 sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.004)
 photo_mask=(np.any(photo!=230,axis=2)*255).astype('uint8')
 kp_photo,desc_photo=sift.detectAndCompute(cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY),photo_mask)
 if desc_photo is None:return finish('photograph lacks usable features')
 matches=[];best=None
 views=[(e,a) for e in (25,50) for a in range(0,360,45)]
 for index,(elev,azim) in enumerate(views):
  path=root/f'view_{index:02d}.png';map_path=root/f'world_{index:02d}.npy'
  if path.exists():
   rendered=np.asarray(Image.open(path).convert('RGB'));world=np.load(map_path) if map_path.exists() else None
  else:
   rendered,world=render_map(local,vt,f,fvt,texture,elev,azim)
   Image.fromarray(rendered).save(path);np.save(map_path,world)
  kp,desc=sift.detectAndCompute(cv2.cvtColor(rendered,cv2.COLOR_RGB2GRAY),None)
  if desc is None:continue
  pairs=cv2.BFMatcher().knnMatch(desc,desc_photo,k=2)
  good=[a for a,b in pairs if a.distance<.85*b.distance]
  if len(good)<8:continue
  a=np.float32([kp[m.queryIdx].pt for m in good]);b=np.float32([kp_photo[m.trainIdx].pt for m in good])
  H,inliers=cv2.findHomography(a,b,cv2.RANSAC,4.)
  if H is None:continue
  if int(inliers.sum())<10:continue
  if world is None:
   _,world=render_map(local,vt,f,fvt,texture,elev,azim);np.save(map_path,world)
  valid=np.flatnonzero(inliers.ravel());q=[];s=[];used=[]
  for j in valid:
   x,y=np.rint(a[j]).astype(int)
   if 0<=y<world.shape[0] and 0<=x<world.shape[1] and np.isfinite(world[y,x]):q.append(world[y,x]);s.append(b[j,0]);used.append(j)
  if len(q)<8:continue
  q=np.asarray(q);s=np.asarray(s);corr=float(np.corrcoef(q,s)[0,1]);coverage=float(np.ptp(q)/span)
  record={'view':index,'elevation':elev,'azimuth':azim,'inliers':len(q),'tentative_matches':len(good),
   'mesh_long_axis_photo_u_correlation':corr,'longitudinal_coverage':coverage,'mapping_sign':int(np.sign(corr)),
   'accepted':len(q)>=10 and coverage>=.35 and abs(corr)>=.8}
  matches.append(record);print('REGISTRATION',record,flush=True)
  if record['accepted'] and (best is None or len(q)*coverage>best[0]):
   best=(len(q)*coverage,record)
   display=cv2.drawMatches(cv2.cvtColor(rendered,cv2.COLOR_RGB2BGR),kp,cv2.cvtColor(photo,cv2.COLOR_RGB2BGR),kp_photo,[good[j] for j in used],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   cv2.imwrite(str(root/'best_photo_model_matches.jpg'),display)
 result['registration_candidates']=matches
 accepted=[m for m in matches if m['accepted']]
 if best is None:return dense_fallback('SIFT correspondence insufficient; attempting offline DINOv2')
 sign=best[1]['mapping_sign'];agreement=sum(m['mapping_sign']==sign for m in accepted)/len(accepted)
 result['registration']={'best':best[1],'accepted_views':len(accepted),'sign_agreement':agreement}
 if len(accepted)<2 or agreement<.8:return dense_fallback('SIFT direction unstable; attempting offline DINOv2')
 result.update(status='ok',bow_sign=direction*sign,long_axis=frame['long_axis'],
  note='photo-assisted heuristic with SIFT/RANSAC correspondence; not calibrated semantic certainty')
 finish()


if __name__=='__main__':main()
