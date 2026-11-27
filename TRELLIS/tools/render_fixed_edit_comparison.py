"""Compare actual exports with shared camera, bounds and scale (no auto-fit bias)."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_photo_ship_direction import read_obj,load_texture,render_map,view_basis

output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True);items=[]
for path in sys.argv[2:]:
 summary=Path(path)
 for row in json.loads(summary.read_text())['rows']:
  if row.get('result',{}).get('code')==0:
   row=dict(row);row['_summary_dir']=str(summary.parent);items.append(row)
# Keep every successful actual example up to six; no rejected proposed mesh rendered as success.
items=items[:6]
if not items:raise RuntimeError('No successful exports')
fig,axes=plt.subplots(len(items),4,figsize=(18,3.7*len(items)),squeeze=False)
for i,row in enumerate(items):
 source=Path(row['source']);dest=Path(row['result']['obj_path'])
 V,VT,F,FVT=read_obj(source);new,_,_,_=read_obj(dest)
 frame=row['localization']['automatic_localization']['frame'];order=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
 before=V[:,order];after=new[:,order];bounds=np.array([np.minimum(before.min(0),after.min(0)),np.maximum(before.max(0),after.max(0))])
 rendered_images=[]
 for col,path,points in [(0,source,before),(1,dest,after)]:
  material=next(t.split(None,1)[1].strip() for t in path.read_text().splitlines() if t.startswith('mtllib '))
  tex=load_texture(str(path.parent/material),str(path.parent))
  if tex is None:raise RuntimeError('Missing texture: '+str(path))
  img,_=render_map(points,VT,F,FVT,tex,25,55,px=440,frame_bounds=bounds)
  rendered_images.append(img)
  axes[i,col].imshow(img);axes[i,col].axis('off')
  axes[i,col].set_title(row['name']+(' / Before' if col==0 else ' / After'))
 with np.load(Path(row['_summary_dir'])/(row['name']+'_selection.npz')) as a:w=a['weights']
 edited_axis='xyz'.index(row['localization']['edit_axis']);a=order[0];b=order[1] if edited_axis==order[1] else order[2]
 ids=np.linspace(0,len(V)-1,min(15000,len(V))).astype(int);selected=ids[w[ids]>.3]
 ax=axes[i,2];ax.scatter(V[ids,a],V[ids,b],s=.5,c='#9daab8',label='Original')
 ax.scatter(new[selected,a],new[selected,b],s=1,c='#eb7c25',label='Edited target')
 ax.axis('equal');ax.grid(alpha=.15);ax.legend(markerscale=4,fontsize=8)
 control=row['localization']['dimension_control'];ax.set_title('Same-scale overlay / core '+f"{control['achieved_relative_change']:+.1%}")
 # Identical pixel crop around the union of original and edited target.
 import itertools
 r,u,_=view_basis(25,55);center=bounds.mean(0)
 corners=np.array(list(itertools.product(*zip(*bounds))))-center
 low=np.array([(corners@r).min(),(corners@u).min()]);high=np.array([(corners@r).max(),(corners@u).max()]);pad=max(high-low)*.06;low-=pad;high+=pad;scale=440/max(high-low)
 selected=np.flatnonzero(w>.7*w.max());target=np.vstack((before[selected],after[selected]))-center
 xx=((target@r)-low[0])*scale;yy=rendered_images[0].shape[0]-1-((target@u)-low[1])*scale
 margin=max(8,.08*max(np.ptp(xx),np.ptp(yy)))
 x0=max(0,int(xx.min()-margin));x1=min(rendered_images[0].shape[1],int(xx.max()+margin)+1)
 y0=max(0,int(yy.min()-margin));y1=min(rendered_images[0].shape[0],int(yy.max()+margin)+1)
 zoom=np.concatenate([im[y0:y1,x0:x1] for im in rendered_images],axis=1)
 axes[i,3].imshow(zoom);axes[i,3].axis('off');axes[i,3].set_title('Target zoom: Before | After')
fig.suptitle('Actual exported models | common camera, centre and scale per before/after pair\nOrange overlay exposes displacement; core % is not whole-object %',fontsize=12)
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output/'fixed_scale_comparison.png',dpi=130);plt.close(fig)
print(output/'fixed_scale_comparison.png',flush=True)
