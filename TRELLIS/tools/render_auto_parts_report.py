"""Actual textured exports + automatic selection, reuse installed rasterizer."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/autodl-tmp/PartField')
from render_textured import read_obj,render,load_texture,view_basis

root=Path(sys.argv[1]);data=json.loads((root/'summary.json').read_text())
preferred=['ship_previous_0','ship_previous_2','f22_0','f22_2']
items=[x for name in preferred for x in data['rows'] if x['name']==name]
if not items:items=data['rows']
items=[x for x in items if x['result']['code']==0]
items=items[:4]
if not items:raise RuntimeError('No successful exported edits to visualize')
fig,axs=plt.subplots(len(items),3,figsize=(14,3.6*len(items)),squeeze=False)
for row,item in enumerate(items):
    source=Path(item['source']);edited=Path(item['result']['obj_path'])
    frame=item['localization']['automatic_localization']['frame']
    axes=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
    with np.load(root/(item['name']+'_selection.npz')) as a:weights=a['weights']
    for col,path in ((0,source),(2,edited)):
        V,VT,F,FVT=read_obj(path);V=V[:,axes]
        material=next(x.split(None,1)[1].strip() for x in path.read_text().splitlines() if x.startswith('mtllib '))
        tex=load_texture(str(path.parent/material),str(path.parent))
        if tex is None:raise RuntimeError('texture missing: '+str(path))
        axs[row,col].imshow(render(V,VT,F,FVT,tex,25,55,px=440),origin='lower')
        axs[row,col].axis('off');axs[row,col].set_title(item['name']+(' / Original' if col==0 else ' / Exported OBJ'))
    V,VT,F,FVT=read_obj(source);V=V[:,axes];P=V-(V.min(0)+V.max(0))/2
    r,u,f=view_basis(25,55);ids=np.linspace(0,len(V)-1,min(17000,len(V))).astype(int)
    ax=axs[row,1];ax.scatter(P[ids]@r,P[ids]@u,c='#c9d1da',s=.5)
    ids=ids[weights[ids]>.3];ax.scatter(P[ids]@r,P[ids]@u,c='#ee8222',s=1)
    ax.axis('equal');ax.axis('off');ax.set_title('Automatic target / orange')
fig.suptitle('Real outputs | no region boxes, manual axes or semantic cluster IDs supplied\nImages auto-fit individually; dimensions must be read from numeric reports',fontsize=13)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/'automatic_parts_overview.png',dpi=130);plt.close(fig)
print(root/'automatic_parts_overview.png',flush=True)
