import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/autodl-tmp/PartField')
from render_textured import read_obj,render,load_texture

root=Path(sys.argv[1]);rows=json.loads((root/'reconstruction_summary.json').read_text())
fig,axes=plt.subplots(len(rows),5,figsize=(18,4*len(rows)),squeeze=False)
for i,item in enumerate(rows):
 name=item['name'];folder=root/'reconstruction'/name
 original=root/'inputs'/('aircraft_original.JPG' if name=='aircraft' else 'ship_source.png')
 for col,p,title in [(0,original,'Real photograph'),(1,folder/'conditioning_518.png','Actual TRELLIS conditioning')]:
  axes[i,col].imshow(Image.open(p));axes[i,col].set_title(name+' / '+title);axes[i,col].axis('off')
 if item['status']!='ok':continue
 obj=Path(item['obj']);V,VT,F,FVT=read_obj(obj);V=V[:,[0,2,1]]
 tex=load_texture(str(obj.parent/'sample.mtl'),str(obj.parent))
 if tex is None:raise RuntimeError('missing texture')
 for col,(elev,azim) in enumerate([(20,30),(20,150),(70,90)],2):
  axes[i,col].imshow(render(V,VT,F,FVT,tex,elev,azim,px=360),origin='lower')
  axes[i,col].set_title('Reconstructed OBJ / '+str(col-1));axes[i,col].axis('off')
fig.tight_layout();fig.savefig(root/'reconstruction_audit.png',dpi=130);plt.close(fig)
print(root/'reconstruction_audit.png',flush=True)
