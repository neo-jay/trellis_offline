"""Re-use existing CPU texture rasterizer for actual before/after results."""
import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/autodl-tmp/PartField')
from render_textured import read_obj,render,load_texture


root=Path(sys.argv[1])
original=Path('/root/autodl-tmp/PartField/data/ship_d04/ship_d04.obj')
items=[('Original',original),('Bow wider 5%',root/'ship_widen/edited.obj'),
       ('Bow narrower 5%',root/'ship_narrow/edited.obj'),('Bow longer 5%',root/'ship_lengthen/edited.obj'),
       ('Stern shorter 5%',root/'ship_shorten/edited.obj'),('Fixed upper region',root/'ship_protected/edited.obj')]
fig,axes=plt.subplots(len(items),3,figsize=(12,3*len(items)))
for row,(name,obj) in enumerate(items):
    V,VT,F,FVT=read_obj(obj)
    mtl=next(x.split(None,1)[1].strip() for x in obj.read_text().splitlines() if x.startswith('mtllib '))
    tex=load_texture(str(obj.parent/mtl),str(obj.parent))
    if tex is None:raise RuntimeError('missing texture: '+str(obj))
    # Renderer is Z-up; source is Y-up. Apply same display rotation only.
    V=V[:,[0,2,1]]
    for col,(el,az) in enumerate([(15,60),(35,130),(75,90)]):
        axes[row,col].imshow(render(V,VT,F,FVT,tex,el,az,px=420))
        axes[row,col].axis('off');axes[row,col].set_title(name+' / view '+str(col+1))
fig.tight_layout();fig.savefig(root/'textured_comparison.png',dpi=130);plt.close(fig)
print(root/'textured_comparison.png')
