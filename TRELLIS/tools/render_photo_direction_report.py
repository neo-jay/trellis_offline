import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit
root=Path(sys.argv[1]);ev=root/'photo_joint_direction_v3';record=json.loads((ev/'direction_evidence.json').read_text())
rows=json.loads((root/'photo_assisted_edits_v2/summary.json').read_text())['rows']
mesh=edit._read_obj(Path(record['mesh']));v=mesh.vertices
frame=rows[0]['localization']['automatic_localization']['frame'];long,lat,up=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
with np.load(root/'photo_assisted_edits_v2/ship_bow_selection.npz') as a:bow=a['weights']
with np.load(root/'photo_assisted_edits_v2/ship_stern_selection.npz') as a:stern=a['weights']
fig,axs=plt.subplots(2,2,figsize=(13,8))
image=np.array(Image.open(ev/'semantics/aligned_photo.png').convert('RGB'))
axs[0,0].imshow(image);axs[0,0].axis('off')
names=['STERN','BOW'] if record['photo_semantics']['bow_end']==1 else ['BOW','STERN']
axs[0,0].set_title('Original photo / automatic end semantics')
for x,name,color in ((.1,names[0],'#139b9b'),(.9,names[1],'#eb8026')):
 axs[0,0].text(x,.04,name,color=color,fontsize=14,weight='bold',ha='center',transform=axs[0,0].transAxes,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
axs[0,1].imshow(Image.open(ev/'best_photo_model_matches.jpg'));axs[0,1].axis('off');axs[0,1].set_title('DINOv2 verified photo-model matches')
ids=np.linspace(0,len(v)-1,min(17000,len(v))).astype(int)
for ax,a,b,title in [(axs[1,0],long,lat,'3D model / top'),(axs[1,1],long,up,'3D model / side')]:
 ax.scatter(v[ids,a],v[ids,b],s=.5,c='#c8d0d9')
 selected=ids[bow[ids]>.3];ax.scatter(v[selected,a],v[selected,b],s=1,c='#eb8026',label='Automatic bow')
 selected=ids[stern[ids]>.3];ax.scatter(v[selected,a],v[selected,b],s=1,c='#139b9b',label='Automatic stern')
 ax.axis('equal');ax.grid(alpha=.15);ax.set_title(title);ax.legend(markerscale=4)
fig.suptitle('Photo semantics + multi-view visual correspondence -> 3D direction -> PartField parts\nNo manually supplied direction, region, landmarks or cluster IDs',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'photo_assisted_direction.png',dpi=130);plt.close(fig)
print(root/'photo_assisted_direction.png')
