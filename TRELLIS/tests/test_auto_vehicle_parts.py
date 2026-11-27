"""Real vehicle localization/export diagnostics; no pre-labelled region inputs."""
import sys,json,time
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit


root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
aircraft=root/'f22_input';aircraft.mkdir()
mesh=trimesh.load('/root/autodl-tmp/PartField/data/trellis_samples2/f22.glb',force='mesh',process=False)
mesh.export(aircraft/'f22.obj')
models=[('ship_previous','/root/TRELLIS/tmp/pvsaon32es/sample.obj',['船头加宽 8%','船尾缩短 5%','桅杆拉长 8%']),
        ('f22',str(aircraft/'f22.obj'),['右翼拉长 8%','左翼拉长 8%','机头加长 8%','尾翼增高 8%','左翼拉长 3%']),
        ('ship_d04','/root/autodl-tmp/PartField/data/ship_d04/ship_d04.obj',['船头加宽 8%']),
        ('mig21','/root/autodl-tmp/PartField/data/mig21/mig21.obj',['右翼拉长 8%'])]
rows=[]
for name,source,instructions in models:
 for i,text in enumerate(instructions):
  tag=f'{name}_{i}';hint={'auto_localize':True}
  row={'name':tag,'source':source,'instruction':text,'hint':hint,'region_or_cluster_input':None}
  try:
   data=edit._read_obj(Path(source));request=edit.parse_instruction(text)
   delta,w,meta=edit._plan(data,request,hint)
   row['localization']=meta
   np.savez_compressed(root/(tag+'_selection.npz'),weights=w,delta=delta)
   result=edit.edit_obj(source,text,str(root/tag),hint)
   row['result']=result
   # Visual diagnostics: surface sample points projected to body axes, with
   # original target selection and actual proposed displacement overlaid.
   from auto_vehicle_parts import samples
   frame=meta['automatic_localization']['frame']
   axes=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
   long,lat,up=axes
   # Deterministic vertex subsample for transparent selection visualization.
   ids=np.unique(np.linspace(0,len(data.vertices)-1,min(15000,len(data.vertices))).astype(int))
   v=data.vertices;new=v+delta
   fig,axs=plt.subplots(2,3,figsize=(13,7))
   for col,(a,b,label) in enumerate([(long,lat,'Top'),(long,up,'Side'),(lat,up,'Front')]):
    ax=axs[0,col]
    ax.scatter(v[ids,a],v[ids,b],s=.5,c='#c8d0d9',rasterized=True)
    selected=ids[w[ids]>.3]
    ax.scatter(v[selected,a],v[selected,b],s=1,c='#ee8222',rasterized=True)
    ax.set_title(label+' / auto selected orange');ax.axis('equal');ax.grid(alpha=.15)
    ax=axs[1,col]
    ax.scatter(v[ids,a],v[ids,b],s=.5,c='#8a98a6',label='Original',rasterized=True)
    ax.scatter(new[selected,a],new[selected,b],s=1,c='#13a5a2',label='Edited target',rasterized=True)
    ax.set_title(label+' / proposed edit');ax.axis('equal');ax.grid(alpha=.15)
   fig.suptitle(tag+' | '+request.target+' '+request.operation+' '+str(request.magnitude)+' | export code '+str(result['code']))
   axs[1,0].legend(markerscale=4);fig.tight_layout();fig.savefig(root/(tag+'_localization.png'),dpi=130);plt.close(fig)
   row['visualization']=str(root/(tag+'_localization.png'))
   row['visualization_note']='proposed coordinates; exported model exists only when result.code=0; no semantic GT'
  except edit.EditError as exc:
   row['result']={'code':exc.code,'message':str(exc),'status':'refused'}
  rows.append(row);(root/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
  print(tag,text,row['result']['code'],row['result'].get('message'),flush=True)
  if 'localization' in row:print(json.dumps(row['localization']['automatic_localization'],ensure_ascii=False),flush=True)
print(root/'summary.json')
