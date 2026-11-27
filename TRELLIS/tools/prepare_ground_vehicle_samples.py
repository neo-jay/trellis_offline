import sys,json
from pathlib import Path
import trimesh,numpy as np
root=Path(sys.argv[1]);root.mkdir(exist_ok=False);rows=[]
base=Path('/root/autodl-tmp/trellis_acceptance/datasets/objv')
for name,folder,category in [('tank','m60a3-patton-medium-tank-toshueyi-3d298af8','tank'),
 ('tank_panzer','panzer-iv-medium-tank-toshueyi-14c74d14','tank'),
 ('truck','gaz-66-offroad-truck-1cc3374e','truck'),
 ('armored','low-poly-humvee-vehicle-fac4178d','car')]:
 source=base/folder/'model.glb';m=trimesh.load(source,force='mesh',process=False)
 dest=root/name;dest.mkdir();m.export(dest/'sample.obj')
 row={'name':name,'source_glb':str(source),'obj':str(dest/'sample.obj'),'category':category,
      'vertices':len(m.vertices),'faces':len(m.faces),'bounds':m.bounds.tolist(),'std':np.std(m.vertices,axis=0).tolist(),
      'data_type':'existing Objaverse asset, NOT a newly reconstructed real-photo sample'}
 rows.append(row);print(row,flush=True)
(root/'samples.json').write_text(json.dumps(rows,indent=2))
