import sys,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit
p=Path(sys.argv[1]);d=edit._read_obj(p);v=d.vertices;f=d.faces
_,inv=np.unique(np.round(v/1e-8).astype('int64'),axis=0,return_inverse=True);ff=inv[f]
edges=np.vstack((ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]));n=inv.max()+1
graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n)).tocsr()
count,labels=connected_components(graph,directed=False);labels=labels[inv]
rows=[]
for label in range(count):
 pts=v[labels==label]
 if len(pts)<15:continue
 span=np.ptp(pts,axis=0)
 if span[2]>.1 and max(span[:2])<.10:
  rows.append({'id':int(label),'vertices':len(pts),'span':span.tolist(),'center':pts.mean(0).tolist(),'bounds':[pts.min(0).tolist(),pts.max(0).tolist()]})
print('components',count);print(json.dumps(rows,indent=2))
