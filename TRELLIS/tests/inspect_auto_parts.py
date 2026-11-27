import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as m
paths=['/root/autodl-tmp/PartField/data/ship_d04/ship_d04.obj','/root/TRELLIS/tmp/pvsaon32es/sample.obj','/root/autodl-tmp/PartField/data/trellis_samples2/f22.obj','/root/autodl-tmp/PartField/data/mig21/mig21.obj']
for path in paths:
 d=m._read_obj(Path(path));v=d.vertices
 print(path,'spans',np.ptp(v,axis=0),'std',np.std(v,axis=0),'quantiles',np.quantile(v,[.01,.1,.5,.9,.99],axis=0),flush=True)
 for a in range(3):
  p=(v[:,a]-v[:,a].min())/np.ptp(v[:,a]);out=[]
  for lo in np.arange(0,1,.1):
   t=v[(p>=lo)&(p<lo+.1)]
   out.append(np.round(np.ptp(t,axis=0),3).tolist() if len(t) else [])
  print('axis',a,out,flush=True)
