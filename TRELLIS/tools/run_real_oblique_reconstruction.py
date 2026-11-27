"""Reconstruct selected real photos with the existing offline TRELLIS pipeline."""
import sys,json,time,traceback,hashlib
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,'/root/TRELLIS')
import reconstruct_cli as cli
import numpy as np
from PIL import Image


class Tee:
 def __init__(self,*streams):self.streams=streams
 def write(self,s):
  for stream in self.streams:stream.write(s);stream.flush()
 def flush(self):
  for stream in self.streams:stream.flush()
 def isatty(self):return False


root=Path(sys.argv[1]);log=(root/'reconstruction.log').open('w',buffering=1)
sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
config={'seed':42,'ss_steps':24,'ss_guidance':7.5,'slat_steps':24,'slat_guidance':3.,
        'simplify':.5,'texture_size':2048,'formats':['mesh','gaussian'],'no_preprocess':True}
(root/'reconstruction_config.json').write_text(json.dumps(config,indent=2))
pipe=cli.load_pipeline();records=[]
for name,cutout in [('aircraft','aircraft_u2net.png'),('ship','ship_isnet-general-use.png')]:
 start=time.time();folder=root/'reconstruction'/name;folder.mkdir(parents=True,exist_ok=False)
 record={'name':name,'cutout':str(root/'inputs'/cutout),'config':config,'status':'running'}
 try:
  rgba=Image.open(root/'inputs'/cutout).convert('RGBA');a=np.array(rgba)
  a[a[:,:,3]<128,3]=0
  prepared=Image.fromarray(a,'RGBA');prepared.save(folder/'foreground_rgba.png')
  # Exactly the shipped foreground crop/resize, recorded before inference.
  condition=pipe.preprocess_image(prepared);condition.save(folder/'conditioning_518.png')
  args=Namespace(**config)
  out,_=cli.generate(pipe,str(folder/'conditioning_518.png'),args)
  raw=out['mesh'][0]
  np.savez_compressed(folder/'raw_mesh.npz',vertices=raw.vertices.detach().cpu().numpy(),faces=raw.faces.detach().cpu().numpy())
  record.update(cli.export_one(out,str(folder),args));record['status']='ok'
 except Exception as exc:
  traceback.print_exc();record.update(status='error',error=type(exc).__name__+': '+str(exc))
 record['elapsed_seconds']=time.time()-start;records.append(record)
 (root/'reconstruction_summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 print('RECONSTRUCTION_RESULT',json.dumps(record,ensure_ascii=False),flush=True)
print('DONE',flush=True)
