import os,sys,json,time,traceback
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,'/root/TRELLIS')
import reconstruct_cli as cli
import numpy as np
from PIL import Image
import rembg

root=Path(sys.argv[1]);root.mkdir(exist_ok=False);source=Path(sys.argv[2])
config=dict(seed=42,ss_steps=24,ss_guidance=7.5,slat_steps=24,slat_guidance=3.,simplify=.5,texture_size=2048,formats=['mesh','gaussian'],no_preprocess=True)
(root/'config.json').write_text(json.dumps(dict(config,input_image=str(source)),indent=2))
os.environ['U2NET_HOME']='/root/.u2net'
image=Image.open(source).convert('RGB');image.save(root/'original.jpg')
rgba=rembg.remove(image,session=rembg.new_session('u2net',providers=['CPUExecutionProvider']))
a=np.array(rgba);a[a[:,:,3]<128,3]=0;rgba=Image.fromarray(a,'RGBA');rgba.save(root/'foreground_rgba.png')
pipe=cli.load_pipeline();condition=pipe.preprocess_image(rgba);condition.save(root/'conditioning_518.png')
out,_=cli.generate(pipe,str(root/'conditioning_518.png'),Namespace(**config))
np.savez_compressed(root/'raw_mesh.npz',vertices=out['mesh'][0].vertices.cpu().numpy(),faces=out['mesh'][0].faces.cpu().numpy())
result=cli.export_one(out,str(root),Namespace(**config));(root/'reconstruction_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
