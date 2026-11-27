"""Audit foreground separation on real photos; preserve source files."""
import os,json,sys
from pathlib import Path
os.environ['U2NET_HOME']='/root/.u2net'
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import rembg

root=Path(sys.argv[1]);inputs=root/'inputs'
sessions={m:rembg.new_session(m,providers=['CPUExecutionProvider']) for m in ('u2net','isnet-general-use')}
rows=[];canvas=Image.new('RGB',(1200,680),'#e7e9ed');draw=ImageDraw.Draw(canvas)
for row,(name,file) in enumerate([('aircraft','aircraft_original.JPG'),('ship','ship_source.png')]):
 im=Image.open(inputs/file).convert('RGB')
 thumb=ImageOps.contain(im,(380,290));canvas.paste(thumb,(10,row*340+35));draw.text((10,row*340+10),name+' original',fill='black')
 for col,(model,session) in enumerate(sessions.items(),1):
  out=rembg.remove(im,session=session)
  path=inputs/(name+'_'+model+'.png');out.save(path)
  alpha=np.array(out.getchannel('A'))
  bg=Image.new('RGBA',out.size,'#adb8c7');bg.alpha_composite(out)
  thumb=ImageOps.contain(bg.convert('RGB'),(380,290));canvas.paste(thumb,(col*400+10,row*340+35));draw.text((col*400+10,row*340+10),name+' '+model,fill='black')
  rows.append({'source':str(inputs/file),'cutout':str(path),'model':model,'alpha_nonzero_fraction':float((alpha>0).mean()),'alpha_foreground_fraction':float((alpha>200).mean())})
  print(rows[-1],flush=True)
canvas.save(root/'foreground_audit.jpg');(root/'foreground_audit.json').write_text(json.dumps(rows,indent=2))
