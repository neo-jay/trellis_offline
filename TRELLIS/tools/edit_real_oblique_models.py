"""Text-only automatic localization/editing of this run's fresh TRELLIS outputs."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit

root=Path(sys.argv[1]);out=root/'edits';out.mkdir(exist_ok=False);rows=[]
cases={'aircraft':[('nose','机头加长 5%'),('right_wing','右翼拉长 5%'),('left_wing','左翼拉长 5%')],
       'ship':[('bow','船头加宽 8%'),('stern','船尾缩短 5%'),('mast','桅杆拉长 5%')]}
for model,requests in cases.items():
 source=root/'reconstruction'/model/'sample.obj'
 for part,text in requests:
  name=model+'_'+part;hint={'auto_localize':True,'dimension_mode':'core_extent','self_intersection_check':True}
  row={'name':name,'source':str(source),'instruction':text,'hint':hint,'manual_region':None}
  try:
   data=edit._read_obj(source);request=edit.parse_instruction(text)
   delta,w,meta=edit._plan(data,request,hint)
   row['localization']=meta
   np.savez_compressed(out/(name+'_selection.npz'),weights=w,delta=delta)
   row['result']=edit.edit_obj(str(source),text,str(out/name),hint)
  except edit.EditError as exc:row['result']={'code':exc.code,'message':str(exc),'status':'refused'}
  rows.append(row);(out/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
  print(name,row['result']['code'],row['result'].get('message'),flush=True)
  if 'localization' in row:print(json.dumps(row['localization']['automatic_localization'],ensure_ascii=False),flush=True)
