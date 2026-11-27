import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit
root=Path(sys.argv[1]);out=root/(sys.argv[2] if len(sys.argv)>2 else 'large_edits_v1');out.mkdir(exist_ok=False);rows=[]
cases=[('ship_bow20','ship','船头加宽 20%'),('ship_bow15','ship','船头加宽 15%'),
       ('ship_stern20','ship','船尾缩短 20%'),('ship_stern10','ship','船尾缩短 10%'),
       ('ship_mast25','ship','桅杆拉长 25%'),
       ('aircraft_nose20','aircraft','机头加长 20%'),('aircraft_right_wing20','aircraft','右翼拉长 20%'),
       ('aircraft_left_wing20','aircraft','左翼拉长 20%')]
for name,model,text in cases:
 source=root/'reconstruction'/model/'sample.obj'
 hint={'auto_localize':True,'dimension_mode':'core_extent','self_intersection_check':True}
 if len(sys.argv)>3:hint['deformation_solver']=sys.argv[3]
 if model=='ship':hint.update(source_photo=str(root/'reconstruction/ship/foreground_rgba.png'),photo_direction_cache_dir=str(root/'photo_direction_cache'))
 row={'name':name,'source':str(source),'instruction':text,'hint':hint}
 try:
  data=edit._read_obj(source);delta,w,meta=edit._plan(data,edit.parse_instruction(text),dict(hint,_source_obj_path=str(source)))
  row['localization']=meta;np.savez_compressed(out/(name+'_selection.npz'),weights=w,delta=delta)
  row['result']=edit.edit_obj(str(source),text,str(out/name),hint)
 except edit.EditError as e:row['result']={'code':e.code,'message':str(e)}
 rows.append(row);(out/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
 print(name,row['result']['code'],row['result'].get('message'),flush=True)
