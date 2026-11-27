import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit
root=Path(sys.argv[1]);out=root/(sys.argv[2] if len(sys.argv)>2 else 'edits_v1');out.mkdir(exist_ok=False);rows=[]
cases=[('tank_turret20','tank','坦克炮塔加宽 20%'),('tank_barrel20','tank','坦克炮管拉长 20%'),
       ('tank_tracks15','tank','坦克履带总成加长 15%'),
       ('panzer_turret20','tank_panzer','坦克炮塔加宽 20%'),
       ('truck_body20','truck','卡车车身加宽 20%'),('truck_body_height15','truck','卡车车身增高 15%'),
       ('armored_body20','armored','装甲车车身加宽 20%')]
if len(sys.argv)>2:
 cases=[('tank_barrel20','tank','坦克炮管拉长 20%'),('tank_whole20','tank','坦克整车加宽 20%'),
        ('truck_whole20','truck','卡车整车加宽 20%'),('armored_whole20','armored','装甲车整车加宽 20%'),
        ('pickup_whole25','real_pickup','汽车整车加宽 25%'),('pickup_body20','real_pickup','汽车车身加宽 20%'),
        ('pickup_height20','real_pickup','汽车车身增高 20%')]
for name,model,text in cases:
 source=root/model/'sample.obj';hint={'auto_localize':True,'dimension_mode':'core_extent','deformation_solver':'harmonic_transition','self_intersection_check':True}
 row={'name':name,'source':str(source),'instruction':text,'hint':hint,
      'data_type':'real-photo TRELLIS reconstruction' if model=='real_pickup' else 'existing Objaverse asset, not a new real-photo reconstruction','manual_region':None}
 try:
  d=edit._read_obj(source);request=edit.parse_instruction(text)
  delta,w,meta=edit._plan(d,request,hint)
  np.savez_compressed(out/(name+'_selection.npz'),weights=w,delta=delta)
  row['localization']=meta;row['result']=edit.edit_obj(str(source),text,str(out/name),hint)
 except edit.EditError as exc:row['result']={'code':exc.code,'message':str(exc)}
 rows.append(row);(out/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
 print(name,row['result']['code'],row['result'].get('message'),flush=True)
print(out/'summary.json',flush=True)
