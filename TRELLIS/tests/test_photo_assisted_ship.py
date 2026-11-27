"""Validate the same new warship with and without photograph evidence."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit
from photo_direction_bridge import asset_hash

root=Path(sys.argv[1]);evidence_dir=root/'photo_joint_direction_v3'
evidence=json.loads((evidence_dir/'direction_evidence.json').read_text())
assert evidence['status']=='ok'
source=root/'reconstruction/ship/sample.obj';photo=root/'reconstruction/ship/foreground_rgba.png'
data=edit._read_obj(source)
geometry=hashlib.sha256(np.asarray(data.vertices,'<f8').tobytes()+np.asarray(data.faces,'<i8').tobytes()).hexdigest()
ph=hashlib.sha256(photo.read_bytes()).hexdigest();appearance=asset_hash(source)
assert evidence['mesh_geometry_sha256']==geometry and evidence['photo_sha256']==ph and evidence['mesh_appearance_sha256']==appearance
key=hashlib.sha256((geometry+ph+appearance+'photo_direction_v1').encode()).hexdigest()
cache=root/'photo_direction_cache';target=cache/key
if not target.exists():shutil.copytree(evidence_dir,target)
out=root/(sys.argv[2] if len(sys.argv)>2 else 'photo_assisted_edits');out.mkdir(exist_ok=False);rows=[]
for name,text in [('ship_bow','船头加宽 5%'),('ship_stern','船尾缩短 5%')]:
 baseline=edit.plan_edit(str(source),text,{'auto_localize':True})
 assert baseline['code']==5,baseline
 hint={'auto_localize':True,'source_photo':str(photo),'photo_direction_cache_dir':str(cache),
       'dimension_mode':'core_extent','self_intersection_check':True}
 row={'name':name,'source':str(source),'instruction':text,'hint':hint,'baseline_without_photo':baseline,
      'manual_region_or_direction':None}
 delta,w,meta=edit._plan(data,edit.parse_instruction(text),dict(hint,_source_obj_path=str(source)))
 np.savez_compressed(out/(name+'_selection.npz'),weights=w,delta=delta)
 row['localization']=meta
 row['result']=edit.edit_obj(str(source),text,str(out/name),hint)
 rows.append(row);(out/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
 print(name,row['result']['code'],row['result'].get('message'),flush=True)
 if row['result']['code']==0:
  edited=edit._read_obj(Path(row['result']['obj_path']));axis='xyz'.index(meta['edit_axis'])
  measured=w>=.9*w.max();actual=np.ptp(edited.vertices[measured,axis])/np.ptp(data.vertices[measured,axis])-1
  assert abs(actual-meta['requested_relative_change'])<1e-6
  row['independent_serialized_core_change']=float(actual)
(out/'summary.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2))
print(out/'summary.json',flush=True)
