"""Actual-model precision and safety matrix, CPU only."""
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit


root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
rows=[]
models=[('d04','/root/autodl-tmp/PartField/data/ship_d04/ship_d04.obj',
         {'long_axis':'x','lateral_axis':'z','up_axis':'y','bow_sign':1,'up_sign':1}),
        ('ship_previous','/root/TRELLIS/tmp/pvsaon32es/sample.obj',
         {'long_axis':'z','lateral_axis':'x','up_axis':'y','bow_sign':1,'up_sign':1})]
for name,source,frame in models:
    for op,text in [('widen','船头加宽 5%'),('narrow','船头变窄 5%'),('lengthen','船头加长 8%'),('shorten','船尾缩短 8%')]:
        hint=dict(frame,accept_geometric_selection=True,dimension_mode='core_extent')
        result=edit.edit_obj(source,text,str(root/(name+'_'+op)),hint)
        row={'name':name+'_'+op,'source':source,'hint':hint,'instruction':text,'result':result}
        if result['code']==0:
            report=json.loads(Path(result['quality_path']).read_text())
            # Independently recompute with the original fixed core vertex set,
            # using the serialized output instead of trusting reported extents.
            data=edit._read_obj(Path(source));out=edit._read_obj(Path(result['obj_path']))
            request=edit.parse_instruction(text);_,w,meta=edit._plan(data,request,hint)
            mask=w>=.9*w.max();axis='xyz'.index(meta['edit_axis'])
            measured=float(np.ptp(out.vertices[mask,axis])/np.ptp(data.vertices[mask,axis])-1)
            requested=meta['requested_relative_change']
            assert abs(measured-requested)<1e-6,(row['name'],measured,requested)
            assert report['deformation_checks']['face_flips']==0
            row['independently_measured_change']=measured
        rows.append(row);print(row['name'],result['code'],result.get('message'),row.get('independently_measured_change'),flush=True)
# Explicitly conflicting measurement: every measurement vertex is protected.
source=models[0][1];data=edit._read_obj(Path(source))
hint=dict(models[0][2],accept_geometric_selection=True,dimension_mode='core_extent',
          measurement_vertex_indices=[0,1,2],preserve_bboxes={'deck':[[float(x)-1 for x in data.vertices.min(0)],[float(x)+1 for x in data.vertices.max(0)]]})
rejected=edit.edit_obj(source,'船头加宽 5%，甲板别动',str(root/'impossible'),hint)
assert rejected['code'] in (5,9) and not (root/'impossible').exists()
rows.append({'name':'conflicting_constraints_refused','result':rejected})
frame=dict(models[0][2],accept_geometric_selection=True)
_,weights,meta=edit._plan(data,edit.parse_instruction('船头加宽 5%'),frame)
outside=np.flatnonzero(weights==0);axis='xyz'.index(meta['edit_axis'])
indices=[int(outside[np.argmin(data.vertices[outside,axis])]),int(outside[np.argmax(data.vertices[outside,axis])])]
impossible=edit.edit_obj(models[0][1],'船头加宽 5%',str(root/'unreachable_measurement'),
                         dict(frame,dimension_mode='core_extent',measurement_vertex_indices=indices))
assert impossible['code']==9 and not (root/'unreachable_measurement').exists()
rows.append({'name':'fixed_measurement_unreachable_refused','result':impossible})
# Real geometry, verified existing PartField cluster; numerical preview only.
source='/root/autodl-tmp/PartField/data/trellis_samples2/f22.obj'
hint={'long_axis':'z','lateral_axis':'x','up_axis':'y','right_sign':1,'front_sign':1,
      'part_masks_npz':'/root/autodl-tmp/data/semantic_edit_tests/codex_closure_20261126_v2/f22_part_masks.npz',
      'dimension_mode':'core_extent'}
data=edit._read_obj(Path(source));req=edit.parse_instruction('右翼拉长 5%')
delta,weights,meta=edit._plan(data,req,hint)
assert meta['edit_axis']=='x'
assert np.all(delta[:,1:]==0)
assert abs(meta['dimension_control']['achieved_relative_change']-.05)<1e-6
rows.append({'name':'f22_wing_length_axis','preview_only':True,'region':meta})
# Real ship spatial region used only to validate operation-axis dispatch,
# not as semantic ground truth for a mast.
frame=dict(models[1][2],bbox_fraction=[[.1,.6,.1],[.9,1,.9]])
d=edit._read_obj(Path(models[1][1]));delta,w,meta=edit._plan(d,edit.parse_instruction('桅杆拉长 5%'),frame)
assert meta['edit_axis']=='y' and np.all(delta[:,[0,2]]==0)
rows.append({'name':'mast_length_axis_dispatch','spatial_proxy_not_semantic_GT':True,'region':meta})
# Self-intersection worker on a real mesh, no direct native call in the parent.
for name,source,frame in models:
    hint=dict(frame,accept_geometric_selection=True,dimension_mode='core_extent',self_intersection_check=True)
    result=edit.edit_obj(source,'船头加长 5%',str(root/(name+'_intersection')),hint)
    if result['code']==0:
        report=json.loads(Path(result['quality_path']).read_text())
        assert report['deformation_checks']['self_intersections_checked']
        assert report['deformation_checks']['new_intersecting_faces']==0
    rows.append({'name':name+'_intersection','source':source,'hint':hint,'result':result})
    print(name+'_intersection',result['code'],result.get('message'),flush=True)
summary={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'rows':rows,
         'successful_exports':sum(x.get('result',{}).get('code')==0 for x in rows),
         'note':'Safety refusals are recorded, not relabeled as successful edits; geometric proxies are not semantic GT.'}
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('saved',root/'summary.json',flush=True)
