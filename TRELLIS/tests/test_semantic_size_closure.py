"""CPU regression tests on real meshes; no synthetic metric results."""
import sys, json, hashlib, time, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/TRELLIS/trellis/utils')
import mesh_text_edit as edit


def main():
    root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False)
    cases=[]
    def record(name,result,expected):
        passed=result['code']==expected
        cases.append({'name':name,'expected_code':expected,'passed':passed,'result':result})
        print(name, 'PASS' if passed else 'FAIL', result.get('message',''), flush=True)
    for text,target,operation,preserve in [
        ('把船头变宽一些，但甲板别动','bow','widen',['deck']),
        ('右侧机翼拉长 8%','right_wing','lengthen',[]),
        ('船尾缩短 5%','stern','shorten',[]),
        ('主桅稍微高一点','mast','heighten',[]),
        ('让机鼻窄一些','nose','narrow',[])]:
        r=edit.parse_instruction(text)
        assert (r.target,r.operation,r.preserve)==(target,operation,preserve),r
        record('parse:'+text,{'code':0,'parsed':vars(r)},0)
    for text in ('不要把船头加宽','船头加宽，船尾缩短','船头加宽但船头别动','船头加宽 -8%','更威武一点'):
        try: edit.parse_instruction(text); r={'code':0}
        except edit.EditError as e: r={'code':e.code,'message':str(e)}
        record('refuse:'+text,r,1)
    source='/root/autodl-tmp/PartField/data/ship_d04/ship_d04.obj'
    hint={'long_axis':'x','lateral_axis':'z','up_axis':'y','bow_sign':1,'up_sign':1,
          'accept_geometric_selection':True}
    for name,text in [('ship_widen','船头加宽 5%'),('ship_narrow','船头变窄 5%'),('ship_lengthen','船头加长 5%'),('ship_shorten','船尾缩短 5%')]:
        result=edit.edit_obj(source,text,str(root/name),hint)
        record(name,result,0)
    unconfirmed=dict(hint);unconfirmed.pop('accept_geometric_selection')
    record('requires_confirmation',edit.edit_obj(source,'船头加宽 5%',str(root/'refused'),unconfirmed),5)
    record('requires_preserve_mask',edit.edit_obj(source,'船头加宽但甲板别动',str(root/'refused_preserve'),hint),5)
    data=edit._read_obj(Path(source))
    # Explicit spatial constraint, NOT a manually validated semantic deck label.
    protected=data.vertices[:,1]>np.quantile(data.vertices[:,1],.75)
    digest=hashlib.sha256(np.asarray(data.vertices,'<f8').tobytes()+np.asarray(data.faces,'<i8').tobytes()).hexdigest()
    np.savez(root/'spatial_constraint.npz',deck=protected.astype(float),mesh_geometry_sha256=np.array(digest))
    constrained=dict(hint,part_masks_npz=str(root/'spatial_constraint.npz'))
    r=edit.edit_obj(source,'船头加宽 5%，甲板别动',str(root/'ship_protected'),constrained)
    if r['code']==0:
        output=edit._read_obj(Path(r['obj_path']))
        assert np.array_equal(output.vertices[protected],data.vertices[protected])
    record('spatial_preserve_constraint',r,0)
    plane='/root/autodl-tmp/PartField/data/trellis_samples2/f22.obj'
    ph={'long_axis':'z','lateral_axis':'x','up_axis':'y','front_sign':1,'right_sign':1}
    record('f22_right_wing_preview',edit.plan_edit(plane,'右翼加宽 8%',ph),0)
    (root/'verified_cluster_mapping.json').write_text(json.dumps({'right_wing':[1926]}))
    subprocess.run([sys.executable,'/root/TRELLIS/tools/prepare_part_masks.py',
        '--input-obj',plane,'--face-labels','/root/autodl-tmp/PartField/exp_results/clustering/f22/cluster_out/f22_0_05.npy',
        '--part-clusters-json',str(root/'verified_cluster_mapping.json'),
        '--output',str(root/'f22_part_masks.npz')],check=True)
    record('f22_partfield_mask_preview',edit.plan_edit(plane,'右翼加宽 8%',dict(ph,part_masks_npz=str(root/'f22_part_masks.npz'))),0)
    record('reject_wrong_mesh_mask',edit.plan_edit(source,'船头加宽 5%',dict(hint,part_masks_npz=str(root/'f22_part_masks.npz'))),5)
    summary={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'cases':cases,
             'passed':sum(x['passed'] for x in cases),'total':len(cases),
             'note':'spatial constraint is not a semantic GT mask; F22 preview is not a textured export test'}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('SUMMARY',summary['passed'],'/',summary['total'])
    return int(summary['passed']!=summary['total'])


if __name__=='__main__':raise SystemExit(main())
