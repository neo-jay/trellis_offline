"""Conservative vehicle part localization without text/VLM service.

Geometry supplies category/position priors; existing PartField multi-resolution
clusters refine these priors when their source geometry can be verified.
Scores are heuristic fit measures, NOT calibrated semantic probabilities.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


def smooth(x):
    t=np.clip(x,0,1);return t*t*(3-2*t)


def samples(v,f,n=12000):
    rng=np.random.default_rng(271828)
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    ids=rng.choice(len(f),n,p=area/area.sum())
    a=rng.random((n,1));b=rng.random((n,1));a=np.sqrt(a)
    return (1-a)*tri[ids,0]+a*(1-b)*tri[ids,1]+a*b*tri[ids,2]


def infer_frame(v,f,category,h,error):
    p=samples(v,f);span=np.ptp(v,axis=0);diag=np.linalg.norm(span)
    tree=cKDTree(p);sym=[];centers=[]
    for axis in range(3):
        options=[np.median(p[:,axis]),np.mean(np.quantile(p[:,axis],[.05,.95])),np.mean([v[:,axis].min(),v[:,axis].max()])]
        scores=[]
        for center in options:
            q=p.copy();q[:,axis]=2*center-q[:,axis]
            d=tree.query(q,workers=1)[0]
            scores.append(np.mean(np.minimum(d,np.quantile(d,.9)))/diag)
        j=int(np.argmin(scores));sym.append(scores[j]);centers.append(float(options[j]))
    if all(k in h for k in ('long_axis','lateral_axis','up_axis')):
        axes=['xyz'.index(h[k]) for k in ('long_axis','lateral_axis','up_axis')]
        if len(set(axes))!=3:raise error('坐标轴不可重复',5)
        long,lat,up=axes;axis_source='provided_frame_no_region'
    elif category in ('ship','car','truck','tank','armored'):
        long=int(np.argmax(np.std(p,axis=0)))
        if span[long]/np.partition(span,-2)[-2]<(1.3 if category=='ship' else 1.1):raise error('载具长轴不明确；无需手工框部件但需确认朝向',5)
        rem=[a for a in range(3) if a!=long];lat=min(rem,key=lambda a:sym[a]);up=next(a for a in rem if a!=lat)
        axis_source='surface_symmetry_and_vehicle_long_axis'
    else:
        up=int(np.argmin(np.std(p,axis=0)));rem=[a for a in range(3) if a!=up]
        lat=min(rem,key=lambda a:sym[a]);long=next(a for a in rem if a!=lat)
        axis_source='surface_symmetry_and_aircraft_thin_axis'
    if category=='aircraft' and span[up]>.6*max(span[long],span[lat]):
        raise error('飞机薄轴/姿态证据不足，未自动定位',5)
    if category=='aircraft' and sym[lat]>.015:
        raise error('飞机左右对称证据不足，疑似缺失/破碎重建；未自动编辑',5)
    # A rotated vehicle cannot safely be edited with this axis-aligned engine.
    corr=np.corrcoef(p.T);off=np.abs(corr-np.eye(3)).max()
    if off>.6 and axis_source!='provided_frame_no_region':raise error('模型疑似斜置；当前自动朝向只支持近似轴对齐载具',5)
    q=(p[:,long]-v[:,long].min())/span[long]
    widths=[]
    for low,high in ((.015,.16),(.84,.985)):
        a=p[(q>=low)&(q<=high),lat]
        widths.append(float(np.ptp(np.quantile(a,[.04,.96]))) if len(a)>30 else float('nan'))
    ratio=max(widths)/max(min(widths),1e-12)
    front=int(1 if widths[1]<widths[0] else -1)
    front_key='bow_sign' if category=='ship' else 'front_sign'
    if front_key in h:front=int(h[front_key]);front_source='provided_direction_no_region'
    elif np.isfinite(ratio) and ratio>=1.25:front_source='end_section_taper'
    else:
        tips=[]
        for region in (q<.04,q>.96):
            values=p[region,lat]
            tips.append(float(np.ptp(np.quantile(values,[.03,.97]))) if len(values)>30 else float('nan'))
        tip_ratio=max(tips)/max(min(tips),1e-12)
        candidate=1 if tips[1]<tips[0] else -1
        index=1 if candidate>0 else 0
        if category=='ship' and np.isfinite(tip_ratio) and tip_ratio>=1.5 and tips[index]<.75*widths[index]:
            front=candidate;front_source='terminal_point_taper'
        else:front=0;front_source='ambiguous'
    median=np.median(p[:,up]);neg=median-np.quantile(p[:,up],.005);pos=np.quantile(p[:,up],.995)-median
    upsign=int(1 if pos>neg else -1)
    up_ratio=max(pos,neg)/max(min(pos,neg),1e-12)
    if 'up_sign' in h:upsign=int(h['up_sign']);up_source='provided_direction_no_region'
    elif up_ratio>=1.3:up_source='upper_appendage_extent'
    else:upsign=0;up_source='ambiguous'
    right=0
    if front and upsign:
        fv=np.zeros(3);uv=np.zeros(3);fv[long]=front;uv[up]=upsign
        right=int(np.cross(fv,uv)[lat])
    if 'right_sign' in h:right=int(h['right_sign'])
    if category in ('car','truck','tank','armored') and 'front_sign' not in h:
        front=0;front_source='ground_vehicle_front_not_inferred_from_taper'
        right=0
    frame={'long_axis':'xyz'[long],'lateral_axis':'xyz'[lat],'up_axis':'xyz'[up],
           front_key:front,'up_sign':upsign,'right_sign':right,'centerline':centers[lat]}
    return frame,{'axis_source':axis_source,'axis_symmetry_error':dict(zip('xyz',sym)),
        'front_source':front_source,'end_widths':widths,'end_taper_ratio':ratio,
        'up_source':up_source,'up_extent_ratio':float(up_ratio),'anatomical_right_convention':'front cross up'},p


def _partfield_results_root():
    """Resolve the PartField exp_results directory, or return a non-existent path."""
    import os
    env = os.environ.get("TRELLIS_PARTFIELD_RESULTS")
    here = Path(__file__).resolve().parents[3]          # .../TRELLIS
    candidates = [
        Path(env) if env else None,
        here.parent / "PartField" / "exp_results",      # sibling of the checkout
        here / "partfield_results",                     # inside the checkout
    ]
    for c in candidates:
        if c is not None and c.is_dir():
            return c
    return Path("/nonexistent/partfield_results")


def partfield_refine(v,f,prior,error,target=None):
    """Find matching precomputed geometry; never assume face count implies identity."""
    # PartField results are optional precomputation; locate them without assuming
    # this dev machine's layout. TRELLIS_PARTFIELD_RESULTS wins, then a sibling
    # directory of the checkout, then the old absolute path for backwards compat.
    root = _partfield_results_root()
    centers=v[f].mean(1);diag=np.linalg.norm(np.ptp(v,axis=0))
    import trimesh
    best=None
    for folder in sorted((root/'partfield_features').glob('*')):
        for source in folder.glob('input_*.ply'):
            mesh=trimesh.load(source,process=False,force='mesh')
            if len(mesh.faces)!=len(f):continue
            # PartField normalizes the source bbox to a max span of 1.8.
            # Undo ONLY verified uniform scale/translation, not arbitrary alignment.
            source_span=np.ptp(mesh.vertices,axis=0)
            scale=np.max(np.ptp(v,axis=0))/np.max(source_span)
            if not np.allclose(source_span*scale,np.ptp(v,axis=0),rtol=1e-4,atol=diag*1e-6):continue
            aligned=(mesh.vertices-mesh.bounds.mean(0))*scale+(v.max(0)+v.min(0))/2
            original=aligned[mesh.faces].mean(1)
            d,mapping=cKDTree(original).query(centers)
            if np.max(d)>max(1e-7,diag*1e-6) or len(np.unique(mapping))<.999*len(f):continue
            # Also check the matched triangles' sorted edge lengths, not only centres.
            def lengths(vv,ff):
                t=vv[ff];return np.sort(np.stack([np.linalg.norm(t[:,i]-t[:,(i+1)%3],axis=1) for i in range(3)],1),1)
            if not np.allclose(lengths(v,f),lengths(aligned,mesh.faces)[mapping],rtol=1e-5,atol=diag*1e-6):continue
            area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
            face_target=prior[f].mean(1)>.35
            for labelpath in (root/'clustering'/folder.name/'cluster_out').glob('*.npy'):
                labels=np.load(labelpath,allow_pickle=False)
                if labels.shape!=(len(f),):continue
                labels=labels[mapping];union=np.zeros(len(f),bool);chosen=[]
                for label in np.unique(labels):
                    region=labels==label
                    overlap=area[region&face_target].sum()
                    purity=overlap/max(area[region].sum(),1e-15)
                    coverage=overlap/max(area[face_target].sum(),1e-15)
                    if purity>=.55 and coverage>=.06:union|=region;chosen.append(int(label))
                overlap=area[union&face_target].sum()
                score=2*overlap/max(area[union].sum()+area[face_target].sum(),1e-15)
                if chosen and (best is None or score>best[0]):best=(float(score),union,labelpath,chosen)
    if best is None or best[0]<.55:return prior,{'used':False,'reason':'no verified matching PartField candidates or weak geometric agreement'}
    score,union,path,chosen=best
    weights=np.zeros(len(v));weights[np.unique(f[union])]=1
    edges=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
    r=np.r_[edges[:,0],edges[:,1]];c=np.r_[edges[:,1],edges[:,0]]
    graph=coo_matrix((np.ones(len(r)),(r,c)),shape=(len(v),)*2).tocsr();degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    for _ in range(8):weights=.5*weights+.5*(graph@weights)/degree
    # The broad spatial support prevents a cluster from leaking to the opposite wing/end.
    weights*=smooth(prior/.25)
    # A cluster boundary is not a deformation boundary. Diffuse the final
    # selection on mesh adjacency to supply a wider, continuous transition.
    for _ in range(100 if target in ('left_wing','right_wing','wing','stern') else 0):weights=.6*weights+.4*(graph@weights)/degree
    weights=smooth(np.clip((weights-1e-5)/(max(weights.max(),1e-5)-1e-5+1e-12),0,1))
    return weights,{'used':True,'labels_file':str(path),'cluster_ids':chosen,'geometric_agreement_f1':score,
                     'note':'agreement with geometric prior, NOT semantic GT IoU'}


def locate(v,f,request,h,error):
    if request.target in ('ground_front','ground_rear','ground_body','cabin','cargo_bed','turret','barrel','tracks','whole_vehicle'):
        try:
            from .ground_vehicle_parts import locate_ground
        except ImportError:
            from ground_vehicle_parts import locate_ground
        return locate_ground(v,f,request,h,error)
    category='ship' if request.target in ('bow','stern','front_hull','hull','deck','mast') else 'aircraft'
    frame,evidence,p=infer_frame(v,f,category,h,error)
    if category=='ship' and frame.get('bow_sign',0)==0 and h.get('source_photo'):
        try:
            from .photo_direction_bridge import resolve
        except ImportError:
            from photo_direction_bridge import resolve
        photo_evidence,path=resolve(v,f,h,error)
        frame['bow_sign']=photo_evidence['bow_sign']
        evidence['front_source']='photo_semantics_plus_verified_visual_correspondence'
        evidence['photo_direction_evidence']=path
        evidence['photo_direction_summary']={'photo_semantics':photo_evidence['photo_semantics'],
                                           'registration':photo_evidence['registration']}
        if frame['up_sign']:
            fv=np.zeros(3);uv=np.zeros(3)
            fv['xyz'.index(frame['long_axis'])]=frame['bow_sign'];uv['xyz'.index(frame['up_axis'])]=frame['up_sign']
            frame['right_sign']=int(np.cross(fv,uv)['xyz'.index(frame['lateral_axis'])])
    long,lat,up=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
    lo=v.min(0);span=np.ptp(v,axis=0);pos=(v-lo)/span
    front=frame.get('bow_sign',frame.get('front_sign',0));target=request.target
    if target in ('bow','stern','nose','tail','front_hull','left_wing','right_wing','wing') and not front:
        raise error('自动定位无法区分前后端；只需一次确认 bow_sign/front_sign，无需提供部件区域',5)
    longitudinal=pos[:,long] if front>0 else 1-pos[:,long]
    if target in ('bow','nose','stern','tail','front_hull'):
        x=1-longitudinal if target in ('stern','tail') else longitudinal
        start=.5 if target=='front_hull' else .70
        prior=smooth((x-start)/.18)
        if target=='tail':
            # Tail query is restricted to appendages; a simple rear slice includes the fuselage.
            if not frame['up_sign']:raise error('尾翼上方向不明确',5)
            height=pos[:,up] if frame['up_sign']>0 else 1-pos[:,up]
            lateral=np.abs(v[:,lat]-frame['centerline'])/(span[lat]/2)
            prior*=np.maximum(smooth((height-.65)/.15),smooth((lateral-.25)/.2))
    elif target in ('left_wing','right_wing','wing'):
        side=frame['right_sign']
        if target!='wing' and not side:raise error('无法判定载具自身左右侧；只需确认up_sign，无需框选机翼',5)
        # Find the main wing lobe in the longitudinal width profile.
        q=(p[:,long]-lo[long])/span[long]
        if front<0:q=1-q
        widths=[]
        for a in np.linspace(0,1,25)[:-1]:
            pp=p[(q>=a)&(q<a+1/24),lat]
            widths.append(float(np.ptp(np.quantile(pp,[.02,.98]))) if len(pp)>20 else 0)
        peak=int(np.argmax(widths));selected=np.asarray(widths)>=.75*max(widths)
        a=b=peak
        while a>0 and selected[a-1]:a-=1
        while b<23 and selected[b+1]:b+=1
        if b-a>14:raise error('缺少可区分的主翼轮廓，未把整机当作机翼',5)
        band=smooth((longitudinal-(a/24-.06))/.06)*(1-smooth((longitudinal-((b+1)/24+.02))/.06))
        lateral=(v[:,lat]-frame['centerline'])/(span[lat]/2)
        if target=='right_wing':lateral*=side
        elif target=='left_wing':lateral*=-side
        else:lateral=np.abs(lateral)
        prior=band*smooth((lateral-.13)/.25)
        # Separate thin main-wing surfaces from nearby high vertical tail fins.
        qlat=np.abs(p[:,lat]-frame['centerline'])/(span[lat]/2)
        wing_samples=p[(q>=a/24)&(q<=(b+1)/24)&(qlat>.65),up]
        if len(wing_samples)>30:
            center=float(np.median(wing_samples))
            thickness=max(float(np.ptp(np.quantile(wing_samples,[.03,.97]))),.025*span[up])
            prior*=1-smooth((np.abs(v[:,up]-center)-thickness)/max(thickness,1e-12))
            evidence['main_wing_height_center_and_band']=[center,thickness]
        evidence['main_wing_longitudinal_band']=[a/24,(b+1)/24]
    elif target=='mast':
        if not frame['up_sign']:raise error('桅杆上下方向不明确',5)
        height=pos[:,up] if frame['up_sign']>0 else 1-pos[:,up]
        seed=height>.8
        if seed.sum()<20:raise error('未找到稳定的高处细长结构',5)
        center=np.median(v[seed],axis=0)
        radius=np.ptp(v[seed],axis=0)
        if max(radius[long],radius[lat])>.25*span[long]:raise error('顶部结构不够细长，不能自动声称是桅杆',5)
        prior=smooth((height-.55)/.2)
        for axis in (long,lat):
            r=max(radius[axis]*.8,span[axis]*.035)
            prior*=1-smooth((np.abs(v[:,axis]-center[axis])-r)/r)
        evidence['mast_seed']='highest slender structure; semantic identity unverified'
    else:raise error('该部件暂无可靠自动定位器，不会将整个包围盒作为目标',5)
    if np.count_nonzero(prior>.5)<20:raise error('自动目标区域过小或缺失',5)
    weights,pf=partfield_refine(v,f,prior,error,target)
    evidence['partfield']=pf
    if target=='tail' and not pf['used']:
        raise error('尾翼候选缺少可靠部件特征支持，拒绝仅按后端区域编辑',5)
    evidence.update({'category_from_text':category,'target':target,'method':'geometry_semantic_prior_plus_verified_multiscale_partfield' if pf['used'] else 'geometry_category_specific_locator',
        'automatic':True,'semantic_GT_available':False,'frame':frame})
    return weights,frame,evidence
