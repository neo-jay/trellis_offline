"""Category-specific conservative ground-vehicle part candidates.

No semantic GT is assumed. Cabin/front/rear are deliberately refused without
additional verified evidence; upper body, turret, barrel, track assemblies have
geometric priors which may be refined by the existing PartField candidates.
"""
import numpy as np


def locate_ground(v,f,request,h,error):
    try:
        from .auto_vehicle_parts import infer_frame,partfield_refine,smooth
    except ImportError:
        from auto_vehicle_parts import infer_frame,partfield_refine,smooth
    category=request.vehicle_category or ('tank' if request.target in ('turret','barrel','tracks') else 'car')
    frame,evidence,p=infer_frame(v,f,category,h,error)
    long,lat,up=['xyz'.index(frame[k]) for k in ('long_axis','lateral_axis','up_axis')]
    if not frame['up_sign'] and (request.target!='whole_vehicle' or request.operation in ('heighten','lower')):raise error('地面载具上下方向不明确，未猜测底盘/炮塔位置',5)
    lo=np.quantile(p,.02,axis=0);hi=np.quantile(p,.98,axis=0);span=np.maximum(hi-lo,1e-12)
    pos=(v-lo)/span;height=pos[:,up] if frame['up_sign']>0 else 1-pos[:,up]
    lateral=np.abs(v[:,lat]-frame['centerline'])/(span[lat]/2)
    longitudinal=pos[:,long];target=request.target
    if target=='whole_vehicle':
        evidence.update(category_from_text=category,target=target,method='explicit_whole_vehicle_affine',
                        automatic=True,semantic_GT_available=False,frame=frame,
                        part_definition='explicit entire vehicle; NOT local semantic part selection')
        return np.ones(len(v)),frame,evidence
    if target=='ground_body':
        prior=smooth((height-.20)/.18)*(1-smooth((height-1.1)/.25))
        evidence['part_definition']='upper vehicle body; lower wheel/track zone excluded geometrically, not a semantic GT mask'
    elif target=='turret':
        prior=smooth((height-.48)/.20)
        prior*=smooth((longitudinal-.18)/.12)*(1-smooth((longitudinal-.73)/.12))
        prior*=1-smooth((lateral-.6)/.25)
        evidence['part_definition']='central upper armored structure candidate; barrel/antennas may require PartField refinement'
    elif target=='tracks':
        if category not in ('tank','armored'):raise error('履带需要履带式载具类别',5)
        prior=(1-smooth((height-.38)/.18))*smooth((lateral-.5)/.25)
        evidence['part_definition']='track and road-wheel assemblies, not the rubber/metal belt alone'
    elif target=='barrel':
        # Prefer an actual slender connected component when the asset has one.
        # Geometric welding is for analysis ONLY; original OBJ/UV indices remain.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        _,inverse=np.unique(np.round(v/max(np.linalg.norm(np.ptp(v,axis=0))*1e-9,1e-12)).astype('int64'),axis=0,return_inverse=True)
        ff=inverse[f];edges=np.vstack((ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]));n=int(inverse.max()+1)
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n)).tocsr()
        count,labels=connected_components(graph,directed=False);labels=labels[inverse];candidates=[]
        for label in range(count):
            mask=labels==label
            if mask.sum()<20:continue
            pts=v[mask];extent=np.ptp(pts,axis=0);center=pts.mean(0)
            height_center=(center[up]-lo[up])/span[up] if frame['up_sign']>0 else (hi[up]-center[up])/span[up]
            elongation=extent[long]/max(extent[lat],extent[up],1e-12)
            if extent[long]>.15*span[long] and extent[lat]<.2*span[lat] and extent[up]<.25*span[up] and elongation>4 and abs(center[lat]-frame['centerline'])<.2*span[lat] and height_center>.3:
                candidates.append((float(elongation),label,mask,center))
        candidates.sort(key=lambda a:-a[0])
        if candidates and (len(candidates)==1 or candidates[0][0]>1.2*candidates[1][0]):
            score,label,mask,center=candidates[0]
            frame['barrel_sign']=1 if center[long]>(v[:,long].min()+v[:,long].max())/2 else -1
            evidence.update(category_from_text=category,target=target,method='automatic_slender_connected_component',
                            automatic=True,semantic_GT_available=False,frame=frame,
                            component_id=int(label),component_elongation=score,
                            part_definition='automatically identified central projecting slender component; no manual component IDs')
            return mask.astype(float),frame,evidence
        # A projecting cannon is geometrically different from the hull front:
        # its own direction is inferred independently, never reused as hull yaw.
        ppos=(p[:,long]-v[:,long].min())/np.ptp(v[:,long]);widths=[]
        for i in range(36):
            section=p[(ppos>=i/36)&(ppos<(i+1)/36)]
            widths.append(float(np.ptp(np.quantile(section[:,lat],[.03,.97]))) if len(section)>8 else 0)
        widths=np.asarray(widths);threshold=.28*np.max(widths)
        counts=[]
        for data in (widths,widths[::-1]):
            n=0
            for width in data:
                if width<threshold:n+=1
                else:break
            counts.append(n)
        if max(counts)<4 or max(counts)<1.5*max(min(counts),1):
            raise error('未找到独立、足够长且轴对齐的细长炮管，拒绝把车头当炮管',5)
        sign=-1 if counts[0]>counts[1] else 1;band=max(counts)/36
        s=(v[:,long]-v[:,long].min())/np.ptp(v[:,long])
        if sign<0:s=1-s
        prior=smooth((s-(1-band-.03))/.06)
        prior*=smooth((height-.27)/.15)*(1-smooth((lateral-.30)/.25))
        frame['barrel_sign']=sign
        evidence.update(barrel_end_band=band,barrel_sign=sign,
                        part_definition='projecting axial cannon candidate; NOT assumed to be vehicle front')
    elif target in ('ground_front','ground_rear','cabin','cargo_bed'):
        # Longitudinal end cut plus an optional height band. This is a
        # positional heuristic, not a learned part segmentation: it assumes the
        # long axis points from rear to front and that the cab/cargo area sits
        # above the chassis. Recorded as such in the evidence dict.
        if not frame.get('front_sign'):
            raise error('自动定位无法区分车头车尾；只需一次确认 front_sign，无需提供部件区域',5)
        fs = frame['front_sign']
        longitudinal = pos[:,long] if fs>0 else 1-pos[:,long]
        if target in ('ground_front','cabin'):
            band = smooth((longitudinal-.72)/.28)
        else:
            band = smooth((.30-longitudinal)/.30)
        if target in ('cabin','cargo_bed'):
            if not frame.get('up_sign'):
                raise error('驾驶室/货厢需要确认上下方向 up_sign，不猜测',5)
            upper = smooth((height-.45)/.40)
            prior = band*upper
        else:
            prior = band
    else:raise error('不支持的地面载具部件',5)
    if np.count_nonzero(prior>.5)<20:raise error('未找到足够的目标部件候选',5)
    weights,pf=partfield_refine(v,f,prior,error,target)
    evidence.update(category_from_text=category,target=target,method='ground_geometry_plus_verified_multiscale_partfield' if pf['used'] else 'ground_category_geometry_candidate',
                    partfield=pf,automatic=True,semantic_GT_available=False,frame=frame)
    return weights,frame,evidence
