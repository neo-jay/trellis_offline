"""Offline controlled-language planning and conservative geometry proposals.

No learned semantic segmentation is claimed: supplied part masks take priority;
axis-based proposals are explicitly identified as geometric approximations.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SizeRequest:
    target: str
    operation: str
    magnitude: float
    instruction: str
    preserve: list = field(default_factory=list)
    vehicle_category: str = ''


_DEFAULT_MAGNITUDE = 0.08

PARTS = {
    'whole_vehicle': ('整辆汽车','整辆卡车','整辆坦克','整台坦克','整辆装甲车','整辆越野车','整车','whole vehicle'),
    'ground_front': ('车头','车首','汽车前部','卡车前部','vehicle front'),
    'ground_rear': ('车尾','汽车后部','vehicle rear'),
    'ground_body': ('车身','车体','汽车主体','卡车主体','坦克底盘','装甲车主体','vehicle body'),
    'cabin': ('驾驶室','驾驶舱','乘员舱','truck cabin'),
    'cargo_bed': ('货厢','货箱','货斗','车斗','载货平台','cargo bed'),
    'turret': ('炮塔','坦克上塔','turret'),
    'barrel': ('炮管','炮筒','主炮身管','gun barrel'),
    'tracks': ('履带总成','履带','tracks'),
    'front_hull': ('船体前段', '船身前段', '前段船体'),
    'right_wing': ('右侧机翼', '右边机翼', '右机翼', '右翼', 'right wing'),
    'left_wing': ('左侧机翼', '左边机翼', '左机翼', '左翼', 'left wing'),
    'bow': ('船头', '船首', '艏部', '舰首', 'bow'),
    'stern': ('船尾', '船艉', '舰尾', 'stern'),
    'mast': ('桅杆', '主桅', 'mast'),
    'deck': ('甲板', 'deck'),
    'hull': ('船体', '船身', '船壳', 'hull'),
    'nose': ('机头', '机鼻', 'nose'),
    'tail': ('尾翼', '垂尾', '平尾', 'tail fin'),
    'wing': ('机翼', '主翼', '翅膀', 'wings'),
    'fuselage': ('机身', '机体', 'fuselage'),

    # ---- submarine ----
    'conning_tower': ('指挥塔', '围壳', '帆罩', '瞭望塔', 'conning tower'),
    'periscope': ('潜望镜', '升降装置', 'periscope'),
    'sub_bow': ('艇首', 'submarine bow'),
    'sub_stern': ('艇尾', 'submarine stern'),
    'sub_hull': ('艇体', '艇身', '耐压壳', 'pressure hull'),
    'propeller': ('螺旋桨', '尾轴', 'propeller', 'screw'),

    # ---- bus ----
    'bus_front': ('公交车车头', '公交前脸', 'bus front'),
    'bus_rear': ('公交车车尾', '公交尾部', 'bus rear'),
    'bus_roof': ('公交车车顶', '公交顶棚', 'bus roof'),
    'bus_body': ('公交车身', '公交车体', 'bus body'),

    # ---- metro / tram ----
    'metro_head': ('地铁车头', '前车厢', 'head car'),
    'metro_body': ('地铁车厢', '地铁车身', 'metro car body'),
    'metro_door': ('车门', '车门板', 'metro door'),

    # ---- hovercraft ----
    'skirt': ('气垫裙', '裙边', '围裙', 'skirt'),
    'fan': ('涵道风扇', '推进风扇', 'ducted fan'),

    # ---- drone ----
    'rotor': ('旋翼', 'rotor', 'drone rotor'),
    'drone_arm': ('机臂', '臂管', '支臂', 'drone arm'),
    'gimbal': ('云台', '相机吊舱', 'gimbal'),

    # ---- trailer / caravan ----
    'caravan_body': ('房车厢体', '房车车箱', 'caravan body'),
    'hitch': ('牵引架', '挂钩', '拖钩', 'hitch', 'tow bar'),
    'awning': ('遮阳棚', '雨棚', 'awning'),

    # ---- agricultural ----
    'header': ('割台', '收割台', 'header'),
    'auger': ('卸粮筒', '输送筒', 'auger'),
    'agri_cabin': ('农机驾驶室', 'operator station'),

    # ---- military / construction extras ----
    'launcher': ('发射架', '发射筒', '导弹舱', 'launcher'),
    'radar_dish': ('雷达天线', '雷达罩', 'radar dish'),
    'excavator_arm': ('大臂', '动臂', '挖掘臂', 'excavator arm', 'boom'),
    'bucket': ('铲斗', '挖斗', 'bucket', 'scoop'),
}


def parts(text):
    # Longest spans win, so 右侧机翼 does not also select generic 机翼.
    hits = []
    occupied = set()
    candidates = [(m.start(), m.end(), p) for p, words in PARTS.items()
                  for w in words for m in re.finditer(re.escape(w), text)]
    # Two surface forms can map to the same canonical part (旋翼 / 桨叶 -> rotor).
    # Counting both made a single-part instruction look like two targets, so
    # collapse by canonical name and keep the earliest match position.
    chosen = {}
    for a, b, p in sorted(candidates, key=lambda x: (-(x[1] - x[0]), x[0])):
        if occupied.intersection(range(a, b)):
            continue
        occupied.update(range(a, b))
        if p not in chosen:
            chosen[p] = a
    return [p for p, _ in sorted(chosen.items(), key=lambda kv: kv[1])]


def parse(text, magnitude, error):
    text = str(text or '').strip().lower()
    if not text:
        raise error('编辑指令不能为空', 1)
    if re.search(r'\d+(?:\.\d+)?\s*(?:米|厘米|毫米|cm\b|mm\b|m\b)', text):
        raise error('绝对尺寸需要真实尺度标定，请使用百分比', 1)
    clauses = re.split(r'[，,。;；]|但是|但|同时|并且|而且', text)
    edits, preserve = [], []
    for clause in clauses:
        ps = parts(clause)
        if re.search(r'别动|不动|保持|不要动|不要改|不改变|不变|保留|不调整|不修改|unchanged|keep', clause):
            if not ps:
                raise error('保留约束未识别到部件', 1)
            preserve += ps
        elif clause.strip():
            edits.append(clause)
    active = ' '.join(edits)
    if re.search(r'不要|别|不想|不是|不能', active):
        raise error('否定或纠正指令存在歧义，请只描述希望执行的单个调整', 1)
    targets = parts(active)
    if len(targets) != 1:
        raise error('每次须明确一个目标部件；未知部件或多个目标不能猜测', 1)
    target = targets[0]
    if target in preserve:
        raise error('目标部件同时被要求保留，请消除冲突', 1)
    matches = []
    for op, pattern in (
        ('widen', r'宽|粗|widen|wider|thicker'),
        ('narrow', r'窄|细|narrow|thinner'),
        ('lengthen', r'拉长|加长|延长|伸长|长一点|长一些|longer|lengthen|extend'),
        ('shorten', r'缩短|变短|短一点|短一些|shorten|shorter'),
        ('heighten', r'增高|加高|升高|抬高|拔高|变高|高一点|高一些|taller|higher'),
        ('lower', r'降低|变矮|矮一点|矮一些|lower|shorter vertically')):
        if re.search(pattern, active):
            matches.append(op)
    if len(matches) != 1:
        raise error('需明确一个尺寸方向：加宽/变窄/加长/缩短/增高/降低', 1)
    pct = re.findall(r'(?<![\d.])(\d+(?:\.\d+)?)\s*[%％]', active)
    if len(pct) > 1 or re.search(r'-\s*\d', active):
        raise error('百分比须是单个正数', 1)
    # Default must match mesh_text_edit.DEFAULT_MAGNITUDE (0.08). It was hardcoded
    # to 0.05 here, so the same instruction produced different magnitudes
    # depending on which entry point was used.
    value = float(pct[0])/100 if pct else _DEFAULT_MAGNITUDE
    if not pct and re.search(r'明显|大幅|很多', active):
        value = .12
    if magnitude is not None:
        value = float(magnitude)
    if not np.isfinite(value) or not 0 < value <= .30:
        raise error('相对幅度须在 (0, 0.30] 内', 1)
    category=''
    for name,words in [
        ('submarine',('潜艇','核潜艇','深潜器')),
        ('bus',('公交车','巴士','大巴','公交')),
        ('tram_metro',('地铁','轻轨','有轨电车','城轨')),
        ('hovercraft',('气垫船','地效')),
        ('drone',('无人机','四旋翼','多旋翼','穿越机')),
        ('trailer',('房车','拖挂','半挂')),
        ('agricultural',('拖拉机','收割机','播种机','农机')),
        ('building_vehicle',('挖掘机','推土机','装载机','起重机','叉车')),
        ('tank',('坦克','炮塔','炮管','炮筒','履带')),
                       ('armored',('装甲车','装甲输送车')),
                       ('truck',('卡车','货车','消防车','货厢','货箱','货斗','车斗','驾驶室')),
                       ('car',('汽车','轿车','越野车','suv','皮卡'))]:
        if any(word in text for word in words):category=name;break
    if target in ('ground_front','ground_rear','ground_body','cabin','whole_vehicle') and not category:category='car'
    return SizeRequest(target, matches[0], value, text, list(dict.fromkeys(preserve)),category)


def plan(data, request, hint, error):
    h = dict(hint or {})
    v, f = data.vertices, data.faces

    # fragmented-mesh guard (mirrors the legacy plan)
    # The legacy plan refused badly fragmented models: a part cannot be located
    # reliably when the largest connected component is a small share of the
    # mesh. That check was lost when this module took over the dispatch, so the
    # same thresholds apply here.
    from scipy.sparse import coo_matrix as _fg_coo
    from scipy.sparse.csgraph import connected_components as _fg_cc
    _edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _graph = _fg_coo((np.ones(len(_edges), dtype=np.uint8), (_edges[:, 0], _edges[:, 1])),
                     shape=(len(v), len(v))).tocsr()
    _ncomp, _labels = _fg_cc(_graph, directed=False)
    _sizes = np.bincount(_labels[f[:, 0]], minlength=_ncomp)
    _occupied = _sizes[_sizes > 0]
    _largest = float(_occupied.max() / len(f)) if len(f) else 0.0
    # Refuse only a model split into a few large disconnected pieces. A very
    # high component count with a tiny largest share is an unindexed triangle
    # soup (typical of GLB->OBJ conversion): each triangle owns its vertices, so
    # the graph has thousands of 1-face components while the surface itself is
    # continuous and perfectly editable. Treating that as "fragmented" rejected
    # every real model tested.
    _few_pieces = len(_occupied) <= 8 and _largest < 0.55
    if _few_pieces:
        raise error('模型碎片化严重（%d 个独立大块，最大块仅占 %.0f%% 面数）；'
                    '请先合并为单一连通体或提供质量更好的 OBJ'
                    % (len(_occupied), 100.0 * _largest), 5)
    automatic=None
    automatic_weights=None
    auto_enabled=h.get('auto_localize',not any(k in h for k in ('bbox','bbox_fraction','part_masks_npz','long_axis')))
    if auto_enabled:
        try:
            from .auto_vehicle_parts import locate
        except ImportError:
            from auto_vehicle_parts import locate
        automatic_weights,frame,automatic=locate(v,f,request,h,error)
        h={**frame,**h}
    lo, hi = v.min(0), v.max(0)
    span = hi-lo
    axes = []
    for key in ('long_axis', 'lateral_axis', 'up_axis'):
        x = h.get(key)
        if x not in ('x', 'y', 'z'):
            raise error('须提供并确认 long_axis/lateral_axis/up_axis，不能从包围盒猜测朝向', 5)
        axes.append('xyz'.index(x))
    if len(set(axes)) != 3 or np.any(span <= 1e-12):
        raise error('坐标轴须互异，模型不可退化为零厚度', 5)
    long, lateral, up = axes
    def sign(key):
        value = h.get(key)
        if isinstance(value, bool) or value not in (-1, 1):
            raise error('须明确 '+key+' 为 +1 或 -1', 5)
        return value
    def smooth(x):
        t = np.clip(x, 0, 1); return t*t*(3-2*t)
    pos = (v-lo)/span
    masks = {}
    path = h.get('part_masks_npz')
    if path:
        with np.load(Path(path), allow_pickle=False) as archive:
            # Mesh identity is mandatory: vertex counts alone do not identify a mesh.
            import hashlib
            digest = hashlib.sha256(np.asarray(v, '<f8').tobytes()+np.asarray(f, '<i8').tobytes()).hexdigest()
            if 'mesh_geometry_sha256' not in archive or str(archive['mesh_geometry_sha256'].item()) != digest:
                raise error('部件掩码与输入网格几何指纹不匹配', 5)
            for key in archive.files:
                if key == 'mesh_geometry_sha256': continue
                a = np.asarray(archive[key], float)
                if a.shape != (len(v),) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
                    raise error('部件掩码须为与顶点一一对应的 [0,1] 权重', 5)
                masks[key] = a
    def box_mask(box):
        b = np.asarray(box, float)
        if b.shape != (2,3) or not np.isfinite(b).all() or np.any(b[0]>=b[1]):
            raise error('bbox 须为有限三维上下界', 5)
        feather = float(h.get('box_feather', .15))
        if not 0 < feather <= .45: raise error('box_feather 须在 (0,.45]',5)
        # BBox encloses full core; fade outward, not inward (extreme tips must move).
        edge = np.maximum((b[1]-b[0])*feather, 1e-12)
        return np.prod(smooth((v-b[0]+edge)/edge)*smooth((b[1]+edge-v)/edge),axis=1)
    method = 'supplied_part_mask'
    target = request.target
    if automatic_weights is not None:
        w=automatic_weights;method=automatic['method']
    elif target in masks:
        w = masks[target].copy()
    elif 'bbox' in h or 'bbox_fraction' in h:
        b = h.get('bbox')
        if b is None:
            fraction = np.asarray(h['bbox_fraction'],float)
            if fraction.shape != (2,3) or not np.isfinite(fraction).all() or np.any((fraction<0)|(fraction>1)):
                raise error('bbox_fraction 须位于 [0,1]',5)
            b = lo+fraction*span
        w = box_mask(b); method = 'supplied_bbox'
    else:
        method = 'axis_geometric_proposal'
        if target in ('bow','nose','front_hull','stern','tail'):
            front = sign('bow_sign' if target in ('bow','front_hull','stern') else 'front_sign')
            p = pos[:,long] if front>0 else 1-pos[:,long]
            if target in ('stern','tail'): p=1-p
            start = float(h.get('bow_start', .6 if target!='front_hull' else .5))
            if not .3<=start<=.9: raise error('bow_start 须在 [.3,.9]',5)
            w = smooth((p-start)/(1-start))
        elif target in ('left_wing','right_wing','wing'):
            side = sign('right_sign')
            p = (pos[:,lateral]-.5)*2*side
            if target=='left_wing': p=-p
            if target=='wing': p=np.abs(p)
            w = smooth((p-.2)/.5)
        elif target in ('hull','fuselage'):
            raise error('整船体/机身需 bbox 或部件掩码，不能把机翼与上层建筑一起当作机身',5)
        else:
            raise error('该部件需要 bbox 或 part_masks_npz，不能仅按高度猜测桅杆/甲板',5)
    protected = np.zeros(len(v), bool)
    for p in request.preserve:
        if p not in masks:
            boxes = h.get('preserve_bboxes',{})
            if p not in boxes:
                raise error('保留 '+p+' 需要部件掩码或 preserve_bboxes，不能假装已保留',5)
            mask = box_mask(boxes[p])
        else: mask=masks[p]
        protected |= mask>0
    if protected.any():
        # Smoothly approach fixed constraints along the mesh, not Euclidean
        # distance (which would leak across nearby but disconnected surfaces).
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        edges = np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
        edges = np.unique(np.sort(edges,axis=1),axis=0)
        lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        roots=np.flatnonzero(protected)
        n=len(v)
        rows=np.r_[edges[:,0],edges[:,1],np.full(len(roots),n)]
        cols=np.r_[edges[:,1],edges[:,0],roots]
        costs=np.r_[lengths,lengths,np.full(len(roots),1e-15)]
        graph=coo_matrix((costs,(rows,cols)),shape=(n+1,n+1)).tocsr()
        distances=dijkstra(graph,directed=False,indices=n)[:n]
        radius=.04*np.linalg.norm(span)
        w *= smooth(distances/max(radius,1e-12))
        w[protected]=0
    # GLB/OBJ assets may duplicate geometric positions at UV/normal seams.
    # Tie their displacements without merging vertices or changing UV indices.
    tolerance=max(float(np.linalg.norm(span))*1e-10,1e-14)
    _,position_groups,group_sizes=np.unique(np.round(v/tolerance).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    if np.any(group_sizes>1):
        averages=np.bincount(position_groups,weights=w)/group_sizes
        group_protected=np.bincount(position_groups,weights=protected.astype(float))>0
        averages[group_protected]=0
        w=averages[position_groups]
    selected = w>1e-4
    if selected.sum()<20: raise error('选区少于20个顶点，请重新确认区域',5)
    op=request.operation
    axis = lateral if op in ('widen','narrow') else up if op in ('heighten','lower') else long
    if op in ('lengthen','shorten') and target=='mast': axis=up
    if op in ('lengthen','shorten') and target in ('left_wing','right_wing','wing'): axis=lateral
    if 'dimension_axis' in h:
        if h['dimension_axis'] not in ('x','y','z'): raise error('dimension_axis 须为 x/y/z',5)
        axis='xyz'.index(h['dimension_axis'])
    signed = request.magnitude * (-1 if op in ('narrow','shorten','lower') else 1)
    delta=np.zeros_like(v)
    if axis==lateral or (axis==long and target in ('ground_body','turret','cargo_bed','cabin','tracks','whole_vehicle')):
        # Shared centreline gives symmetric bilateral edits; single-wing edit uses root anchor.
        anchor=float(h.get('centerline',(lo[lateral]+hi[lateral])/2)) if axis==lateral else float(np.median(v[selected,axis]))
    else:
        direction=sign('up_sign') if axis==up else sign('barrel_sign' if target=='barrel' else 'bow_sign' if target in ('bow','front_hull','hull','stern') else 'front_sign')
        if target in ('stern','tail','ground_rear'): direction=-direction
        anchor=float(v[selected,axis].min() if direction>0 else v[selected,axis].max())
    delta[:,axis]=(v[:,axis]-anchor)*signed*w
    mode=h.get('dimension_mode','local_scale')
    if mode not in ('local_scale','core_extent'):
        raise error('dimension_mode 须为 local_scale 或 core_extent',5)
    threshold=float(h.get('measurement_core_threshold',.9))
    if not np.isfinite(threshold) or not .5<=threshold<=1:
        raise error('measurement_core_threshold 须在 [.5,1]',5)
    measure_mask=w>=threshold*w.max()
    if 'measurement_vertex_indices' in h:
        indices=np.asarray(h['measurement_vertex_indices'])
        if indices.ndim!=1 or len(indices)<2 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(v)):
            raise error('measurement_vertex_indices 须为至少两个合法原始顶点索引',5)
        measure_mask=np.zeros(len(v),bool);measure_mask[indices]=True
    coefficient=signed
    unit=(v[:,axis]-anchor)*w
    if h.get('deformation_solver')=='harmonic_transition':
        core_handles=measure_mask&(w>1e-9)
        unit[core_handles]=v[core_handles,axis]-anchor
        delta[:,axis]=unit*signed
    base_extent=float(np.ptp(v[measure_mask,axis]))
    if mode=='core_extent':
        # Solve against a FIXED vertex set; never reselect using deformed bounds.
        # A bounded root solve avoids silently accepting a 0.4% edit for a 5% request.
        target_extent=base_extent*(1+signed)
        if base_extent<1e-10: raise error('测量区域在调整轴上退化',5)
        def extent(s):return float(np.ptp(v[measure_mask,axis]+unit[measure_mask]*s))
        grid=np.linspace(0, .30 if signed>0 else -.30, 121)
        bracket=None
        for a,b in zip(grid[:-1],grid[1:]):
            if (extent(a)-target_extent)*(extent(b)-target_extent)<=0:
                bracket=(a,b);break
        if bracket is None:
            raise error('尺寸目标在安全缩放上限30%内不可达；固定区/测量区域可能冲突，未放宽质量门槛',9)
        a,b=bracket
        for _ in range(55):
            mid=(a+b)/2
            if (extent(a)-target_extent)*(extent(mid)-target_extent)<=0:b=mid
            else:a=mid
        coefficient=(a+b)/2
        delta[:,axis]=unit*coefficient
    solver=h.get('deformation_solver','weighted')
    if solver not in ('weighted','harmonic_transition'):raise error('deformation_solver 须为 weighted/harmonic_transition',5)
    if solver=='harmonic_transition':
        from scipy.sparse import coo_matrix,diags
        from scipy.sparse.linalg import spsolve
        edges=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
        lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        strength=1/np.maximum(lengths,max(np.median(lengths)*.05,1e-12))
        rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
        graph=coo_matrix((np.r_[strength,strength],(rows,cols)),shape=(len(v),)*2).tocsr()
        degree=np.asarray(graph.sum(1)).ravel();L=diags(degree)-graph
        fixed=(w<=1e-9)|measure_mask;free=~fixed
        if free.any():
            # Positive screened graph Laplacian: preserve exact core/far-field
            # handles, smooth only the transition. No face/UV changes.
            regularizer=1e-4*np.maximum(degree[free],1)
            A=L[free][:,free]+diags(regularizer)
            rhs=-(L[free][:,fixed]@delta[fixed,axis])+regularizer*delta[free,axis]
            delta[free,axis]=spsolve(A,rhs)
            if not np.isfinite(delta).all():raise error('平滑过渡求解失败',9)
    if np.any(group_sizes>1):
        for coordinate in range(3):
            averaged=np.bincount(position_groups,weights=delta[:,coordinate])/group_sizes
            delta[:,coordinate]=averaged[position_groups]
    measured_extent=float(np.ptp((v+delta)[measure_mask,axis]))
    measured_change=measured_extent/max(base_extent,1e-15)-1
    if mode=='core_extent' and abs(measured_change-signed)>1e-6:
        raise error('尺寸闭环误差超过容限，未输出模型',9)
    if np.max(np.linalg.norm(delta,axis=1))<1e-10: raise error('选区没有有效尺寸变化',5)
    meta={'long_axis':'xyz'[long], 'lateral_axis':'xyz'[lateral], 'up_axis':'xyz'[up],
          'axes_confirmed':True, 'selection_method':method,
          'requires_region_confirmation':method=='axis_geometric_proposal',
          'selected_vertices':int(selected.sum()), 'selected_fraction':float(selected.mean()),
          'edit_axis':'xyz'[axis], 'anchor':anchor, 'preserved_vertices':int(protected.sum()),
          'requested_relative_change':signed,
          'deformation_solver':solver,
          'coincident_position_groups_tied':int(np.count_nonzero(group_sizes>1)),
          'dimension_control':{'mode':mode,'measurement_axis':'xyz'[axis],
              'fixed_measurement_vertices':int(measure_mask.sum()),'core_threshold':threshold,
              'before':base_extent,'after':measured_extent,'achieved_relative_change':measured_change,
              'error_relative_change':measured_change-signed,'applied_scale_coefficient':coefficient},
          'magnitude_semantics':'maximum local weighted affine scale; not a guarantee of whole-region bounding-box change',
          'self_intersections_checked':False}
    if automatic is not None:
        meta['automatic_localization']=automatic
    if method=='axis_geometric_proposal' and not h.get('accept_geometric_selection',False):
        meta['requires_region_confirmation']=True
    else: meta['requires_region_confirmation']=False
    return delta,w,meta
