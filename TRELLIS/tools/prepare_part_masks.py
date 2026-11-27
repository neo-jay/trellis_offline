"""Convert verified PartField cluster IDs into fingerprinted soft vertex masks.
This tool never assigns semantic names automatically: mapping must be verified.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trellis'/'utils'))
import mesh_text_edit as edit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-obj',required=True)
    p.add_argument('--face-labels',required=True)
    p.add_argument('--part-clusters-json',required=True,help='JSON file mapping part names to verified cluster IDs')
    p.add_argument('--output',required=True)
    p.add_argument('--smooth-iterations',type=int,default=8)
    a=p.parse_args()
    if not 0<=a.smooth_iterations<=100:p.error('smooth-iterations must be 0..100')
    out=Path(a.output)
    if out.exists():p.error('output already exists')
    data=edit._read_obj(Path(a.input_obj))
    labels=np.load(a.face_labels,allow_pickle=False)
    if labels.shape!=(len(data.faces),):p.error('face labels must match original OBJ face order/count')
    mapping=json.loads(Path(a.part_clusters_json).read_text())
    edges=np.vstack((data.faces[:,[0,1]],data.faces[:,[1,2]],data.faces[:,[2,0]]))
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    r=np.r_[edges[:,0],edges[:,1]]; c=np.r_[edges[:,1],edges[:,0]]
    graph=coo_matrix((np.ones(len(r)),(r,c)),shape=(len(data.vertices),)*2).tocsr()
    degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    masks={}
    for part, ids in mapping.items():
        if not isinstance(ids,list) or not ids or not set(ids)<=set(labels.tolist()):p.error('unknown or empty cluster IDs: '+part)
        w=np.zeros(len(data.vertices));w[np.unique(data.faces[np.isin(labels,ids)])]=1
        for _ in range(a.smooth_iterations): w=.5*w+.5*(graph@w)/degree
        masks[part]=w
    masks['mesh_geometry_sha256']=np.array(hashlib.sha256(np.asarray(data.vertices,'<f8').tobytes()+np.asarray(data.faces,'<i8').tobytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**masks)
    print(json.dumps({'output':str(out),'parts':list(mapping),'semantic_mapping':'caller_verified','face_order_must_match':True}))


if __name__=='__main__':main()
