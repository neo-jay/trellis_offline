"""Isolated MeshLab self-intersection face detection (no mesh mutation)."""
import json, sys
import numpy as np
import pymeshlab


def selected(vertices,faces):
    ms=pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces))
    ms.compute_selection_by_self_intersections_per_face()
    return np.flatnonzero(ms.current_mesh().face_selection_array()).tolist()


if __name__=='__main__':
    with np.load(sys.argv[1],allow_pickle=False) as a:
        before=selected(a['before'],a['faces'])
        after=selected(a['after'],a['faces'])
    result={'backend':'pymeshlab.compute_selection_by_self_intersections_per_face',
        'checked':True,'before_faces':len(before),'after_faces':len(after),
        'new_intersecting_face_indices':sorted(set(after)-set(before)),
        'note':'face-set comparison, not exhaustive intersection-pair comparison; backend adjacency/tolerance rules apply'}
    with open(sys.argv[2],'w') as f:json.dump(result,f)
