"""Run PyMeshFix in a disposable process so a native crash cannot stop Gradio."""

import sys

import numpy as np
import pymeshfix


def main(method, input_path, output_path):
    with np.load(input_path, allow_pickle=False) as data:
        vertices = np.asarray(data['vertices'], dtype=np.float64)
        faces = np.asarray(data['faces'], dtype=np.int32)
    if method == 'cap_only':
        meshfix = pymeshfix.PyTMesh(False)
        meshfix.load_array(vertices, faces)
        meshfix.fill_small_boundaries(nbe=0, refine=False)
        vertices, faces = meshfix.return_arrays()
    elif method == 'local_repair':
        meshfix = pymeshfix.MeshFix(vertices, faces)
        meshfix.repair(verbose=False, joincomp=False, remove_smallest_components=False)
        vertices, faces = meshfix.v, meshfix.f
    else:
        raise ValueError(f'Unknown repair method: {method}')
    np.savez(output_path, vertices=vertices, faces=faces)


if __name__ == '__main__':
    main(*sys.argv[1:])
