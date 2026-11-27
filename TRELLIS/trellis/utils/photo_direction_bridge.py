"""Invoke the photo-direction worker offline and reuse content-bound evidence."""
import sys
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np


def asset_hash(source):
    import shlex
    files=[Path(source)]
    for line in files[0].read_text().splitlines():
        if line.startswith('mtllib '):
            material=files[0].parent/line.split(None,1)[1].strip();files.append(material)
            for text in material.read_text().splitlines():
                tokens=shlex.split(text)
                if tokens and (tokens[0].startswith('map_') or tokens[0] in ('bump','norm','disp')):
                    files.append(material.parent/tokens[-1])
    digest=hashlib.sha256()
    for file in files:digest.update(file.read_bytes())
    return digest.hexdigest()


def resolve(v,f,h,error):
    source=Path(h.get('_source_obj_path',''));photo=Path(h['source_photo']).expanduser().resolve()
    if not source.is_file() or not photo.is_file():raise error('照片辅助方向判别需要可读取的OBJ和原始照片',5)
    geometry=hashlib.sha256(np.asarray(v,'<f8').tobytes()+np.asarray(f,'<i8').tobytes()).hexdigest()
    photo_hash=hashlib.sha256(photo.read_bytes()).hexdigest();appearance=asset_hash(source)
    key=hashlib.sha256((geometry+photo_hash+appearance+'photo_direction_v1').encode()).hexdigest()
    parent=Path(h.get('photo_direction_cache_dir',os.environ.get('TRELLIS_PHOTO_DIRECTION_CACHE',str(Path(tempfile.gettempdir())/'trellis_photo_direction'))))
    folder=parent/key;record=folder/'direction_evidence.json'
    if not record.is_file():
        script=Path(__file__).resolve().parents[2]/'tools'/'build_photo_ship_direction.py'
        # Default to the interpreter running this process so the worker uses
        # the same environment; an explicit TRELLIS_PHOTO_PYTHON still wins.
        python=os.environ.get('TRELLIS_PHOTO_PYTHON', sys.executable)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.log').open('w') as log:
            try:run=subprocess.run([python,str(script),'--mesh',str(source),'--photo',str(photo),'--output-dir',str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
            except subprocess.TimeoutExpired:raise error('照片—模型方向配准超时，未猜测方向',5)
        if run.returncode or not record.is_file():raise error('照片—模型方向配准失败，日志：'+str(folder/'worker.log'),5)
    evidence=json.loads(record.read_text())
    if evidence.get('mesh_geometry_sha256')!=geometry or evidence.get('photo_sha256')!=photo_hash:
        raise error('照片方向缓存与输入内容不匹配',5)
    if evidence.get('mesh_appearance_sha256')!=appearance:
        raise error('照片方向缓存与模型纹理不匹配',5)
    if evidence.get('status')!='ok' or evidence.get('bow_sign') not in (-1,1):
        raise error('照片辅助方向证据不足：'+evidence.get('reason','unknown'),5)
    return evidence,str(record)