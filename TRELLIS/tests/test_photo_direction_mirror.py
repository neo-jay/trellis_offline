"""Metamorphic check: mirroring the photograph must not reverse the 3D bow."""
import sys,json,shutil,subprocess
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1]);baseline=root/'photo_joint_direction_v3';out=root/'photo_direction_mirror_test';out.mkdir(exist_ok=False)
source=root/'reconstruction/ship/foreground_rgba.png';photo=out/'photo_mirrored.png'
Image.open(source).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(photo)
for pattern in ('view_*.png','world_*.npy'):
 for file in baseline.glob(pattern):shutil.copy2(file,out/file.name)
command=[sys.executable,'/root/TRELLIS/tools/build_photo_ship_direction.py',
 '--mesh',str(root/'reconstruction/ship/sample.obj'),'--photo',str(photo),'--output-dir',str(out)]
with (out/'run.log').open('w') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
before=json.loads((baseline/'direction_evidence.json').read_text())
after=json.loads((out/'direction_evidence.json').read_text()) if (out/'direction_evidence.json').exists() else {}
record={'command':command,'returncode':run.returncode,'original_status':before['status'],
 'mirrored_status':after.get('status'),'original_bow_sign':before.get('bow_sign'),
 'mirrored_bow_sign':after.get('bow_sign'),'passed':after.get('status')=='ok' and after.get('bow_sign')==before.get('bow_sign')}
(out/'mirror_test_result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
