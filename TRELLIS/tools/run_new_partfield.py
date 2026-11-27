"""Extract fresh PartField candidates for the newly reconstructed real photos."""
import os,sys,json,subprocess,shutil,time
from pathlib import Path
root=Path(sys.argv[1]);pf=Path('/root/autodl-tmp/PartField')
source=root/'partfield_inputs';source.mkdir(exist_ok=True)
is_ground=(root/'samples.json').exists()
rows=json.loads((root/('samples.json' if is_ground else 'reconstruction_summary.json')).read_text())
for row in rows:
 if row.get('status','ok')=='ok':shutil.copy2(row['obj'],source/(row['name']+'.obj'))
python='/root/autodl-tmp/conda-envs/partfield/bin/python'
tag=root.name
commands=[
 [python,'/root/TRELLIS/tools/partfield_single_gpu_inference.py','-c','configs/final/demo.yaml','--opts',
  'continue_ckpt','model/model_objaverse.ckpt','result_name','partfield_features/'+tag,
  'dataset.data_path',str(source),'n_point_per_face','200' if is_ground else '500'],
 [python,'run_part_clustering.py','--root','exp_results/partfield_features/'+tag,
  '--dump_dir','exp_results/clustering/'+tag,'--source_dir',str(source),
  '--use_agglo','True','--max_num_clusters','30' if is_ground else '20','--option','1','--with_knn','True']]
records=[]
compat=root/'compat_cudnn';compat.mkdir(exist_ok=True)
for lib in Path('/root/autodl-tmp/conda-envs/gcity/lib/python3.11/site-packages/torch/lib').glob('libcudnn*.so*'):
 target=compat/lib.name
 if not target.exists():target.symlink_to(lib)
environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 LD_PRELOAD=str(compat/'libcudnn.so.9'),
                 LD_LIBRARY_PATH=str(compat)+':'+os.environ.get('LD_LIBRARY_PATH',''))
for index,command in enumerate(commands):
 path=root/('partfield_compat_stage_'+str(index)+'.log');t=time.time()
 print('START',command,flush=True)
 with path.open('w') as log:
  run=subprocess.run(command,cwd=pf,stdout=log,stderr=subprocess.STDOUT,
      env=environment)
 records.append({'command':command,'returncode':run.returncode,'elapsed_seconds':time.time()-t,'log':str(path),
                 'runtime_overrides':{k:environment[k] for k in ('LD_PRELOAD','LD_LIBRARY_PATH')}})
 (root/'partfield_run_compat.json').write_text(json.dumps(records,indent=2))
 print('END',records[-1],flush=True)
 if run.returncode:raise SystemExit(run.returncode)
